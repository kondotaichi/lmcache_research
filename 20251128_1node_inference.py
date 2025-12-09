#!/usr/bin/env python3
"""
1ノード構成 (単一 vLLM エンドポイント) + LMCache ベンチマーク。
2node スクリプトと同じ指標 / 出力形式を維持しつつ、prefill/decode 分離の代わりに
1本の推論リクエストで計測する。
"""

import json
import logging
import os
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from transformers import AutoTokenizer, AutoConfig

# ==== 環境設定 ====
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8010/v1/completions")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", MODEL_NAME)
GENERATION_LENGTHS = [int(x.strip()) for x in os.environ.get("GENERATION_LENGTHS", "128,256,512").split(",")]
CONCURRENCY_LEVELS = [int(x.strip()) for x in os.environ.get("CONCURRENCY_LEVELS", "1,2,3").split(",")]
CACHE_CLEAR_METHOD = os.environ.get("CACHE_CLEAR_METHOD", "unique_prompt")
BASE_PROMPT_TEXT = os.environ.get("BASE_PROMPT_TEXT", "Write a story about a cat.")
BASE_PROMPT = BASE_PROMPT_TEXT * 100

# KV 1トークンあたりのバイト数（環境変数またはモデル設定から算出）
KV_BYTES_PER_TOKEN: Optional[float] = None
_KV_BYTES_PER_TOKEN_ENV = os.environ.get("KV_BYTES_PER_TOKEN")
_KV_DTYPE_BYTES_ENV = os.environ.get("KV_DTYPE_BYTES")  # 1要素あたりのバイト数 (デフォルト: 2=fp16/bf16)

CONTROLLER_URL = os.environ.get("CONTROLLER_URL", "http://localhost:9000")
LMCACHE_INSTANCE_ID = os.environ.get("LMCACHE_INSTANCE_ID", "lmcache_default_instance")
CLEAR_CACHE_LOCATION = os.environ.get("CLEAR_CACHE_LOCATION", "all")

LOG_DIR = Path(os.environ.get("BENCHMARK_DIR", "benchmark_lmcache_logs"))
LOG_DIR.mkdir(exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"benchmark_1node_{TIMESTAMP}.log"
RESULTS_FILE = LOG_DIR / f"results_1node_{TIMESTAMP}.json"
GPU_LOG_FILE = LOG_DIR / f"gpu_dmon_{TIMESTAMP}.log"
CPU_LOG_FILE = LOG_DIR / f"cpu_sar_{TIMESTAMP}.log"
RESOURCE_MONITOR_INTERVAL = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# グローバル
TOKENIZER: Optional[AutoTokenizer] = None
GPU_PROC: Optional[subprocess.Popen] = None
CPU_PROC: Optional[subprocess.Popen] = None
CPU_FALLBACK_THREAD: Optional[threading.Thread] = None
CPU_FALLBACK_RUNNING = False
PATTERN_COUNTER = 0


def init_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        logger.info("Loading tokenizer...")
        TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        logger.info("Tokenizer ready")


def init_kv_bytes_per_token():
    """KV 1トークンあたりのバイト数を初期化する。

    優先順:
      1) 環境変数 KV_BYTES_PER_TOKEN で明示指定
      2) transformers の AutoConfig から (layers * hidden_size * 2(K+V) * bytes_per_elem) を推定
    """
    global KV_BYTES_PER_TOKEN
    if KV_BYTES_PER_TOKEN is not None:
        return

    # 1) 環境変数での明示指定
    if _KV_BYTES_PER_TOKEN_ENV is not None:
        try:
            KV_BYTES_PER_TOKEN = float(_KV_BYTES_PER_TOKEN_ENV)
            logger.info("Using KV_BYTES_PER_TOKEN from env: %s bytes/token", KV_BYTES_PER_TOKEN)
            return
        except ValueError:
            logger.warning("Invalid KV_BYTES_PER_TOKEN env value: %s", _KV_BYTES_PER_TOKEN_ENV)

    # 2) モデル設定からの推定
    try:
        cfg = AutoConfig.from_pretrained(MODEL_NAME)
        hidden_size = getattr(cfg, "hidden_size", None)
        num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layers", None)
        if hidden_size is None or num_layers is None:
            logger.warning("Cannot infer KV_BYTES_PER_TOKEN (missing hidden_size or num_layers in config)")
            return
        try:
            dtype_bytes = int(_KV_DTYPE_BYTES_ENV) if _KV_DTYPE_BYTES_ENV is not None else 2
        except ValueError:
            logger.warning("Invalid KV_DTYPE_BYTES env value: %s, fallback to 2", _KV_DTYPE_BYTES_ENV)
            dtype_bytes = 2
        # KVサイズ ≒ 2(K+V) * num_layers * hidden_size * bytes_per_elem
        KV_BYTES_PER_TOKEN = 2 * num_layers * hidden_size * dtype_bytes
        logger.info(
            "Estimated KV_BYTES_PER_TOKEN=%s bytes/token (layers=%s, hidden_size=%s, bytes/elem=%s)",
            KV_BYTES_PER_TOKEN,
            num_layers,
            hidden_size,
            dtype_bytes,
        )
    except Exception as exc:
        logger.warning("Failed to estimate KV_BYTES_PER_TOKEN from model config: %s", exc)


def calc_percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    idx = (p / 100.0) * (len(values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    w = idx - lo
    return values[lo] * (1 - w) + values[hi] * w


def create_prompt(experiment_id: str) -> str:
    if CACHE_CLEAR_METHOD in {"unique_prompt", "both"}:
        return f"[ExperimentID: {experiment_id}]\n\n{BASE_PROMPT}"
    return BASE_PROMPT


def clear_cache_controller():
    if CACHE_CLEAR_METHOD not in {"controller", "both"}:
        return
    try:
        resp = requests.post(
            f"{CONTROLLER_URL}/clear",
            json={"instance_id": LMCACHE_INSTANCE_ID, "location": CLEAR_CACHE_LOCATION},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("Controller clear failed: %s %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("Controller clear exception: %s", exc)


def start_resource_monitors():
    global GPU_PROC, CPU_PROC, CPU_FALLBACK_THREAD, CPU_FALLBACK_RUNNING
    try:
        GPU_PROC = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "u", "-d", str(RESOURCE_MONITOR_INTERVAL)],
            stdout=open(GPU_LOG_FILE, "w", encoding="utf-8"),
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info("GPU monitor -> %s", GPU_LOG_FILE)
    except Exception as exc:
        logger.warning("Failed to start nvidia-smi dmon: %s", exc)
        GPU_PROC = None
    try:
        CPU_PROC = subprocess.Popen(
            ["sar", "-u", str(RESOURCE_MONITOR_INTERVAL)],
            stdout=open(CPU_LOG_FILE, "w", encoding="utf-8"),
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info("CPU monitor (sar) -> %s", CPU_LOG_FILE)
    except Exception as exc:
        logger.warning("Failed to start sar: %s", exc)
        CPU_PROC = None
        CPU_FALLBACK_RUNNING = True
        cpu_file = open(CPU_LOG_FILE, "w", encoding="utf-8")
        cpu_file.write("timestamp,cpu_user,cpu_system,cpu_idle\n")

        def cpu_loop():
            import psutil

            while CPU_FALLBACK_RUNNING:
                cpu_file.write(f"{time.time()},{psutil.cpu_percent()},0,{100-psutil.cpu_percent()}\n")
                cpu_file.flush()
                time.sleep(RESOURCE_MONITOR_INTERVAL)
            cpu_file.close()

        CPU_FALLBACK_THREAD = threading.Thread(target=cpu_loop, daemon=True)
        CPU_FALLBACK_THREAD.start()


def stop_resource_monitors():
    global CPU_FALLBACK_RUNNING
    if GPU_PROC:
        GPU_PROC.terminate()
    if CPU_PROC:
        CPU_PROC.terminate()
    CPU_FALLBACK_RUNNING = False
    logger.info("Resource monitors stopped")


def run_inference(prompt: str, max_tokens: int) -> Tuple[Optional[float], Optional[float], int]:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    first = None
    last = None
    generated = ""
    chunks: List[Dict] = []
    with requests.post(VLLM_URL, json=payload, stream=True, timeout=600) as resp:
        if resp.status_code != 200:
            error_text = resp.text[:500] if hasattr(resp, "text") else str(resp)
            raise RuntimeError(f"Inference HTTP {resp.status_code}: {error_text}")
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            if raw.strip() == "data: [DONE]":
                break
            try:
                data_str = raw.replace("data: ", "").strip()
                if not data_str:
                    continue
                chunk = json.loads(data_str)
                chunks.append(chunk)
                if first is None:
                    first = time.perf_counter()
                choice = (chunk.get("choices") or [None])[0]
                delta = (choice or {}).get("delta") or {}
                text = delta.get("text", "")
                if text:
                    generated += text
                    last = time.perf_counter()
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON chunk: %s (raw: %s)", exc, raw[:200])
            except Exception as exc:
                logger.warning("Error processing chunk: %s (raw: %s)", exc, raw[:200])
    tokens = 0
    if chunks:
        usage = chunks[-1].get("usage") or {}
        tokens = usage.get("completion_tokens", 0) or 0
    if tokens == 0 and generated:
        init_tokenizer()
        tokens = len(TOKENIZER.encode(generated))
    return first, last, tokens


def measure_single(pattern_id: str, prompt: str, prompt_len: int, max_tokens: int, concurrency: int, request_id: int) -> Dict:
    t_req = time.perf_counter()
    status = "ok"
    error_type = None
    try:
        first, last, tokens = run_inference(prompt, max_tokens)
        ttft_ms = (first - t_req) * 1000 if first else None
        tbt_ms = (last - t_req) * 1000 if last else None
        prefill_time_ms = ttft_ms
        # 1node構成ではノード間KV転送は行わないため、ネットワーク的なkv_transfer_msは0とする
        kv_transfer_ms = 0.0
        decode_time_ms = (tbt_ms - ttft_ms) if (tbt_ms is not None and ttft_ms is not None) else None
        tokens_per_s = tokens / (decode_time_ms / 1000) if decode_time_ms and decode_time_ms > 0 else None
        # 1node構成では「KV構築スループット」として、prefill_time_ms を使ってKVスループットを算出
        kv_tokens_per_s = None
        kv_bytes_per_s = None
        if prefill_time_ms and prefill_time_ms > 0:
            kv_tokens_per_s = prompt_len / (prefill_time_ms / 1000.0)
            if KV_BYTES_PER_TOKEN is not None:
                kv_bytes_per_s = kv_tokens_per_s * KV_BYTES_PER_TOKEN
        return {
            "pattern_id": pattern_id,
            "prompt_len": prompt_len,
            "output_len": max_tokens,
            "concurrency": concurrency,
            "request_id": request_id,
            "t_req_start": t_req,
            "t_first_token": first,
            "t_last_token": last,
            "ttft_ms": ttft_ms,
            "tbt_ms": tbt_ms,
            "prefill_time_ms": prefill_time_ms,
            "kv_transfer_ms": kv_transfer_ms,
            "decode_time_ms": decode_time_ms,
            "total_tokens": tokens,
            "tokens_per_s": tokens_per_s,
            "kv_tokens_per_s": kv_tokens_per_s,
            "kv_bytes_per_s": kv_bytes_per_s,
            "status": status,
            "error_type": error_type,
        }
    except Exception as exc:
        status = "error"
        msg = str(exc)
        if "timeout" in msg.lower():
            error_type = "timeout"
        elif "500" in msg:
            error_type = "5xx"
        elif "400" in msg:
            error_type = "4xx"
        else:
            error_type = "other"
        logger.error("Request %s failed: %s", request_id, exc)
        return {
            "pattern_id": pattern_id,
            "prompt_len": prompt_len,
            "output_len": max_tokens,
            "concurrency": concurrency,
            "request_id": request_id,
            "t_req_start": t_req,
            "t_first_token": None,
            "t_last_token": None,
            "ttft_ms": None,
            "tbt_ms": None,
            "prefill_time_ms": None,
            "kv_transfer_ms": 0.0,
            "decode_time_ms": None,
            "total_tokens": 0,
            "tokens_per_s": None,
            "kv_tokens_per_s": None,
            "kv_bytes_per_s": None,
            "status": status,
            "error_type": error_type,
        }


def measure_pattern(pattern_id: str, prompt: str, prompt_len: int, max_tokens: int, concurrency: int) -> Dict:
    results: List[Dict] = []
    for rid in range(concurrency):
        results.append(measure_single(pattern_id, prompt, prompt_len, max_tokens, concurrency, rid))
    success = [r for r in results if r["status"] == "ok"]

    def avg(key):
        vals = [r[key] for r in success if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def perc(key, p):
        vals = [r[key] for r in success if r.get(key) is not None]
        return calc_percentile(vals, p)

    return {
        "pattern_id": pattern_id,
        "generation_tokens": max_tokens,
        "concurrency": concurrency,
        "num_requests": len(results),
        "success_count": len(success),
        "error_count": len(results) - len(success),
        "error_rate": (len(results) - len(success)) / len(results) if results else 0,
        "TTFT_avg_ms": avg("ttft_ms"),
        "TBT_avg_ms": avg("tbt_ms"),
        "prefill_time_avg_ms": avg("prefill_time_ms"),
        "kv_transfer_avg_ms": avg("kv_transfer_ms"),
        "decode_time_avg_ms": avg("decode_time_ms"),
        "tokens_per_s_avg": avg("tokens_per_s"),
        "kv_tokens_per_s_avg": avg("kv_tokens_per_s"),
        "kv_bytes_per_s_avg": avg("kv_bytes_per_s"),
        "TTFT_p50_ms": perc("ttft_ms", 50),
        "TTFT_p95_ms": perc("ttft_ms", 95),
        "TBT_p50_ms": perc("tbt_ms", 50),
        "TBT_p95_ms": perc("tbt_ms", 95),
        "requests": results,
    }


def run():
    global PATTERN_COUNTER
    logger.info("Starting 1-node benchmark: vllm=%s", VLLM_URL)
    init_tokenizer()
    init_kv_bytes_per_token()
    base_tokens = len(TOKENIZER.encode(BASE_PROMPT))
    start_resource_monitors()
    pattern_results: List[Dict] = []
    try:
        total = len(GENERATION_LENGTHS) * len(CONCURRENCY_LEVELS)
        done = 0
        for gen in GENERATION_LENGTHS:
            for conc in CONCURRENCY_LEVELS:
                done += 1
                PATTERN_COUNTER += 1
                pattern_id = f"P{PATTERN_COUNTER}"
                experiment_id = str(uuid.uuid4())
                logger.info("[%s/%s] %s | gen=%s | conc=%s", done, total, pattern_id, gen, conc)
                clear_cache_controller()
                prompt = create_prompt(experiment_id)
                prompt_len = len(TOKENIZER.encode(prompt))
                pattern_results.append(measure_pattern(pattern_id, prompt, prompt_len, gen, conc))
                time.sleep(1)
        json.dump(
            {
                "timestamp": TIMESTAMP,
                "config": {
                    "vllm_url": VLLM_URL,
                    "model": MODEL_NAME,
                    "base_prompt_tokens": base_tokens,
                    "generation_lengths": GENERATION_LENGTHS,
                    "concurrency_levels": CONCURRENCY_LEVELS,
                    "cache_clear_method": CACHE_CLEAR_METHOD,
                    "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
                },
                "patterns": pattern_results,
                "resource_logs": {"gpu_dmon": str(GPU_LOG_FILE), "cpu_sar": str(CPU_LOG_FILE)},
            },
            open(RESULTS_FILE, "w", encoding="utf-8"),
            indent=2,
            ensure_ascii=False,
        )
        logger.info("Results saved -> %s", RESULTS_FILE)
    finally:
        stop_resource_monitors()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        stop_resource_monitors()
    except Exception as exc:
        logger.error("Benchmark failed: %s", exc, exc_info=True)
        stop_resource_monitors()
        raise

