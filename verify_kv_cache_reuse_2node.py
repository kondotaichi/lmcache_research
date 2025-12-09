#!/usr/bin/env python3
"""
2ノード構成（prefill/decode分離）でのKVキャッシュ再利用確認スクリプト

このスクリプトは以下のステップを実行します：
1. 最初のプロンプトをprefillサーバー（192.168.110.13）に送信し、KVキャッシュを生成・保存
2. KVキャッシュをdecodeサーバー（192.168.110.97）に転送
3. 2回目のプロンプト（共通プレフィックスあり）をdecodeサーバーに送信し、
   KVキャッシュが再利用されているかを確認
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import requests

# transformersはオプショナル（トークン数計算に使用）
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None  # type: ignore

# ==== 環境設定 ====
PREFILL_URL = os.environ.get("PREFILL_URL", "http://192.168.110.13:8010/v1/completions")
DECODE_URL = os.environ.get("DECODE_URL", "http://192.168.110.97:8011/v1/completions")
PREFILL_METRICS_URL = os.environ.get("PREFILL_METRICS_URL", "http://192.168.110.13:8010/metrics")
DECODE_METRICS_URL = os.environ.get("DECODE_METRICS_URL", "http://192.168.110.97:8011/metrics")

MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", MODEL_NAME)
PREFILL_MAX_TOKENS = int(os.environ.get("PREFILL_MAX_TOKENS", "1"))  # prefillでは1トークンだけ生成

# デフォルトのプロンプト設定
# 公式ドキュメントの例に合わせ、同一フレーズを繰り返して長文化し、chunk_size(256)超を確実にする
LONG_PROMPT_MULTIPLIER = int(os.environ.get("LONG_PROMPT_MULTIPLIER", "20"))
PROMPT_PHRASE = os.environ.get(
    "PROMPT_PHRASE",
    "Explain the significance of KV cache in language models."
)

# BASE_PROMPT は PHRASE の繰り返しで生成（BASE_PROMPT を環境変数で直接指定した場合はそちらを優先）
_ENV_BASE_PROMPT = os.environ.get("BASE_PROMPT", None)
if _ENV_BASE_PROMPT:
    BASE_PROMPT = _ENV_BASE_PROMPT
else:
    BASE_PROMPT = " ".join([PROMPT_PHRASE] * LONG_PROMPT_MULTIPLIER)

# 2回目も同一プロンプトを送る（完全一致でプレフィックス一致を保証）
FOLLOW_UP_PROMPT = os.environ.get(
    "FOLLOW_UP_PROMPT",
    BASE_PROMPT
)

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
# P2P転送や非同期保存の遅延を考慮し、長めに待機（デフォルト10秒）
WAIT_TIME_BETWEEN_REQUESTS = float(os.environ.get("WAIT_TIME_BETWEEN_REQUESTS", "10.0"))  # 2ノード構成では長めに待機

# サーバーログファイルのパス（オプション）
PREFILL_LOG_FILE = os.environ.get("PREFILL_LOG_FILE", None)
DECODE_LOG_FILE = os.environ.get("DECODE_LOG_FILE", None)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# グローバル
TOKENIZER: Optional[AutoTokenizer] = None


def init_tokenizer():
    """トークナイザーを初期化する"""
    global TOKENIZER
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("transformersライブラリが利用できません。トークン数計算はスキップされます。")
        return
    if TOKENIZER is None:
        logger.info("Loading tokenizer...")
        TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        logger.info("Tokenizer ready")


def get_prometheus_metrics(metrics_url: str) -> Dict[str, float]:
    """PrometheusメトリクスエンドポイントからLMCache関連のメトリクスを取得する"""
    try:
        resp = requests.get(metrics_url, timeout=10)
        if resp.status_code != 200:
            logger.warning("Failed to fetch metrics from %s: HTTP %d", metrics_url, resp.status_code)
            return {}

        metrics = {}
        for line in resp.text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "lmcache:" in line:
                match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)', line)
                if match:
                    metric_name = match.group(1)
                    metric_value = float(match.group(2))
                    if metric_name not in metrics:
                        metrics[metric_name] = metric_value
                    else:
                        if "total" in metric_name.lower() or "count" in metric_name.lower():
                            metrics[metric_name] += metric_value
                        else:
                            metrics[metric_name] = metric_value

        return metrics
    except Exception as exc:
        logger.warning("Error fetching metrics from %s: %s", metrics_url, exc)
        return {}


def extract_metric_value(metrics: Dict[str, float], metric_name: str) -> Optional[float]:
    """メトリクス辞書から特定のメトリクス値を取得する"""
    if metric_name in metrics:
        return metrics[metric_name]
    
    total_name = metric_name + "_total"
    if total_name in metrics:
        return metrics[total_name]
    
    for key, value in metrics.items():
        if key.startswith(metric_name):
            return value
    
    return None


def run_prefill(prompt: str) -> Tuple[float, str]:
    """
    Prefillサーバーでプロンプトを処理し、KVキャッシュを生成
    
    Returns:
        Tuple[float, str]: (処理時間, 生成されたテキスト)
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": PREFILL_MAX_TOKENS,
        "stream": True,
        "temperature": 0.0,
    }
    
    t0 = time.perf_counter()
    generated = ""
    chunks = []
    
    logger.info("Prefillサーバーにリクエスト送信中: %s", PREFILL_URL)
    logger.info("プロンプト (first 100 chars): %s...", prompt[:100])
    
    with requests.post(PREFILL_URL, json=payload, stream=True, timeout=600) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"Prefill HTTP {resp.status_code}: {resp.text[:200]}")
        
        chunk_count = 0
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if not raw.startswith("data: "):
                continue
            if raw.strip() == "data: [DONE]":
                break
            
            try:
                data_str = raw.replace("data: ", "").strip()
                if not data_str:
                    continue
                chunk = json.loads(data_str)
                chunks.append(chunk)
                chunk_count += 1
                
                choice = (chunk.get("choices") or [None])[0]
                if choice:
                    delta = choice.get("delta") or {}
                    text = delta.get("text", "")
                    if text:
                        generated += text
                
                # 最初のチャンクが来たら時間を記録して返す
                if chunk_count == 1:
                    elapsed = time.perf_counter() - t0
                    return elapsed, generated
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON chunk: %s", exc)
            except Exception as exc:
                logger.warning("Error processing chunk: %s", exc)
    
    if chunk_count == 0:
        raise RuntimeError("Prefill produced no tokens (no chunks received)")
    
    elapsed = time.perf_counter() - t0
    return elapsed, generated


def run_decode(prompt: str, max_tokens: int) -> Tuple[str, float, float, Dict]:
    """
    Decodeサーバーでプロンプトを処理し、生成を行う
    
    Returns:
        Tuple[str, float, float, Dict]: (生成されたテキスト, TTFT, TBT, usage情報)
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    last_token_time = None
    generated = ""
    chunks = []
    
    logger.info("Decodeサーバーにリクエスト送信中: %s", DECODE_URL)
    logger.info("プロンプト (first 100 chars): %s...", prompt[:100])
    
    with requests.post(DECODE_URL, json=payload, stream=True, timeout=600) as resp:
        if resp.status_code != 200:
            error_text = resp.text[:500] if hasattr(resp, "text") else str(resp)
            raise RuntimeError(f"Decode HTTP {resp.status_code}: {error_text}")
        
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
                
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                
                choice = (chunk.get("choices") or [None])[0]
                if choice:
                    delta = choice.get("delta") or {}
                    text = delta.get("text", "")
                    if not text:
                        text = choice.get("text", "")
                    
                    if text:
                        generated += text
                        last_token_time = time.perf_counter()
            
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON chunk: %s", exc)
            except Exception as exc:
                logger.warning("Error processing chunk: %s", exc)
    
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else None
    tbt_ms = (last_token_time - start_time) * 1000 if last_token_time else None
    
    usage = {}
    if chunks:
        last_chunk = chunks[-1]
        usage = last_chunk.get("usage") or {}
        if not usage:
            for chunk in reversed(chunks):
                if chunk.get("usage"):
                    usage = chunk.get("usage")
                    break
    
    return generated, ttft_ms, tbt_ms, usage


def parse_lmcache_log_line(line: str) -> Optional[Dict[str, int]]:
    """vLLMサーバーログからLMCacheのキャッシュヒット情報を抽出する"""
    pattern = r'LMCache INFO: Reqid: ([^,]+), Total tokens (\d+), LMCache hit tokens: ([\d]+|None), need to load: (-?\d+)'
    match = re.search(pattern, line)
    if match:
        hit_raw = match.group(3)
        try:
            hit_val = int(hit_raw)
        except Exception:
            hit_val = 0
        need_raw = match.group(4)
        try:
            need_val = int(need_raw)
        except Exception:
            need_val = 0
        return {
            'req_id': match.group(1),
            'total_tokens': int(match.group(2)),
            'hit_tokens': hit_val,
            'need_to_load': need_val,
        }
    return None


def extract_lmcache_logs_recent(log_file: str, lines: int = 100) -> List[Dict[str, int]]:
    """ログファイルの最後のN行からLMCacheのキャッシュヒット情報を抽出する"""
    results = []
    try:
        if not os.path.exists(log_file):
            logger.warning("ログファイルが存在しません: %s", log_file)
            return results
        
        file_size = os.path.getsize(log_file)
        if file_size == 0:
            logger.warning("ログファイルが空です: %s", log_file)
            return results
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            for line in recent_lines:
                parsed = parse_lmcache_log_line(line)
                if parsed:
                    results.append(parsed)
    except Exception as exc:
        logger.warning("ログファイルの読み込みエラー: %s", exc)
    return results


def calculate_common_prefix_length(text1: str, text2: str) -> int:
    """2つのテキストの共通プレフィックスの長さを計算する"""
    min_len = min(len(text1), len(text2))
    for i in range(min_len):
        if text1[i] != text2[i]:
            return i
    return min_len


def main():
    """メイン処理"""
    logger.info("=" * 80)
    logger.info("2ノード構成（prefill/decode分離）でのKVキャッシュ再利用確認スクリプト")
    logger.info("=" * 80)
    logger.info("Prefillサーバー: %s", PREFILL_URL)
    logger.info("Decodeサーバー: %s", DECODE_URL)
    logger.info("長さ補強倍率 (LONG_PROMPT_MULTIPLIER): %d", LONG_PROMPT_MULTIPLIER)
    
    # transformers が無い場合やプロンプトが短い場合は強制的に長文化
    if not TRANSFORMERS_AVAILABLE:
        if LONG_PROMPT_MULTIPLIER > 1:
            logger.warning("transformersなしのため、BASE_PROMPTを %dx 繰り返して長文化します", LONG_PROMPT_MULTIPLIER)
            globals()['BASE_PROMPT'] = BASE_PROMPT * LONG_PROMPT_MULTIPLIER
            globals()['FOLLOW_UP_PROMPT'] = FOLLOW_UP_PROMPT * LONG_PROMPT_MULTIPLIER
    else:
        # transformersがあっても明示的に倍率を指定されたら長文化
        if LONG_PROMPT_MULTIPLIER > 1:
            logger.info("BASE_PROMPTを %dx 繰り返して長文化します", LONG_PROMPT_MULTIPLIER)
            globals()['BASE_PROMPT'] = BASE_PROMPT * LONG_PROMPT_MULTIPLIER
            globals()['FOLLOW_UP_PROMPT'] = FOLLOW_UP_PROMPT * LONG_PROMPT_MULTIPLIER
    
    init_tokenizer()
    
    # プロンプトのトークン数を計算
    base_tokens = None
    follow_up_tokens = None
    common_prefix_tokens = None
    if TOKENIZER is not None:
        base_tokens = len(TOKENIZER.encode(BASE_PROMPT))
        follow_up_tokens = len(TOKENIZER.encode(FOLLOW_UP_PROMPT))
        common_prefix_tokens = len(TOKENIZER.encode(BASE_PROMPT[:calculate_common_prefix_length(BASE_PROMPT, FOLLOW_UP_PROMPT)]))
        logger.info("\n【プロンプト情報】")
        logger.info("Base prompt: %d chars (%d tokens)", len(BASE_PROMPT), base_tokens)
        logger.info("Follow-up prompt: %d chars (%d tokens)", len(FOLLOW_UP_PROMPT), follow_up_tokens)
        logger.info("Common prefix: %d tokens", common_prefix_tokens)
        
        chunk_size = 256
        logger.info("\n【重要】LMCacheのchunk_size: %d トークン", chunk_size)
        if base_tokens < chunk_size:
            logger.warning("⚠ 警告: 1回目のプロンプトが短すぎます（%d < %d トークン）", base_tokens, chunk_size)
        else:
            logger.info("✓ 1回目のプロンプトはchunk_size以上です（%d >= %d トークン）", base_tokens, chunk_size)
    else:
        logger.info("トークン数計算はスキップされます（transformersが利用できません）")
    
    # ==== ステップ1: PrefillサーバーでKVキャッシュを生成 ====
    logger.info("\n" + "=" * 80)
    logger.info("ステップ1: PrefillサーバーでKVキャッシュを生成")
    logger.info("=" * 80)
    
    # Prefillサーバーのベースラインメトリクス
    logger.info("Prefillサーバーのベースラインメトリクスを取得中...")
    prefill_baseline_metrics = get_prometheus_metrics(PREFILL_METRICS_URL)
    
    # Prefillリクエストを送信
    logger.info("\nPrefillサーバーにリクエスト送信中...")
    prefill_time, prefill_response = run_prefill(BASE_PROMPT)
    
    logger.info("Prefillリクエスト完了:")
    logger.info("  処理時間: %.2f ms", prefill_time * 1000)
    logger.info("  生成されたテキスト: %s", prefill_response)
    logger.info("\n--- 1回目のプロンプト（完全） ---")
    logger.info("%s", BASE_PROMPT)
    
    # KVキャッシュ転送の待機時間
    logger.info("\nKVキャッシュの転送と保存のため %f 秒待機中...", WAIT_TIME_BETWEEN_REQUESTS)
    time.sleep(WAIT_TIME_BETWEEN_REQUESTS)
    
    # ==== ステップ2: Decodeサーバーでキャッシュ再利用を確認 ====
    logger.info("\n" + "=" * 80)
    logger.info("ステップ2: DecodeサーバーでKVキャッシュ再利用を確認")
    logger.info("=" * 80)
    
    # Decodeサーバーのベースラインメトリクス
    logger.info("Decodeサーバーのベースラインメトリクスを取得中...")
    decode_before_metrics = get_prometheus_metrics(DECODE_METRICS_URL)
    
    # Decodeリクエストを送信（共通プレフィックスあり）
    logger.info("\nDecodeサーバーにリクエスト送信中（共通プレフィックスあり）...")
    decode_response, decode_ttft, decode_tbt, decode_usage = run_decode(FOLLOW_UP_PROMPT, MAX_TOKENS)
    
    logger.info("Decodeリクエスト完了:")
    logger.info("  TTFT: %.2f ms", decode_ttft if decode_ttft else 0)
    logger.info("  TBT: %.2f ms", decode_tbt if decode_tbt else 0)
    logger.info("  Generated tokens: %s", decode_usage.get("completion_tokens", "N/A"))
    logger.info("\n--- 2回目のプロンプト（完全） ---")
    logger.info("%s", FOLLOW_UP_PROMPT)
    logger.info("\n--- 2回目のレスポンス（完全） ---")
    logger.info("%s", decode_response)
    
    # メトリクス更新の待機時間
    logger.info("\nメトリクス更新のため %f 秒待機中...", WAIT_TIME_BETWEEN_REQUESTS)
    time.sleep(WAIT_TIME_BETWEEN_REQUESTS)
    
    # Decodeサーバーのメトリクスを再取得
    logger.info("Decodeサーバーのメトリクスを再取得中...")
    decode_after_metrics = get_prometheus_metrics(DECODE_METRICS_URL)
    
    # ==== 結果の分析 ====
    logger.info("\n" + "=" * 80)
    logger.info("結果の分析")
    logger.info("=" * 80)
    
    # メトリクス分析
    decode_retrieve_hit_rate = extract_metric_value(decode_after_metrics, "lmcache:retrieve_hit_rate")
    decode_lookup_hit_rate = extract_metric_value(decode_after_metrics, "lmcache:lookup_hit_rate")
    
    before_hit_tokens = extract_metric_value(decode_before_metrics, "lmcache:num_hit_tokens_total")
    after_hit_tokens = extract_metric_value(decode_after_metrics, "lmcache:num_hit_tokens_total")
    hit_tokens = (after_hit_tokens - before_hit_tokens) if (before_hit_tokens is not None and after_hit_tokens is not None) else None
    
    if hit_tokens is None:
        before_hit_tokens = extract_metric_value(decode_before_metrics, "lmcache:num_hit_tokens")
        after_hit_tokens = extract_metric_value(decode_after_metrics, "lmcache:num_hit_tokens")
        hit_tokens = (after_hit_tokens - before_hit_tokens) if (before_hit_tokens is not None and after_hit_tokens is not None) else None
    
    # サーバーログからキャッシュヒット情報を取得
    server_log_hit_tokens = None
    if DECODE_LOG_FILE:
        recent_logs = extract_lmcache_logs_recent(DECODE_LOG_FILE, lines=10)
        if recent_logs:
            server_log_hit_tokens = recent_logs[-1]['hit_tokens'] if recent_logs else None
            logger.info("\n--- Decodeサーバーログからの情報 ---")
            for log_entry in recent_logs[-1:]:
                logger.info("  ReqID: %s", log_entry['req_id'])
                logger.info("  Total tokens: %d", log_entry['total_tokens'])
                logger.info("  LMCache hit tokens: %d", log_entry['hit_tokens'])
                logger.info("  Need to load: %d", log_entry['need_to_load'])
    
    logger.info("\n--- メトリクス結果 ---")
    logger.info("Decodeサーバー - retrieve_hit_rate: %s", decode_retrieve_hit_rate)
    logger.info("Decodeサーバー - lookup_hit_rate: %s", decode_lookup_hit_rate)
    logger.info("Decodeサーバー - num_hit_tokens (差分): %s", hit_tokens)
    
    # キャッシュヒットの確認
    logger.info("\n--- キャッシュヒット確認 ---")
    cache_hit_detected = False
    
    if server_log_hit_tokens is not None:
        if server_log_hit_tokens > 0:
            logger.info("✓ サーバーログ: LMCache hit tokens = %d (キャッシュが正しく再利用されています)", server_log_hit_tokens)
            cache_hit_detected = True
        else:
            logger.warning("✗ サーバーログ: LMCache hit tokens = 0 (キャッシュヒットがありません)")
    
    if decode_retrieve_hit_rate is not None and decode_retrieve_hit_rate > 0:
        logger.info("✓ retrieve_hit_rate > 0: KVキャッシュが再利用されています (%.2f%%)", decode_retrieve_hit_rate * 100)
        cache_hit_detected = True
    
    if hit_tokens is not None and hit_tokens > 0:
        logger.info("✓ num_hit_tokens > 0: %.0f トークンがキャッシュから取得されました", hit_tokens)
        cache_hit_detected = True
    
    # パフォーマンス比較
    logger.info("\n--- パフォーマンス情報 ---")
    logger.info("Prefill処理時間: %.2f ms", prefill_time * 1000)
    logger.info("Decode TTFT: %.2f ms", decode_ttft if decode_ttft else 0)
    logger.info("Decode TBT: %.2f ms", decode_tbt if decode_tbt else 0)
    
    # レスポンス内容の確認
    logger.info("\n--- レスポンス内容の確認 ---")
    logger.info("Decodeサーバーのレスポンスが、Prefillサーバーで処理したプロンプトの内容を")
    logger.info("考慮しているか確認:")
    
    if "1960s" in decode_response or "1970s" in decode_response:
        logger.info("✓ レスポンスは2回目のプロンプトの質問（1960s-1970s）に適切に応答しています")
    
    # 最終結果
    logger.info("\n" + "=" * 80)
    logger.info("最終結果")
    logger.info("=" * 80)
    if cache_hit_detected:
        logger.info("✓ 2ノード間でのKVキャッシュの再利用が確認されました")
        logger.info("  - Prefillサーバー（%s）で生成されたKVキャッシュが", PREFILL_URL)
        logger.info("  - Decodeサーバー（%s）で正しく再利用されています", DECODE_URL)
    else:
        logger.warning("✗ KVキャッシュの再利用が確認されませんでした")
        logger.warning("  以下を確認してください:")
        logger.warning("  1. LMCacheのP2P設定が正しいか")
        logger.warning("  2. Controllerが正しく動作しているか")
        logger.warning("  3. KVキャッシュの転送が完了しているか（待機時間を増やす）")
        logger.warning("  4. Decodeサーバーのログで 'LMCache hit tokens: X' を確認")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.error("Script failed: %s", exc, exc_info=True)
        raise

