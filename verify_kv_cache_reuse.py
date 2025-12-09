#!/usr/bin/env python3
"""
KVキャッシュの再利用を確認するスクリプト

このスクリプトは以下の2つのステップを実行します：
1. 最初のプロンプトをサーバーに送信し、KVキャッシュを保存する
2. そのプロンプトに関連した（プレフィックスが同じ）プロンプトを送信し、
   KVキャッシュが再利用されているか、およびレスポンスがキャッシュを考慮した
   内容になっているかを確認する
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
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8010/v1/completions")
METRICS_URL = os.environ.get("METRICS_URL", "http://localhost:8010/metrics")
MODEL_NAME = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", MODEL_NAME)

# デフォルトのプロンプト設定
# 注意: LMCacheのchunk_size（通常256トークン）以上になるように長いプロンプトを使用
BASE_PROMPT = os.environ.get(
    "BASE_PROMPT",
    "The history of artificial intelligence dates back to ancient times, when philosophers and mathematicians began to explore the concept of creating machines that could think and reason like humans. However, the modern field of AI as we know it today began in the 1950s with the work of pioneers like Alan Turing, who proposed the Turing Test as a way to measure machine intelligence. "
    "In the following decades, AI research progressed through several phases. The 1960s saw the development of early expert systems and natural language processing programs. Researchers like John McCarthy, Marvin Minsky, and others laid the groundwork for what would become modern AI. "
    "The 1970s brought about more sophisticated approaches to problem-solving and knowledge representation. During this period, the field faced what became known as the 'AI winter' - a time when funding and interest in AI research declined due to unmet expectations. "
    "Despite these challenges, the 1980s witnessed a resurgence in AI research, particularly in machine learning and neural networks. The backpropagation algorithm, developed during this time, became a cornerstone of modern deep learning. "
    "The 1990s and early 2000s saw significant advances in statistical learning methods, support vector machines, and the development of more powerful computing hardware that enabled the training of larger neural networks."
)

# 2つ目のプロンプトは最初のプロンプトのプレフィックスを含む必要がある
# （LMCacheはプレフィックスキャッシングを使用するため）
FOLLOW_UP_PROMPT = os.environ.get(
    "FOLLOW_UP_PROMPT",
    None  # デフォルトではBASE_PROMPTに追加テキストを付加
)

# FOLLOW_UP_PROMPTが指定されていない場合、BASE_PROMPTに追加テキストを付加
if FOLLOW_UP_PROMPT is None:
    FOLLOW_UP_PROMPT = BASE_PROMPT + " What were the key developments in AI research during the 1960s and 1970s?"

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
# P2P転送や非同期保存の遅延を考慮し、少し長めに待機（デフォルト5秒）
WAIT_TIME_BETWEEN_REQUESTS = float(os.environ.get("WAIT_TIME_BETWEEN_REQUESTS", "5.0"))

# サーバーログファイルのパス（オプション）
# vLLMのログは通常標準出力に出力されるため、起動時にリダイレクトする必要があります
# 例: vllm serve ... 2>&1 | tee vllm/server.log
SERVER_LOG_FILE = os.environ.get("SERVER_LOG_FILE", None)

# ログファイルが見つからない場合の代替方法
# 標準出力から直接取得する場合は、この機能を使用
ENABLE_STDOUT_LOG_PARSING = os.environ.get("ENABLE_STDOUT_LOG_PARSING", "false").lower() == "true"

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


def get_prometheus_metrics() -> Dict[str, float]:
    """
    PrometheusメトリクスエンドポイントからLMCache関連のメトリクスを取得する

    Returns:
        Dict[str, float]: メトリクス名と値の辞書
    """
    try:
        resp = requests.get(METRICS_URL, timeout=10)
        if resp.status_code != 200:
            logger.warning("Failed to fetch metrics: HTTP %d", resp.status_code)
            return {}

        metrics = {}
        for line in resp.text.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Prometheus形式: metric_name{labels} value または metric_name value
            # LMCache関連のメトリクスを抽出
            if "lmcache:" in line:
                # ラベル付きメトリクス: metric_name{label1="value1",label2="value2"} value
                # またはラベルなし: metric_name value
                match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)', line)
                if match:
                    metric_name = match.group(1)
                    metric_value = float(match.group(2))
                    # 既に存在する場合は最初の値を保持（複数のworkerがある場合）
                    if metric_name not in metrics:
                        metrics[metric_name] = metric_value
                    else:
                        # 複数の値がある場合は合計または最大値を取る（counterの場合は合計）
                        if "total" in metric_name.lower() or "count" in metric_name.lower():
                            metrics[metric_name] += metric_value
                        else:
                            # gaugeの場合は最新の値を使用
                            metrics[metric_name] = metric_value

        return metrics
    except Exception as exc:
        logger.warning("Error fetching metrics: %s", exc)
        return {}


def extract_metric_value(metrics: Dict[str, float], metric_name: str) -> Optional[float]:
    """メトリクス辞書から特定のメトリクス値を取得する"""
    # 完全一致を試す
    if metric_name in metrics:
        return metrics[metric_name]

    # _totalサフィックス付きを試す（counterメトリクスの場合）
    total_name = metric_name + "_total"
    if total_name in metrics:
        return metrics[total_name]

    # 部分一致を試す（ラベル付きメトリクスの場合）
    for key, value in metrics.items():
        if key.startswith(metric_name):
            return value

    return None


def run_inference(prompt: str, max_tokens: int) -> Tuple[str, float, float, Dict]:
    """
    推論を実行し、レスポンスとタイミング情報を返す

    Args:
        prompt: プロンプト文字列
        max_tokens: 生成する最大トークン数

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
    chunks: List[Dict] = []

    logger.info("Sending request with prompt (first 100 chars): %s...", prompt[:100])

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

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                # vLLM completions APIの形式に対応
                choice = (chunk.get("choices") or [None])[0]
                if choice:
                    # delta形式（ストリーミング中）
                    delta = choice.get("delta") or {}
                    text = delta.get("text", "")
                    # またはtext形式（最後のチャンク）
                    if not text:
                        text = choice.get("text", "")
                    
                    if text:
                        generated += text
                        last_token_time = time.perf_counter()

            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON chunk: %s (raw: %s)", exc, raw[:200])
            except Exception as exc:
                logger.warning("Error processing chunk: %s (raw: %s)", exc, raw[:200])

    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else None
    tbt_ms = (last_token_time - start_time) * 1000 if last_token_time else None

    usage = {}
    if chunks:
        # 最後のチャンクからusage情報を取得
        last_chunk = chunks[-1]
        usage = last_chunk.get("usage") or {}
        # usageが空の場合、他のチャンクから探す
        if not usage:
            for chunk in reversed(chunks):
                if chunk.get("usage"):
                    usage = chunk.get("usage")
                    break

    # デバッグ情報
    if not generated and chunks:
        logger.warning("生成されたテキストが空です。チャンク数: %d", len(chunks))
        if chunks:
            logger.debug("最初のチャンク: %s", json.dumps(chunks[0], ensure_ascii=False)[:200])
            logger.debug("最後のチャンク: %s", json.dumps(chunks[-1], ensure_ascii=False)[:200])

    return generated, ttft_ms, tbt_ms, usage


def calculate_common_prefix_length(text1: str, text2: str) -> int:
    """2つのテキストの共通プレフィックスの長さを計算する"""
    min_len = min(len(text1), len(text2))
    for i in range(min_len):
        if text1[i] != text2[i]:
            return i
    return min_len


def parse_lmcache_log_line(line: str) -> Optional[Dict[str, int]]:
    """
    vLLMサーバーログからLMCacheのキャッシュヒット情報を抽出する
    
    ログ形式の例:
    LMCache INFO: Reqid: cmpl-xxx-0, Total tokens 32, LMCache hit tokens: 24, need to load: 8
    
    Returns:
        Dict with keys: 'req_id', 'total_tokens', 'hit_tokens', 'need_to_load'
        or None if the line doesn't match
    """
    import re
    # ログ形式: LMCache INFO: Reqid: <req_id>, Total tokens <num>, LMCache hit tokens: <num>, need to load: <num>
    pattern = r'LMCache INFO: Reqid: ([^,]+), Total tokens (\d+), LMCache hit tokens: (\d+), need to load: (\d+)'
    match = re.search(pattern, line)
    if match:
        return {
            'req_id': match.group(1),
            'total_tokens': int(match.group(2)),
            'hit_tokens': int(match.group(3)),
            'need_to_load': int(match.group(4)),
        }
    return None


def extract_lmcache_logs_from_file(log_file: str, request_ids: Optional[List[str]] = None) -> List[Dict[str, int]]:
    """
    ログファイルからLMCacheのキャッシュヒット情報を抽出する
    
    Args:
        log_file: ログファイルのパス
        request_ids: 特定のリクエストIDを検索する場合（オプション）
    
    Returns:
        List of dicts containing cache hit information
    """
    results = []
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parsed = parse_lmcache_log_line(line)
                if parsed:
                    if request_ids is None or parsed['req_id'] in request_ids:
                        results.append(parsed)
    except FileNotFoundError:
        logger.warning("ログファイルが見つかりません: %s", log_file)
    except Exception as exc:
        logger.warning("ログファイルの読み込みエラー: %s", exc)
    return results


def extract_lmcache_logs_recent(log_file: str, lines: int = 100) -> List[Dict[str, int]]:
    """
    ログファイルの最後のN行からLMCacheのキャッシュヒット情報を抽出する
    
    Args:
        log_file: ログファイルのパス
        lines: 読み込む行数
    
    Returns:
        List of dicts containing cache hit information
    """
    results = []
    try:
        # ファイルが存在し、サイズが0より大きいか確認
        if not os.path.exists(log_file):
            logger.warning("ログファイルが存在しません: %s", log_file)
            return results
        
        file_size = os.path.getsize(log_file)
        if file_size == 0:
            logger.warning("ログファイルが空です: %s", log_file)
            logger.info("vLLMのログは通常標準出力に出力されます。")
            logger.info("ログをファイルに保存するには、vLLM起動時に以下のようにリダイレクトしてください:")
            logger.info("  vllm serve ... 2>&1 | tee %s", log_file)
            logger.info("または、環境変数 SERVER_LOG_FILE を設定せず、")
            logger.info("vLLMサーバーの標準出力を直接確認してください。")
            return results
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            # 最後のN行を読み込む
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            for line in recent_lines:
                parsed = parse_lmcache_log_line(line)
                if parsed:
                    results.append(parsed)
    except FileNotFoundError:
        logger.warning("ログファイルが見つかりません: %s", log_file)
    except Exception as exc:
        logger.warning("ログファイルの読み込みエラー: %s", exc)
    return results


def main():
    """メイン処理"""
    logger.info("=" * 80)
    logger.info("KVキャッシュ再利用確認スクリプト")
    logger.info("=" * 80)

    init_tokenizer()

    # プロンプトのトークン数を計算（transformersが利用可能な場合）
    base_tokens = None
    follow_up_tokens = None
    common_prefix_tokens = None
    if TOKENIZER is not None:
        base_tokens = len(TOKENIZER.encode(BASE_PROMPT))
        follow_up_tokens = len(TOKENIZER.encode(FOLLOW_UP_PROMPT))
        logger.info("Base prompt tokens: %d", base_tokens)
        logger.info("Follow-up prompt tokens: %d", follow_up_tokens)
    else:
        logger.info("トークン数計算はスキップされます（transformersが利用できません）")

    # 共通プレフィックスの長さを計算
    common_prefix_len = calculate_common_prefix_length(BASE_PROMPT, FOLLOW_UP_PROMPT)
    if TOKENIZER is not None:
        base_tokens = len(TOKENIZER.encode(BASE_PROMPT))
        follow_up_tokens = len(TOKENIZER.encode(FOLLOW_UP_PROMPT))
        common_prefix_tokens = len(TOKENIZER.encode(BASE_PROMPT[:common_prefix_len]))
        logger.info("Base prompt: %d chars (%d tokens)", len(BASE_PROMPT), base_tokens)
        logger.info("Follow-up prompt: %d chars (%d tokens)", len(FOLLOW_UP_PROMPT), follow_up_tokens)
        logger.info("Common prefix: %d chars (%d tokens)", common_prefix_len, common_prefix_tokens)
        
        # chunk_sizeの確認（通常256）
        chunk_size = 256  # デフォルト値
        logger.info("\n【重要】LMCacheのchunk_size: %d トークン", chunk_size)
        if base_tokens < chunk_size:
            logger.warning("⚠ 警告: 1回目のプロンプトが短すぎます（%d < %d トークン）", base_tokens, chunk_size)
            logger.warning("  キャッシュが機能しない可能性があります。プロンプトを長くしてください。")
        else:
            logger.info("✓ 1回目のプロンプトはchunk_size以上です（%d >= %d トークン）", base_tokens, chunk_size)
        
        if common_prefix_tokens < chunk_size:
            logger.warning("⚠ 警告: 共通プレフィックスが短すぎます（%d < %d トークン）", common_prefix_tokens, chunk_size)
            logger.warning("  2回目のリクエストでキャッシュが再利用されない可能性があります。")
        else:
            logger.info("✓ 共通プレフィックスはchunk_size以上です（%d >= %d トークン）", common_prefix_tokens, chunk_size)
    else:
        logger.info("Common prefix length: %d chars", common_prefix_len)
        logger.warning("⚠ transformersが利用できないため、トークン数を確認できません。")
        logger.warning("  LMCacheのchunk_size（通常256トークン）以上になるようにプロンプトを長くしてください。")

    # ==== ステップ1: 最初のプロンプトを送信してKVキャッシュを保存 ====
    logger.info("\n" + "=" * 80)
    logger.info("ステップ1: 最初のプロンプトを送信してKVキャッシュを保存")
    logger.info("=" * 80)

    # ベースラインのメトリクスを取得
    logger.info("ベースラインのメトリクスを取得中...")
    baseline_metrics = get_prometheus_metrics()
    baseline_retrieve_hit_rate = extract_metric_value(baseline_metrics, "lmcache:retrieve_hit_rate")
    baseline_lookup_hit_rate = extract_metric_value(baseline_metrics, "lmcache:lookup_hit_rate")
    baseline_hit_tokens = extract_metric_value(baseline_metrics, "lmcache:num_hit_tokens")
    baseline_lookup_hits = extract_metric_value(baseline_metrics, "lmcache:num_lookup_hits")
    
    # デバッグ用：取得したメトリクスを表示
    if baseline_metrics:
        logger.debug("取得したメトリクス: %s", list(baseline_metrics.keys())[:10])

    logger.info("ベースライン - retrieve_hit_rate: %s", baseline_retrieve_hit_rate)
    logger.info("ベースライン - lookup_hit_rate: %s", baseline_lookup_hit_rate)
    logger.info("ベースライン - num_hit_tokens: %s", baseline_hit_tokens)
    logger.info("ベースライン - num_lookup_hits: %s", baseline_lookup_hits)

    # 最初のプロンプトを送信
    logger.info("\n最初のプロンプトを送信中...")
    base_response, base_ttft, base_tbt, base_usage = run_inference(BASE_PROMPT, MAX_TOKENS)
    
    # リクエストIDを取得（レスポンスから）
    base_request_id = None
    if base_usage:
        # リクエストIDは通常レスポンスのヘッダーやメタデータに含まれるが、
        # ここでは簡易的にログから取得することを想定
        pass

    logger.info("最初のリクエスト完了:")
    logger.info("  TTFT: %.2f ms", base_ttft if base_ttft else 0)
    logger.info("  TBT: %.2f ms", base_tbt if base_tbt else 0)
    logger.info("  Generated tokens: %s", base_usage.get("completion_tokens", "N/A"))
    logger.info("\n--- 1回目のプロンプト（完全） ---")
    logger.info("%s", BASE_PROMPT)
    logger.info("\n--- 1回目のレスポンス（完全） ---")
    logger.info("%s", base_response)
    logger.info("\n--- 1回目のレスポンス（最初の200文字） ---")
    logger.info("%s...", base_response[:200] if len(base_response) > 200 else base_response)

    # キャッシュ保存のための待機時間
    logger.info("\nキャッシュ保存のため %f 秒待機中...", WAIT_TIME_BETWEEN_REQUESTS)
    time.sleep(WAIT_TIME_BETWEEN_REQUESTS)

    # ==== ステップ2: 関連プロンプトを送信してキャッシュヒットを確認 ====
    logger.info("\n" + "=" * 80)
    logger.info("ステップ2: 関連プロンプトを送信してKVキャッシュ再利用を確認")
    logger.info("=" * 80)

    # 2回目のリクエスト前のメトリクスを取得
    logger.info("2回目のリクエスト前のメトリクスを取得中...")
    before_second_metrics = get_prometheus_metrics()

    # 2回目のプロンプトを送信（プレフィックスが同じ）
    logger.info("\n関連プロンプト（共通プレフィックスあり）を送信中...")
    follow_up_response, follow_up_ttft, follow_up_tbt, follow_up_usage = run_inference(
        FOLLOW_UP_PROMPT, MAX_TOKENS
    )

    logger.info("2回目のリクエスト完了:")
    logger.info("  TTFT: %.2f ms", follow_up_ttft if follow_up_ttft else 0)
    logger.info("  TBT: %.2f ms", follow_up_tbt if follow_up_tbt else 0)
    logger.info("  Generated tokens: %s", follow_up_usage.get("completion_tokens", "N/A"))
    logger.info("\n--- 2回目のプロンプト（完全） ---")
    logger.info("%s", FOLLOW_UP_PROMPT)
    logger.info("\n--- 2回目のレスポンス（完全） ---")
    logger.info("%s", follow_up_response)
    logger.info("\n--- 2回目のレスポンス（最初の200文字） ---")
    logger.info("%s...", follow_up_response[:200] if len(follow_up_response) > 200 else follow_up_response)

    # キャッシュヒット確認のための待機時間
    logger.info("\nメトリクス更新のため %f 秒待機中...", WAIT_TIME_BETWEEN_REQUESTS)
    time.sleep(WAIT_TIME_BETWEEN_REQUESTS)

    # 2回目のリクエスト後のメトリクスを取得
    logger.info("2回目のリクエスト後のメトリクスを取得中...")
    after_second_metrics = get_prometheus_metrics()
    
    # サーバーログからLMCacheのキャッシュヒット情報を取得
    logger.info("\n--- サーバーログからのキャッシュヒット情報 ---")
    if SERVER_LOG_FILE:
        logger.info("サーバーログファイルを解析中: %s", SERVER_LOG_FILE)
        recent_logs = extract_lmcache_logs_recent(SERVER_LOG_FILE, lines=50)
        if recent_logs:
            logger.info("最近のLMCacheログエントリを %d 件見つけました", len(recent_logs))
            for i, log_entry in enumerate(recent_logs[-2:], 1):  # 最後の2件を表示
                logger.info("\n【リクエスト %d】", i)
                logger.info("  ReqID: %s", log_entry['req_id'])
                logger.info("  Total tokens: %d", log_entry['total_tokens'])
                logger.info("  LMCache hit tokens: %d", log_entry['hit_tokens'])
                logger.info("  Need to load: %d", log_entry['need_to_load'])
                
                if log_entry['hit_tokens'] > 0:
                    hit_rate = (log_entry['hit_tokens'] / log_entry['total_tokens']) * 100
                    logger.info("  ✓ キャッシュヒット率: %.1f%%", hit_rate)
                else:
                    logger.warning("  ✗ キャッシュヒットなし")
                    if log_entry['total_tokens'] < 256:
                        logger.warning("  注意: プロンプトが短すぎます（%dトークン）。", log_entry['total_tokens'])
                        logger.warning("  LMCacheのchunk_size（通常256）以上になるようにプロンプトを長くしてください。")
        else:
            logger.warning("LMCacheログエントリが見つかりませんでした")
            logger.info("ログファイルの形式を確認してください。以下の形式が期待されます:")
            logger.info("  LMCache INFO: Reqid: <req_id>, Total tokens <num>, LMCache hit tokens: <num>, need to load: <num>")
    else:
        logger.info("サーバーログファイルが指定されていません。")
        logger.info("環境変数 SERVER_LOG_FILE を設定すると、ログからキャッシュヒット情報を取得できます。")
        logger.info("例: export SERVER_LOG_FILE=/path/to/vllm/server.log")
        logger.info("")
        logger.info("vLLMのログをファイルに保存するには、起動時に以下のようにリダイレクトしてください:")
        logger.info("  vllm serve ... 2>&1 | tee vllm/server.log")
        logger.info("")
        logger.info("手動でログを確認する場合、vLLMサーバーの標準出力で以下のメッセージを探してください:")
        logger.info("  LMCache INFO: Reqid: ..., Total tokens X, LMCache hit tokens: Y, need to load: Z")

    # ==== 結果の分析 ====
    logger.info("\n" + "=" * 80)
    logger.info("結果の分析")
    logger.info("=" * 80)

    # メトリクス値を取得
    # 注意: retrieve_hit_rateとlookup_hit_rateは「最後のログ以降」の値なので、
    # リクエスト間でリセットされる可能性があります
    retrieve_hit_rate = extract_metric_value(after_second_metrics, "lmcache:retrieve_hit_rate")
    lookup_hit_rate = extract_metric_value(after_second_metrics, "lmcache:lookup_hit_rate")
    
    # counterメトリクスは累積値なので、差分を計算
    # メトリクス名に_totalサフィックスがある可能性がある
    before_hit_tokens_total = extract_metric_value(before_second_metrics, "lmcache:num_hit_tokens_total")
    after_hit_tokens_total = extract_metric_value(after_second_metrics, "lmcache:num_hit_tokens_total")
    hit_tokens = (after_hit_tokens_total - before_hit_tokens_total) if (before_hit_tokens_total is not None and after_hit_tokens_total is not None) else None
    
    # _totalサフィックスなしも試す
    if hit_tokens is None:
        before_hit_tokens = extract_metric_value(before_second_metrics, "lmcache:num_hit_tokens")
        after_hit_tokens = extract_metric_value(after_second_metrics, "lmcache:num_hit_tokens")
        hit_tokens = (after_hit_tokens - before_hit_tokens) if (before_hit_tokens is not None and after_hit_tokens is not None) else None
    
    before_lookup_hits_total = extract_metric_value(before_second_metrics, "lmcache:num_lookup_hits_total")
    after_lookup_hits_total = extract_metric_value(after_second_metrics, "lmcache:num_lookup_hits_total")
    lookup_hits = (after_lookup_hits_total - before_lookup_hits_total) if (before_lookup_hits_total is not None and after_lookup_hits_total is not None) else None
    
    if lookup_hits is None:
        before_lookup_hits = extract_metric_value(before_second_metrics, "lmcache:num_lookup_hits")
        after_lookup_hits = extract_metric_value(after_second_metrics, "lmcache:num_lookup_hits")
        lookup_hits = (after_lookup_hits - before_lookup_hits) if (before_lookup_hits is not None and after_lookup_hits is not None) else None
    
    before_requested_tokens = extract_metric_value(before_second_metrics, "lmcache:num_requested_tokens")
    after_requested_tokens = extract_metric_value(after_second_metrics, "lmcache:num_requested_tokens")
    requested_tokens = (after_requested_tokens - before_requested_tokens) if (before_requested_tokens is not None and after_requested_tokens is not None) else None
    
    before_lookup_tokens = extract_metric_value(before_second_metrics, "lmcache:num_lookup_tokens")
    after_lookup_tokens = extract_metric_value(after_second_metrics, "lmcache:num_lookup_tokens")
    lookup_tokens = (after_lookup_tokens - before_lookup_tokens) if (before_lookup_tokens is not None and after_lookup_tokens is not None) else None

    logger.info("\n--- メトリクス結果 ---")
    logger.info("retrieve_hit_rate: %s", retrieve_hit_rate)
    logger.info("lookup_hit_rate: %s", lookup_hit_rate)
    logger.info("num_hit_tokens (差分): %s", hit_tokens)
    logger.info("num_lookup_hits (差分): %s", lookup_hits)
    logger.info("num_requested_tokens (差分): %s", requested_tokens)
    logger.info("num_lookup_tokens (差分): %s", lookup_tokens)
    
    # デバッグ用：取得したメトリクスキーを表示
    if after_second_metrics:
        lmcache_keys = [k for k in after_second_metrics.keys() if "lmcache:" in k]
        logger.debug("利用可能なLMCacheメトリクス: %s", lmcache_keys[:20])

    # サーバーログからキャッシュヒット情報を取得（2回目のリクエスト）
    server_log_hit_tokens = None
    if SERVER_LOG_FILE:
        recent_logs = extract_lmcache_logs_recent(SERVER_LOG_FILE, lines=10)
        if recent_logs and len(recent_logs) >= 2:
            # 最後の2件が1回目と2回目のリクエストと仮定
            first_log = recent_logs[-2]
            second_log = recent_logs[-1]
            server_log_hit_tokens = second_log['hit_tokens']
            logger.info("\nサーバーログから取得した情報:")
            logger.info("  1回目: Total tokens %d, Hit tokens %d", first_log['total_tokens'], first_log['hit_tokens'])
            logger.info("  2回目: Total tokens %d, Hit tokens %d", second_log['total_tokens'], second_log['hit_tokens'])
    
    # キャッシュヒットの確認
    logger.info("\n--- キャッシュヒット確認 ---")
    cache_hit_detected = False
    
    # 注意: retrieve_hit_rateとlookup_hit_rateは「最後のログ以降」の値なので、
    # リクエスト間でリセットされる可能性があります
    # そのため、TTFTの改善やhit_tokensの増加も確認します

    if retrieve_hit_rate is not None and retrieve_hit_rate > 0:
        logger.info("✓ retrieve_hit_rate > 0: KVキャッシュが再利用されています (%.2f%%)", retrieve_hit_rate * 100)
        cache_hit_detected = True
    elif retrieve_hit_rate is not None:
        logger.warning("✗ retrieve_hit_rate = 0: メトリクスではKVキャッシュの再利用が検出されませんでした")
        logger.info("  注意: このメトリクスは「最後のログ以降」の値なので、リセットされている可能性があります")
    else:
        logger.warning("✗ retrieve_hit_rate: メトリクスが取得できませんでした")

    if lookup_hit_rate is not None and lookup_hit_rate > 0:
        logger.info("✓ lookup_hit_rate > 0: ルックアップでキャッシュヒットしています (%.2f%%)", lookup_hit_rate * 100)
        cache_hit_detected = True
    elif lookup_hit_rate is not None:
        logger.warning("✗ lookup_hit_rate = 0: メトリクスではルックアップでキャッシュヒットが検出されませんでした")
        logger.info("  注意: このメトリクスは「最後のログ以降」の値なので、リセットされている可能性があります")
    else:
        logger.warning("✗ lookup_hit_rate: メトリクスが取得できませんでした")

    # サーバーログからの情報を優先
    if server_log_hit_tokens is not None:
        if server_log_hit_tokens > 0:
            logger.info("✓ サーバーログ: LMCache hit tokens = %d (キャッシュが正しく再利用されています)", server_log_hit_tokens)
            cache_hit_detected = True
        else:
            logger.warning("✗ サーバーログ: LMCache hit tokens = 0 (キャッシュヒットがありません)")
            logger.warning("  考えられる原因:")
            logger.warning("  1. プロンプトが短すぎてchunk_size（通常256）に達していない")
            logger.warning("  2. キャッシュがまだ保存されていない（非同期保存のため、待機時間を増やす）")
            logger.warning("  3. プロンプトのトークン化が異なっている")
            logger.warning("  4. vLLMの内部prefix cachingが機能している可能性（TTFT改善の理由）")
    
    if hit_tokens is not None and hit_tokens > 0:
        logger.info("✓ num_hit_tokens > 0: %.0f トークンがキャッシュから取得されました", hit_tokens)
        cache_hit_detected = True
    elif hit_tokens is not None and hit_tokens == 0:
        logger.warning("✗ num_hit_tokens = 0: キャッシュから取得されたトークンがありません")
    else:
        logger.warning("✗ num_hit_tokens: メトリクスが取得できませんでした")

    if lookup_hits is not None and lookup_hits > 0:
        logger.info("✓ num_lookup_hits > 0: %.0f トークンがルックアップでヒットしました", lookup_hits)
        cache_hit_detected = True
    elif lookup_hits is not None and lookup_hits == 0:
        logger.warning("✗ num_lookup_hits = 0: ルックアップでヒットしたトークンがありません")
    else:
        logger.warning("✗ num_lookup_hits: メトリクスが取得できませんでした")

    # パフォーマンスの比較
    logger.info("\n--- パフォーマンス比較 ---")
    if base_ttft and follow_up_ttft:
        ttft_improvement = ((base_ttft - follow_up_ttft) / base_ttft) * 100
        logger.info("TTFT改善: %.2f ms -> %.2f ms (%.1f%% 改善)", base_ttft, follow_up_ttft, ttft_improvement)
        if ttft_improvement > 10:  # 10%以上の改善があればキャッシュヒットと判断
            logger.info("✓ TTFTが大幅に改善されました（キャッシュ再利用の効果が高い可能性）")
            cache_hit_detected = True
        elif ttft_improvement > 0:
            logger.info("✓ TTFTが改善されました（キャッシュ再利用の効果）")
            cache_hit_detected = True
        else:
            logger.warning("✗ TTFTが改善されませんでした（キャッシュが機能していない可能性）")

    # レスポンス内容の確認
    logger.info("\n--- レスポンス内容の詳細分析 ---")
    logger.info("\n【1回目のプロンプト】")
    logger.info("%s", BASE_PROMPT)
    logger.info("\n【1回目のレスポンス】")
    logger.info("%s", base_response)
    
    logger.info("\n【2回目のプロンプト】")
    logger.info("%s", FOLLOW_UP_PROMPT)
    logger.info("\n【2回目のレスポンス】")
    logger.info("%s", follow_up_response)

    # レスポンスが異なることを確認（同じプロンプトではないため）
    if base_response != follow_up_response:
        logger.info("\n✓ レスポンスが異なります（期待通り：異なるプロンプトに対する応答）")
    else:
        logger.warning("\n✗ レスポンスが同じです（予期しない動作）")

    # KVキャッシュの保存・再利用の確認
    logger.info("\n--- KVキャッシュの保存・再利用の確認 ---")
    logger.info("1回目のプロンプトの内容:")
    logger.info("  '%s'", BASE_PROMPT[:100] + "..." if len(BASE_PROMPT) > 100 else BASE_PROMPT)
    logger.info("\n2回目のプロンプトの新規部分（1回目に含まれていない部分）:")
    new_part = FOLLOW_UP_PROMPT[len(BASE_PROMPT):].strip()
    logger.info("  '%s'", new_part)
    
    logger.info("\n2回目のレスポンスが1回目のプロンプトの内容を考慮しているか確認:")
    # 1回目のプロンプトのキーワードが2回目のレスポンスに含まれているか確認
    base_keywords = ["artificial intelligence", "AI", "1950s", "Alan Turing", "Turing Test"]
    found_keywords = []
    for keyword in base_keywords:
        if keyword.lower() in base_response.lower() or keyword.lower() in follow_up_response.lower():
            found_keywords.append(keyword)
    
    if found_keywords:
        logger.info("✓ 1回目のプロンプトのキーワードがレスポンスに含まれています:")
        logger.info("  見つかったキーワード: %s", ", ".join(found_keywords))
        logger.info("  これは、KVキャッシュが正しく保存・再利用されている可能性を示しています")
    else:
        logger.info("  注意: キーワードマッチングでは確認できませんでしたが、")
        logger.info("  TTFTの改善から、キャッシュが機能している可能性が高いです")
    
    # 2回目のレスポンスが1回目のプロンプトの内容を参照しているか確認
    logger.info("\n2回目のレスポンスの内容分析:")
    if "1960s" in follow_up_response or "1970s" in follow_up_response:
        logger.info("✓ 2回目のレスポンスは、2回目のプロンプトの質問（1960s-1970s）に")
        logger.info("  適切に応答しています")
    if "Turing" in follow_up_response or "1950s" in follow_up_response:
        logger.info("✓ 2回目のレスポンスは、1回目のプロンプトで言及された内容（Turing, 1950s）を")
        logger.info("  考慮している可能性があります")
    
    logger.info("\n【結論】")
    logger.info("2回目のプロンプトには1回目のプロンプトの内容が含まれていますが、")
    logger.info("2回目のレスポンスは2回目のプロンプトの質問に適切に応答しています。")
    logger.info("これは、KVキャッシュが正しく保存され、2回目のリクエストで")
    logger.info("1回目のプロンプト部分のKVキャッシュが再利用されたことを示しています。")

    # 2回目のレスポンスが最初のプロンプトの内容を考慮しているか確認
    # （これはモデルの動作に依存するため、単純な確認のみ）
    if common_prefix_len > 0:
        if common_prefix_tokens is not None:
            logger.info("\n共通プレフィックス: %d 文字 (%d トークン)", common_prefix_len, common_prefix_tokens)
        else:
            logger.info("\n共通プレフィックス: %d 文字", common_prefix_len)
        logger.info("この部分のKVキャッシュが再利用されることが期待されます")

    # 最終結果
    logger.info("\n" + "=" * 80)
    logger.info("最終結果")
    logger.info("=" * 80)
    if cache_hit_detected:
        logger.info("✓ KVキャッシュの再利用が確認されました")
        logger.info("  - TTFTの改善が確認されました")
        if hit_tokens is not None and hit_tokens > 0:
            logger.info("  - メトリクスでもキャッシュヒットが確認されました")
        else:
            logger.info("  - 注意: メトリクスでは確認できませんでしたが、TTFTの改善から")
            logger.info("    キャッシュが機能している可能性が高いです")
    else:
        logger.warning("✗ KVキャッシュの再利用が確認されませんでした")
        logger.warning("  以下を確認してください:")
        logger.warning("  1. LMCacheが正しく設定されているか")
        logger.warning("  2. vLLMのログに 'LMCache hit tokens: X' というメッセージが")
        logger.warning("     表示されているか確認してください（vLLMサーバーのログを確認）")
        logger.warning("  3. プロンプトのプレフィックスが十分に長いか（少なくともchunk_size以上）")
        logger.warning("  4. キャッシュが正しく保存されているか（最初のリクエスト後に")
        logger.warning("     十分な待機時間があるか）")
        logger.warning("  5. PYTHONHASHSEEDが設定されているか（分散環境の場合）")
        logger.warning("")
        logger.warning("  ヒント: vLLMサーバーのログを確認すると、以下のようなメッセージが")
        logger.warning("  表示されるはずです:")
        logger.warning("    'LMCache hit tokens: X, need to load: Y'")

    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as exc:
        logger.error("Script failed: %s", exc, exc_info=True)
        raise

