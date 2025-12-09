# 2ノード構成でのKVキャッシュ再利用確認スクリプト

このスクリプトは、2ノード構成（prefill/decode分離）でのKVキャッシュの再利用が正しく機能しているかを確認するためのツールです。

## 概要

このスクリプトは以下のステップを実行します：

1. **PrefillサーバーでKVキャッシュ生成**: 最初のプロンプトをprefillサーバー（192.168.110.13）に送信し、KVキャッシュを生成・保存します
2. **KVキャッシュの転送**: LMCacheのP2P機能により、KVキャッシュがdecodeサーバー（192.168.110.97）に転送されます
3. **Decodeサーバーでキャッシュ再利用**: 2回目のプロンプト（共通プレフィックスあり）をdecodeサーバーに送信し、KVキャッシュが再利用されているかを確認します

## 前提条件

- 2ノード構成のvLLMサーバーが起動していること
  - Prefillサーバー: 192.168.110.13:8010
  - Decodeサーバー: 192.168.110.97:8011
- LMCacheが設定され、P2P転送が有効になっていること
- Controllerが起動していること
- Prometheusメトリクスが有効になっていること（オプション）
- Python 3.8以上
- 必要なパッケージがインストールされていること（`requests`は必須、`transformers`はオプショナル）

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install requests
# オプション
pip install transformers
```

### 2. 環境変数の設定（オプション）

```bash
# PrefillサーバーのURL
export PREFILL_URL="http://192.168.110.13:8010/v1/completions"

# DecodeサーバーのURL
export DECODE_URL="http://192.168.110.97:8011/v1/completions"

# メトリクスエンドポイント
export PREFILL_METRICS_URL="http://192.168.110.13:8010/metrics"
export DECODE_METRICS_URL="http://192.168.110.97:8011/metrics"

# モデル名
export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"

# 生成する最大トークン数
export MAX_TOKENS=256

# リクエスト間の待機時間（秒）- 2ノード構成では長めに設定
export WAIT_TIME_BETWEEN_REQUESTS=5.0

# ログファイル（オプション）
export PREFILL_LOG_FILE="/path/to/prefill/server.log"
export DECODE_LOG_FILE="/path/to/decode/server.log"
```

## 使用方法

### 基本的な使用方法

```bash
python verify_kv_cache_reuse_2node.py
```

### サーバーログファイルを指定する場合

```bash
# vLLMサーバー起動時にログをリダイレクト
# Prefillサーバー
vllm serve ... 2>&1 | tee prefill_server.log

# Decodeサーバー
vllm serve ... 2>&1 | tee decode_server.log

# スクリプト実行
export PREFILL_LOG_FILE="prefill_server.log"
export DECODE_LOG_FILE="decode_server.log"
python verify_kv_cache_reuse_2node.py
```

## 動作の流れ

1. **Prefillフェーズ**:
   - Prefillサーバーに最初のプロンプトを送信
   - KVキャッシュを生成（prefill処理）
   - LMCacheがKVキャッシュを保存

2. **転送フェーズ**:
   - LMCacheのP2P機能により、KVキャッシュがdecodeサーバーに転送
   - Controllerが転送を管理

3. **Decodeフェーズ**:
   - Decodeサーバーに2回目のプロンプト（共通プレフィックスあり）を送信
   - KVキャッシュが再利用されることを確認
   - 生成されたテキストを確認

## 確認ポイント

スクリプトは以下の点を確認します：

1. **KVキャッシュの再利用**: 
   - Decodeサーバーのメトリクスから `retrieve_hit_rate > 0` または `num_hit_tokens > 0`
   - Decodeサーバーのログから `LMCache hit tokens: X` (X > 0)

2. **パフォーマンス**:
   - Prefill処理時間
   - Decode TTFT（Time-to-First-Token）
   - Decode TBT（Time-to-Best-Token）

3. **レスポンスの妥当性**:
   - 2回目のレスポンスが適切な内容になっているか

## 重要な注意事項

### プロンプトの構造

LMCacheは**プレフィックスキャッシング**を使用します。2回目のプロンプトは最初のプロンプトのプレフィックスを含む必要があります。

### 待機時間

2ノード構成では、KVキャッシュの転送に時間がかかるため、`WAIT_TIME_BETWEEN_REQUESTS`を長めに設定することを推奨します（デフォルト: 5.0秒）。

### P2P設定

LMCacheのP2P設定が正しく構成されていることを確認してください：
- `enable_p2p: True`
- `p2p_host`が正しく設定されている
- Controllerが起動している

### プロンプトの長さ

LMCacheのchunk_size（通常256トークン）以上になるようにプロンプトを長くしてください。

## トラブルシューティング

### キャッシュヒットが検出されない場合

1. **P2P設定の確認**
   - `enable_p2p: True` が設定されているか
   - `p2p_host`が正しいIPアドレスになっているか

2. **Controllerの確認**
   - Controllerが起動しているか
   - ControllerのURLが正しいか

3. **待機時間の確認**
   - `WAIT_TIME_BETWEEN_REQUESTS`を増やす（例: 10.0秒）

4. **ログの確認**
   - Decodeサーバーのログで `LMCache hit tokens: X` を確認
   - PrefillサーバーのログでKVキャッシュの保存を確認

### メトリクスが取得できない場合

- メトリクスURLが正しく設定されているか確認
- Prometheusメトリクスが有効になっているか確認

## 関連ドキュメント

- [LMCache公式ドキュメント](https://lmcache.readthedocs.io/)
- [vLLMドキュメント](https://docs.vllm.ai/)
- [1ノード構成の確認スクリプト](verify_kv_cache_reuse_README.md)

