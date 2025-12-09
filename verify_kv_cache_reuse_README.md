# KVキャッシュ再利用確認スクリプト

このスクリプトは、LMCacheを使用したKVキャッシュの再利用が正しく機能しているかを確認するためのツールです。

## 概要

このスクリプトは以下の2つのステップを実行します：

1. **最初のプロンプト送信**: サーバーに最初のプロンプトを送信し、KVキャッシュを保存します
2. **関連プロンプト送信**: 最初のプロンプトと共通のプレフィックスを持つプロンプトを送信し、KVキャッシュが再利用されているかを確認します

## 前提条件

- LMCacheが設定され、vLLMサーバーが起動していること
- Prometheusメトリクスが有効になっていること（`/metrics`エンドポイントが利用可能）
- Python 3.8以上
- 必要なパッケージがインストールされていること（`requests`は必須、`transformers`はオプショナル）

**注意**: `transformers`ライブラリは、トークン数の計算に使用されますが、必須ではありません。`transformers`がない場合でも、スクリプトは動作しますが、トークン数の表示はスキップされます。

## セットアップ

### 1. 依存パッケージのインストール

必須パッケージ：
```bash
pip install requests
```

オプショナルパッケージ（トークン数計算用）：
```bash
pip install transformers
```

または、両方を一度にインストール：
```bash
pip install requests transformers
```

### 2. 環境変数の設定（オプション）

以下の環境変数を設定することで、スクリプトの動作をカスタマイズできます：

```bash
# vLLMサーバーのURL
export VLLM_URL="http://localhost:8010/v1/completions"

# PrometheusメトリクスエンドポイントのURL
export METRICS_URL="http://localhost:8010/metrics"

# モデル名
export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"

# トークナイザー名（通常はMODEL_NAMEと同じ）
export TOKENIZER_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"

# 生成する最大トークン数
export MAX_TOKENS=256

# リクエスト間の待機時間（秒）
export WAIT_TIME_BETWEEN_REQUESTS=2.0

# カスタムプロンプト（オプション）
export BASE_PROMPT="Your custom base prompt here..."
export FOLLOW_UP_PROMPT="Your custom follow-up prompt here..."
```

## 使用方法

### 基本的な使用方法

```bash
python verify_kv_cache_reuse.py
```

### Prometheusメトリクスが有効な場合

vLLMサーバーを起動する際に、Prometheusメトリクスを有効にする必要があります：

```bash
PROMETHEUS_MULTIPROC_DIR=/tmp/lmcache_prometheus \
PYTHONHASHSEED=123 \
UCX_TLS=tcp \
CUDA_VISIBLE_DEVICES=0 \
LMCACHE_CONFIG_FILE=example1.yaml \
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 1600 \
  --port 8010 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

その後、メトリクスエンドポイントが利用可能になります：
- `http://localhost:8010/metrics`

### カスタムプロンプトを使用する場合

環境変数でプロンプトを指定できます：

```bash
export BASE_PROMPT="The history of artificial intelligence dates back to ancient times..."
export FOLLOW_UP_PROMPT="The history of artificial intelligence dates back to ancient times... What were the key developments in AI research during the 1960s and 1970s?"

python verify_kv_cache_reuse.py
```

### サーバーログファイルを指定する場合

サーバーログから直接キャッシュヒット情報を取得する場合：

**重要**: vLLMのログは通常標準出力に出力されるため、起動時にリダイレクトする必要があります：

```bash
# vLLMサーバー起動時にログをファイルに保存
vllm serve ... 2>&1 | tee vllm/server.log

# 別のターミナルでスクリプトを実行
export SERVER_LOG_FILE="vllm/server.log"
python verify_kv_cache_reuse.py
```

これにより、vLLMサーバーのログから `LMCache hit tokens: X` というメッセージを自動的に解析し、キャッシュヒットを確認できます。

**注意**: ログファイルが空の場合、スクリプトは警告を表示し、ログをリダイレクトする方法を案内します。

## 出力の説明

スクリプトは以下の情報を出力します：

### 1. ベースライン情報
- 最初のリクエスト前のメトリクス値
- プロンプトのトークン数
- 共通プレフィックスの長さ

### 2. 最初のリクエスト結果
- Time-to-First-Token (TTFT)
- Time-to-Best-Token (TBT)
- 生成されたトークン数
- レスポンスの一部

### 3. 2回目のリクエスト結果
- TTFTとTBT（キャッシュ再利用により改善されることが期待される）
- 生成されたトークン数
- レスポンスの一部

### 4. メトリクス分析
- `retrieve_hit_rate`: リトリーブリクエストのヒット率
- `lookup_hit_rate`: ルックアップリクエストのヒット率
- `num_hit_tokens`: キャッシュから取得されたトークン数
- `num_lookup_hits`: ルックアップでヒットしたトークン数

### 5. キャッシュヒット確認
- キャッシュが再利用されているかどうかの確認結果
- パフォーマンス改善の確認
- レスポンス内容の確認

## 確認ポイント

スクリプトは以下の点を確認します：

1. **KVキャッシュの再利用**: 
   - `retrieve_hit_rate > 0` または `lookup_hit_rate > 0`（メトリクスが利用可能な場合）
   - `num_hit_tokens > 0` または `num_lookup_hits > 0`（メトリクスが利用可能な場合）
   - **重要**: メトリクスは「最後のログ以降」の値なので、リクエスト間でリセットされる可能性があります

2. **パフォーマンス改善**:
   - 2回目のリクエストのTTFTが1回目より短い（キャッシュ再利用の効果）
   - **最も確実な確認方法**: TTFTの改善が確認できれば、キャッシュが機能している可能性が高いです

3. **レスポンスの妥当性**:
   - 2回目のレスポンスが最初のプロンプトとは異なる（異なるプロンプトに対する応答）
   - レスポンスが適切な内容になっている

## 重要な注意事項

### プロンプトの構造

LMCacheは**プレフィックスキャッシング**を使用します。つまり：

- ✅ **正しい例**: 
  - 1回目: "The history of AI..."
  - 2回目: "The history of AI... What were the key developments?"（1回目のプロンプトが2回目の先頭部分）

- ❌ **間違った例**: 
  - 1回目: "The history of AI..."
  - 2回目: "What were the key developments? The history of AI..."（1回目のプロンプトが2回目の途中にある）

### メトリクスの制限

`retrieve_hit_rate`と`lookup_hit_rate`は「最後のログ以降」の値なので、リクエスト間でリセットされる可能性があります。そのため、メトリクスが0でも、TTFTが改善されていればキャッシュが機能している可能性があります。

### vLLMログの確認

より確実な確認方法は、vLLMサーバーのログを直接確認することです。キャッシュヒットが発生すると、以下のようなメッセージが表示されます：

```
LMCache INFO: Reqid: ..., Total tokens X, LMCache hit tokens: Y, need to load: Z
```

このメッセージが表示されれば、キャッシュが正しく機能していることが確認できます。

**重要**: ログで `LMCache hit tokens: 0` と表示される場合、以下の原因が考えられます：

1. **プロンプトが短すぎる**: LMCacheはchunk_size（通常256トークン）ごとにキャッシュを管理します。プロンプトがchunk_size未満の場合、キャッシュが機能しない可能性があります。
2. **キャッシュがまだ保存されていない**: 非同期でキャッシュが保存されるため、最初のリクエスト直後は保存が完了していない可能性があります。`WAIT_TIME_BETWEEN_REQUESTS`を増やすことを検討してください。
3. **vLLMの内部prefix caching**: vLLM自体にもprefix caching機能があり、これが機能している場合、LMCacheのキャッシュヒットが0でもTTFTが改善される可能性があります。

スクリプトに `SERVER_LOG_FILE` 環境変数を設定すると、ログから自動的にキャッシュヒット情報を取得できます。

## トラブルシューティング

### キャッシュヒットが検出されない場合

1. **LMCacheの設定を確認**
   - `LMCACHE_CONFIG_FILE`が正しく設定されているか
   - `kv-transfer-config`が正しく設定されているか

2. **Prometheusメトリクスの確認**
   - `/metrics`エンドポイントにアクセスできるか
   - `PROMETHEUS_MULTIPROC_DIR`が設定されているか

3. **プロンプトの確認**
   - 共通プレフィックスが十分に長いか（**重要**: LMCacheのchunk_size（通常256トークン）以上である必要があります）
   - プロンプトが正しく送信されているか
   - **注意**: プロンプトが短すぎる場合（chunk_size未満）、キャッシュが機能しない可能性があります
   - **デフォルトのプロンプト**: スクリプトのデフォルトプロンプトは、chunk_size（256トークン）以上になるように設計されています

4. **キャッシュの保存確認**
   - 最初のリクエスト後に十分な待機時間があるか（`WAIT_TIME_BETWEEN_REQUESTS`）
   - キャッシュが正しく保存されているか（ログを確認）

### メトリクスが取得できない場合

- `METRICS_URL`が正しく設定されているか確認
- vLLMサーバーが起動しているか確認
- ファイアウォールやネットワーク設定を確認

### ログファイルが空の場合

vLLMのログは通常標準出力に出力されます。ログをファイルに保存するには：

1. **vLLM起動時にリダイレクト**:
   ```bash
   vllm serve ... 2>&1 | tee vllm/server.log
   ```

2. **環境変数を設定**:
   ```bash
   export SERVER_LOG_FILE="vllm/server.log"
   ```

3. **手動でログを確認**: vLLMサーバーの標準出力で以下のメッセージを探してください:
   ```
   LMCache INFO: Reqid: ..., Total tokens X, LMCache hit tokens: Y, need to load: Z
   ```

### レスポンスが期待と異なる場合

- モデルの動作は確率的な場合があるため、完全に同じレスポンスになるとは限りません
- `temperature=0.0`に設定しているため、同じプロンプトに対しては同じレスポンスが期待されます
- 異なるプロンプトに対しては異なるレスポンスが期待されます

## 注意事項

- このスクリプトは、KVキャッシュの再利用が機能しているかを確認するためのものです
- 実際のパフォーマンスは、ハードウェア、ネットワーク、モデルサイズなどに依存します
- Prometheusメトリクスが利用できない場合、一部の確認ができませんが、基本的な動作確認は可能です

## 関連ドキュメント

- [LMCache公式ドキュメント](https://lmcache.readthedocs.io/)
- [vLLMドキュメント](https://docs.vllm.ai/)
- [Prometheusメトリクス](docs/source/production/observability/vllm_endpoint.rst)

