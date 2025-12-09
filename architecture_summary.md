# 実装したアーキテクチャのまとめ

## アーキテクチャ概要

**2ノード構成（Prefill/Decode分離）でのKVキャッシュ再利用システム**

```
┌─────────────────────────────────────────────────────────────────┐
│                        2ノード構成アーキテクチャ                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐         ┌──────────────────────┐
│  Prefillサーバー      │         │  Decodeサーバー      │
│  192.168.110.13:8010 │         │  192.168.110.97:8011 │
│                      │         │                      │
│  - Prefill処理        │         │  - Decode処理        │
│  - KVキャッシュ生成   │         │  - KVキャッシュ再利用 │
│  - LMCache Worker    │         │  - LMCache Worker    │
│    (instance_1)      │         │    (instance_2)      │
└──────────┬───────────┘         └──────────┬───────────┘
           │                                │
           │                                │
           │         ┌──────────────┐       │
           └────────▶│   Controller │◀──────┘
                     │ 192.168.110. │
                     │   13:8300/   │
                     │   8400/9000  │
                     └──────┬───────┘
                            │
                            │ P2P転送制御
                            │ (NIXL経由)
                            │
                     ┌──────▼───────┐
                     │  P2P転送      │
                     │  (NIXL)      │
                     │              │
                     │ KVキャッシュ  │
                     │ を転送       │
                     └──────────────┘
```

## コンポーネント詳細

### 1. Prefillサーバー（192.168.110.13:8010）

**役割:**
- プロンプトのPrefill処理（初期トークン処理）
- KVキャッシュの生成と保存
- LMCacheへのKVキャッシュの登録

**設定ファイル:** `example1.yaml`
- `lmcache_instance_id: "lmcache_instance_1"`
- `p2p_init_ports: 8200`
- `p2p_lookup_ports: 8201`
- `lmcache_worker_ports: 8500`

**起動コマンド:**
```bash
PYTHONHASHSEED=123 UCX_TLS=tcp CUDA_VISIBLE_DEVICES=0 \
LMCACHE_CONFIG_FILE=example1.yaml \
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 1600 \
  --port 8010 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### 2. Decodeサーバー（192.168.110.97:8011）

**役割:**
- トークンのDecode処理（生成）
- Prefillサーバーから転送されたKVキャッシュの再利用
- 共通プレフィックスを持つプロンプトでのキャッシュヒット

**設定ファイル:** `example2.yaml`（推測）
- `lmcache_instance_id: "lmcache_instance_2"`
- `p2p_init_ports: 8202`（推測）
- `p2p_lookup_ports: 8203`（推測）
- `lmcache_worker_ports: 8501`（推測）

**起動コマンド:**
```bash
PYTHONHASHSEED=123 UCX_TLS=tcp CUDA_VISIBLE_DEVICES=0 \
LMCACHE_CONFIG_FILE=example2.yaml \
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 1600 \
  --port 8011 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### 3. Controller（192.168.110.13:8300, 8400, 9000）

**役割:**
- LMCache Workerの登録管理
- KVキャッシュのメタデータ管理
- P2P転送の制御とルーティング

**起動コマンド:**
```bash
PYTHONHASHSEED=123 lmcache_controller \
  --host 0.0.0.0 \
  --port 9000 \
  --monitor-ports '{"pull": 8300, "reply": 8400}'
```

## データフロー

### KVキャッシュ再利用の流れ

```
1. Prefillフェーズ
   ┌─────────────────────────────────────────┐
   │ Client → Prefill Server (192.168.110.13)│
   │                                         │
   │ - プロンプト送信                        │
   │ - Prefill処理（KVキャッシュ生成）       │
   │ - LMCacheにKVキャッシュを保存           │
   └─────────────────────────────────────────┘
                    │
                    ▼
2. KVキャッシュ転送
   ┌─────────────────────────────────────────┐
   │ Controller → P2P転送 (NIXL)             │
   │                                         │
   │ - KVキャッシュのメタデータを管理        │
   │ - Decodeサーバーへの転送を指示          │
   │ - NIXL経由で高速転送                    │
   └─────────────────────────────────────────┘
                    │
                    ▼
3. Decodeフェーズ
   ┌─────────────────────────────────────────┐
   │ Decode Server (192.168.110.97)           │
   │                                         │
   │ - 共通プレフィックスを持つプロンプト受信│
   │ - KVキャッシュを再利用（キャッシュヒット）│
   │ - Decode処理（生成）                    │
   └─────────────────────────────────────────┘
```

## 実装したスクリプト

### 1. `verify_kv_cache_reuse_2node.py`

**目的:** 2ノード構成でのKVキャッシュ再利用を確認

**動作:**
1. Prefillサーバーに最初のプロンプトを送信
2. KVキャッシュを生成・保存
3. KVキャッシュの転送を待機（5秒）
4. Decodeサーバーに共通プレフィックスを持つプロンプトを送信
5. KVキャッシュの再利用を確認

**確認方法:**
- Prometheusメトリクス（`retrieve_hit_rate`, `num_hit_tokens`）
- サーバーログ（`LMCache hit tokens: X`）
- TTFT（Time-to-First-Token）の改善

### 2. `20251128_2node_inference.py`

**目的:** 2ノード構成でのベンチマーク測定

**測定項目:**
- Prefill処理時間
- KVキャッシュ転送時間
- Decode処理時間
- TTFT, TBT
- スループット

## 技術スタック

### LMCacheの機能

1. **P2P転送機能**
   - NIXL（NVIDIA Inference Xfer Library）を使用
   - NVLink、RDMA、TCP経由での高速転送
   - Controllerベースの制御

2. **プレフィックスキャッシング**
   - 共通プレフィックスを持つプロンプトでKVキャッシュを再利用
   - `chunk_size: 256`トークン単位で管理

3. **ストレージバックエンド**
   - Local CPU: `max_local_cpu_size: 5` GB
   - 非同期ローディング: `enable_async_loading: True`

### 転送チャネル

- **NIXL**: GPU間の高速転送（NVLink/RDMA/TCP）
- **UCX_TLS=tcp**: TCP経由での転送設定

## 設定のポイント

### P2P設定

```yaml
enable_p2p: True
p2p_host: "192.168.110.13"  # PrefillサーバーのIP
p2p_init_ports: 8200        # P2P初期化ポート
p2p_lookup_ports: 8201      # P2Pルックアップポート
transfer_channel: "nixl"    # NIXLを使用
```

### Controller設定

```yaml
enable_controller: True
lmcache_instance_id: "lmcache_instance_1"  # インスタンスID
controller_pull_url: "192.168.110.13:8300"  # Controller pull URL
controller_reply_url: "192.168.110.13:8400" # Controller reply URL
lmcache_worker_ports: 8500                  # Workerポート
```

## 修正した問題

### P2P設定エラー

**問題:**
```
msgspec.ValidationError: Object missing required field `peer_init_url`
```

**原因:**
- `RegisterMsg`クラスの`peer_init_url`フィールドにデフォルト値がなかった
- msgspecが必須フィールドとして扱っていた

**修正:**
- 仮想環境内のパッケージファイルを修正
- `peer_init_url: Optional[str] = None`に変更

**修正箇所:**
- Prefillサーバー: `/home/nakaolab/kondo/LMCache/.venv/lib/python3.12/site-packages/lmcache/v1/cache_controller/message.py`
- Decodeサーバー: 同様の修正が必要

## 実験の流れ

1. **Controller起動**
   ```bash
   PYTHONHASHSEED=123 lmcache_controller --host 0.0.0.0 --port 9000 --monitor-ports '{"pull": 8300, "reply": 8400}'
   ```

2. **Prefillサーバー起動**（192.168.110.13）
   ```bash
   PYTHONHASHSEED=123 UCX_TLS=tcp CUDA_VISIBLE_DEVICES=0 \
   LMCACHE_CONFIG_FILE=example1.yaml \
   vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
     --host 0.0.0.0 --port 8010 \
     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
   ```

3. **Decodeサーバー起動**（192.168.110.97）
   ```bash
   PYTHONHASHSEED=123 UCX_TLS=tcp CUDA_VISIBLE_DEVICES=0 \
   LMCACHE_CONFIG_FILE=example2.yaml \
   vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
     --host 0.0.0.0 --port 8011 \
     --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
   ```

4. **KVキャッシュ再利用確認**
   ```bash
   python verify_kv_cache_reuse_2node.py
   ```

## 確認ポイント

1. **KVキャッシュの再利用**
   - Decodeサーバーのログで`LMCache hit tokens: X`（X > 0）を確認
   - Prometheusメトリクスで`retrieve_hit_rate > 0`を確認

2. **パフォーマンス改善**
   - TTFT（Time-to-First-Token）の改善
   - KVキャッシュ転送時間の測定

3. **エラーの解消**
   - Controllerのログで`peer_init_url`エラーが発生していないことを確認

