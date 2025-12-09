# 2ノード構成でのP2P設定エラー修正ガイド

## 概要

2ノード構成（prefill/decode分離）では、**両方のvLLMサーバー**で同じ修正が必要です。

## 構成

- **Prefillサーバー**: 192.168.110.13:8010
- **Decodeサーバー**: 192.168.110.97:8011
- **Controller**: 192.168.110.13:8300, 8400

## 修正が必要な理由

両方のvLLMサーバーがControllerにメッセージ（`RegisterMsg`、`HeartbeatMsg`など）を送信するため、両方で同じエラーが発生する可能性があります。

## 修正方法

### 方法1: 各サーバーで個別に修正

#### Prefillサーバー（192.168.110.13）

```bash
# 現在のマシンで実行
cd /home/nakaolab/kondo/LMCache
./fix_p2p_error.sh
```

#### Decodeサーバー（192.168.110.97）

**オプションA: SSH経由で修正（推奨）**

```bash
# 現在のマシンから実行
./fix_p2p_error_remote.sh 192.168.110.97
```

**オプションB: Decodeサーバーに直接ログインして修正**

```bash
# DecodeサーバーにSSH接続
ssh 192.168.110.97

# Decodeサーバーで実行
cd /home/nakaolab/kondo/LMCache  # または適切なパス
./fix_p2p_error.sh
```

### 方法2: 手動で修正

各サーバーで以下のコマンドを実行：

```bash
VENV_PATH="/home/nakaolab/kondo/LMCache/.venv"
MESSAGE_FILE="${VENV_PATH}/lib/python3.12/site-packages/lmcache/v1/cache_controller/message.py"

# バックアップを作成
cp "$MESSAGE_FILE" "${MESSAGE_FILE}.backup.$(date +%Y%m%d_%H%M%S)"

# 修正を適用
sed -i 's/peer_init_url: Optional\[str\]$/peer_init_url: Optional[str] = None/' "$MESSAGE_FILE"

# 修正を確認
grep "peer_init_url: Optional\[str\] = None" "$MESSAGE_FILE"
```

## 確認方法

### 1. 修正の確認

各サーバーで以下を実行：

```bash
grep "peer_init_url: Optional\[str\] = None" \
  /home/nakaolab/kondo/LMCache/.venv/lib/python3.12/site-packages/lmcache/v1/cache_controller/message.py
```

### 2. サーバー再起動

両方のサーバーを再起動：

```bash
# Prefillサーバー（192.168.110.13）
# vLLMサーバーを再起動

# Decodeサーバー（192.168.110.97）
# vLLMサーバーを再起動
```

### 3. エラーの確認

Controllerのログで、以下のエラーが表示されなくなっていることを確認：

```
msgspec.ValidationError: Object missing required field `peer_init_url`
```

## 注意事項

1. **仮想環境のパスが異なる場合**
   - Decodeサーバーで仮想環境のパスが異なる場合は、`fix_p2p_error_remote.sh`の`VENV_PATH`を修正してください

2. **Pythonバージョンが異なる場合**
   - DecodeサーバーでPythonバージョンが異なる場合（例: 3.11）、パス内の`python3.12`を適切なバージョンに変更してください

3. **LMCacheパッケージの再インストール**
   - LMCacheパッケージを再インストールすると、この修正が上書きされる可能性があります
   - その場合は、再修正が必要です

## トラブルシューティング

### エラーが解消されない場合

1. **両方のサーバーで修正が適用されているか確認**
   ```bash
   # Prefillサーバー
   grep "peer_init_url" /path/to/venv/.../message.py
   
   # Decodeサーバー
   ssh 192.168.110.97 "grep 'peer_init_url' /path/to/venv/.../message.py"
   ```

2. **サーバーが再起動されているか確認**
   - 両方のサーバーが再起動されていることを確認してください

3. **Controllerのログを確認**
   - Controllerのログで、どのサーバーからエラーが発生しているか確認してください

## 次のステップ

修正が完了したら：

1. 両方のvLLMサーバーを再起動
2. Controllerのログでエラーが解消されているか確認
3. `verify_kv_cache_reuse_2node.py`を実行して、KVキャッシュの再利用が正常に動作するか確認

