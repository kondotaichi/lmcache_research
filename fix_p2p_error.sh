#!/bin/bash
# P2P設定エラー修正スクリプト
# 仮想環境内のLMCacheパッケージファイルを修正します

VENV_PATH="/home/nakaolab/kondo/LMCache/.venv"
MESSAGE_FILE="${VENV_PATH}/lib/python3.12/site-packages/lmcache/v1/cache_controller/message.py"

echo "LMCache P2P設定エラー修正スクリプト"
echo "=================================="

if [ ! -f "$MESSAGE_FILE" ]; then
    echo "エラー: ファイルが見つかりません: $MESSAGE_FILE"
    exit 1
fi

# バックアップを作成
BACKUP_FILE="${MESSAGE_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
cp "$MESSAGE_FILE" "$BACKUP_FILE"
echo "✓ バックアップを作成しました: $BACKUP_FILE"

# 修正を適用
sed -i 's/peer_init_url: Optional\[str\]$/peer_init_url: Optional[str] = None/' "$MESSAGE_FILE"

# 修正を確認
if grep -q "peer_init_url: Optional\[str\] = None" "$MESSAGE_FILE"; then
    echo "✓ 修正が正常に適用されました"
    echo ""
    echo "修正内容:"
    echo "  peer_init_url: Optional[str]"
    echo "  → peer_init_url: Optional[str] = None"
    echo ""
    echo "次のステップ:"
    echo "  1. vLLMサーバーを再起動してください"
    echo "  2. Controllerのログでエラーが解消されているか確認してください"
else
    echo "エラー: 修正が適用されませんでした"
    echo "バックアップから復元しますか？ (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        cp "$BACKUP_FILE" "$MESSAGE_FILE"
        echo "バックアップから復元しました"
    fi
    exit 1
fi

