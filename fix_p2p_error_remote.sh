#!/bin/bash
# リモートサーバーでのP2P設定エラー修正スクリプト
# 使用方法: ./fix_p2p_error_remote.sh <remote_host> <venv_path>

REMOTE_HOST="${1:-192.168.110.97}"
VENV_PATH="${2:-/home/nakaolab/kondo/LMCache/.venv}"
MESSAGE_FILE="${VENV_PATH}/lib/python3.12/site-packages/lmcache/v1/cache_controller/message.py"

echo "リモートサーバーでのLMCache P2P設定エラー修正スクリプト"
echo "=================================================="
echo "対象ホスト: $REMOTE_HOST"
echo "仮想環境パス: $VENV_PATH"
echo ""

# リモートサーバーで修正を実行
ssh "$REMOTE_HOST" << EOF
    if [ ! -f "$MESSAGE_FILE" ]; then
        echo "エラー: ファイルが見つかりません: $MESSAGE_FILE"
        exit 1
    fi

    # バックアップを作成
    BACKUP_FILE="${MESSAGE_FILE}.backup.\$(date +%Y%m%d_%H%M%S)"
    cp "$MESSAGE_FILE" "\$BACKUP_FILE"
    echo "✓ バックアップを作成しました: \$BACKUP_FILE"

    # 修正を適用
    sed -i 's/peer_init_url: Optional\[str\]\$/peer_init_url: Optional[str] = None/' "$MESSAGE_FILE"

    # 修正を確認
    if grep -q "peer_init_url: Optional\[str\] = None" "$MESSAGE_FILE"; then
        echo "✓ 修正が正常に適用されました"
        echo ""
        echo "修正内容:"
        echo "  peer_init_url: Optional[str]"
        echo "  → peer_init_url: Optional[str] = None"
    else
        echo "エラー: 修正が適用されませんでした"
        exit 1
    fi
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ リモートサーバーでの修正が完了しました"
    echo ""
    echo "次のステップ:"
    echo "  1. リモートサーバー（$REMOTE_HOST）のvLLMサーバーを再起動してください"
    echo "  2. Controllerのログでエラーが解消されているか確認してください"
else
    echo ""
    echo "エラー: リモートサーバーでの修正に失敗しました"
    echo "手動で修正する場合は、以下のコマンドをリモートサーバーで実行してください:"
    echo "  sed -i 's/peer_init_url: Optional\\[str\\]\$/peer_init_url: Optional[str] = None/' $MESSAGE_FILE"
fi

