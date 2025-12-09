#!/usr/bin/env python3
"""
P2P設定の確認スクリプト

このスクリプトは、LMCacheのP2P設定が正しく構成されているかを確認します。
"""

import os
import yaml
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_config_file(config_path: str = "example1.yaml"):
    """設定ファイルを確認する"""
    logger.info("=" * 80)
    logger.info("1. 設定ファイルの確認: %s", config_path)
    logger.info("=" * 80)
    
    if not os.path.exists(config_path):
        logger.error("設定ファイルが見つかりません: %s", config_path)
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # P2P設定の確認
    enable_p2p = config.get("enable_p2p", False)
    logger.info("enable_p2p: %s", enable_p2p)
    
    if not enable_p2p:
        logger.warning("⚠ P2Pが無効になっています")
        return False
    
    p2p_host = config.get("p2p_host")
    p2p_init_ports = config.get("p2p_init_ports")
    p2p_lookup_ports = config.get("p2p_lookup_ports")
    
    logger.info("p2p_host: %s", p2p_host)
    logger.info("p2p_init_ports: %s", p2p_init_ports)
    logger.info("p2p_lookup_ports: %s", p2p_lookup_ports)
    
    if not p2p_host:
        logger.error("✗ p2p_hostが設定されていません")
        return False
    
    if not p2p_init_ports:
        logger.error("✗ p2p_init_portsが設定されていません")
        return False
    
    if not p2p_lookup_ports:
        logger.error("✗ p2p_lookup_portsが設定されていません")
        return False
    
    # peer_init_urlの計算
    if isinstance(p2p_init_ports, list):
        peer_init_url = f"{p2p_host}:{p2p_init_ports[0]}"
    else:
        peer_init_url = f"{p2p_host}:{p2p_init_ports}"
    
    logger.info("✓ 計算されたpeer_init_url: %s", peer_init_url)
    
    # Controller設定の確認
    enable_controller = config.get("enable_controller", False)
    controller_pull_url = config.get("controller_pull_url")
    controller_reply_url = config.get("controller_reply_url")
    
    logger.info("\nController設定:")
    logger.info("  enable_controller: %s", enable_controller)
    logger.info("  controller_pull_url: %s", controller_pull_url)
    logger.info("  controller_reply_url: %s", controller_reply_url)
    
    if not enable_controller:
        logger.warning("⚠ Controllerが無効になっています")
    
    if not controller_pull_url or not controller_reply_url:
        logger.warning("⚠ Controller URLが設定されていません")
    
    return True


def check_server_log(log_path: str = "vllm/server.log"):
    """サーバーログを確認する"""
    logger.info("\n" + "=" * 80)
    logger.info("2. サーバーログの確認: %s", log_path)
    logger.info("=" * 80)
    
    if not os.path.exists(log_path):
        logger.warning("ログファイルが見つかりません: %s", log_path)
        return False
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # P2P関連のログを検索
    p2p_found = False
    peer_init_found = False
    registration_found = False
    
    for line in lines:
        if "enable_p2p" in line and "True" in line:
            p2p_found = True
            logger.info("✓ P2Pが有効になっていることを確認: %s", line.strip()[:100])
        
        if "p2p_host" in line and "192.168" in line:
            logger.info("✓ p2p_host設定を確認: %s", line.strip()[:100])
        
        if "p2p_init_ports" in line:
            logger.info("✓ p2p_init_ports設定を確認: %s", line.strip()[:100])
        
        if "Starting P2P backend" in line:
            p2p_found = True
            logger.info("✓ P2Pバックエンドが起動していることを確認: %s", line.strip())
        
        if "Registering lmcache instance-worker" in line:
            registration_found = True
            logger.info("✓ Worker登録を確認: %s", line.strip())
        
        if "peer_init_url" in line.lower():
            peer_init_found = True
            logger.info("✓ peer_init_url関連のログ: %s", line.strip())
    
    if not p2p_found:
        logger.warning("⚠ P2P関連のログが見つかりません")
    
    if not registration_found:
        logger.warning("⚠ Worker登録のログが見つかりません")
    
    if not peer_init_found:
        logger.warning("⚠ peer_init_urlの明示的なログが見つかりません（これは正常な場合もあります）")
    
    return True


def check_controller_connection(controller_pull_url: str = "192.168.110.13:8300"):
    """Controllerへの接続を確認する"""
    logger.info("\n" + "=" * 80)
    logger.info("3. Controllerへの接続確認")
    logger.info("=" * 80)
    
    try:
        import socket
        host, port = controller_pull_url.split(":")
        port = int(port)
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            logger.info("✓ Controller (%s) への接続が成功しました", controller_pull_url)
            return True
        else:
            logger.error("✗ Controller (%s) への接続に失敗しました (エラーコード: %d)", controller_pull_url, result)
            return False
    except Exception as exc:
        logger.error("✗ Controller接続確認中にエラーが発生しました: %s", exc)
        return False


def main():
    """メイン処理"""
    logger.info("LMCache P2P設定確認スクリプト")
    logger.info("=" * 80)
    
    # 設定ファイルの確認
    config_ok = check_config_file()
    
    # サーバーログの確認
    log_ok = check_server_log()
    
    # Controller接続の確認
    controller_ok = check_controller_connection()
    
    # 結果のまとめ
    logger.info("\n" + "=" * 80)
    logger.info("確認結果のまとめ")
    logger.info("=" * 80)
    
    if config_ok and log_ok and controller_ok:
        logger.info("✓ すべての確認が成功しました")
        logger.info("\n推奨事項:")
        logger.info("  1. vLLMサーバーが起動していることを確認してください")
        logger.info("  2. Controllerが起動していることを確認してください")
        logger.info("  3. HeartbeatMsgのエラーが発生する場合は、")
        logger.info("     worker.pyの120-124行目でp2p_init_urlが正しく設定されているか確認してください")
    else:
        logger.warning("⚠ いくつかの問題が検出されました")
        logger.warning("\nトラブルシューティング:")
        logger.warning("  1. 設定ファイル（example1.yaml）のP2P設定を確認してください")
        logger.warning("  2. vLLMサーバーのログでP2Pバックエンドの起動を確認してください")
        logger.warning("  3. Controllerが起動していることを確認してください")
        logger.warning("  4. エラーログで 'peer_init_url' が欠けているメッセージを確認してください")


if __name__ == "__main__":
    main()

