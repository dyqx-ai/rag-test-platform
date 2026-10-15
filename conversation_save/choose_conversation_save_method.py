import os
import sys
from typing import Optional

# 确保项目根目录在 Python 路径中
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from config.config_load import config_load
from myproject.conversation_save.save_conversation_history_redis import SaveChatRedis
from myproject.conversation_save.save_conversation_history_sqlite import SaveChatSqlite


def choose_conversation_save_method(
    yaml_path: str,
    sessionID: str,
    persist_time: Optional[int] = None,
    chat_data: Optional[list] = None
):
    """根据配置选择会话保存方式（Redis / SQLite）"""
    try:
        config = config_load(yaml_path)
    except Exception as e:
        raise RuntimeError(f"【RAG系统】choose_conversation_save_method 配置加载失败: {e}")

    save_method = config["save_method"]["method"]
    if save_method == "redis":
        return SaveChatRedis(sessionID, persist_time, chat_data)
    elif save_method == "sqlite":
        return SaveChatSqlite(sessionID, persist_time, chat_data)
    else:
        raise ValueError(f"【RAG系统】不支持的保存方法: {save_method}，可选 redis / sqlite")