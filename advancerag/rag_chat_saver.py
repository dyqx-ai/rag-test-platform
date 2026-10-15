from typing import Optional

from config.config_load import config_load
from myproject.conversation_save.choose_conversation_save_method import choose_conversation_save_method


class RagChatSaver:
    """对话历史持久化封装：受yaml配置全局开关管控"""

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.cfg = config_load(yaml_path)
        self.enable_save = self.cfg["save_method"]["if_save"]

    def save_chat(self, user_query: str, answer: str, session_id: Optional[str], persist_time: Optional[int] = None):
        """保存单轮对话；配置关闭或无session_id时直接跳过"""
        if not self.enable_save:
            return
        if not session_id or not session_id.strip():
            return

        try:
            chat_pair = [f"human:{user_query}", f"ai:{answer}"]
            store_handler = choose_conversation_save_method(
                self.yaml_path,
                session_id,
                persist_time,
                chat_pair
            )
            if hasattr(store_handler, "redis_save"):
                store_handler.redis_save()
            elif hasattr(store_handler, "sqlite_save"):
                store_handler.sqlite_save()
        except Exception as e:
            print(f"【会话持久化失败】{str(e)}")
