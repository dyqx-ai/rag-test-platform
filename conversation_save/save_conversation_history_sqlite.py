import sqlite3
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from config.config_load import config_load

# 读取yaml配置（相对路径自动基于项目根目录解析）
try:
    sql_config = config_load()
    DB_PATH = sql_config["save_method"]["sqlite"]["db_path"]
except Exception as e:
    raise RuntimeError(f"【RAG系统】SQLite 配置加载失败: {e}")

class SaveChatSqlite:
    def __init__(self, sessionID: str, persist_time: Optional[int] = None, chat_data: Optional[list] = None):
        self.sessionID = sessionID
        self.persist_time = persist_time
        self.chat_data = chat_data if chat_data is not None else []
        self.db_path = DB_PATH
        self._init_table()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS chat_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()
        conn.close()

    def sqlite_save(self, sessionID: str = "", persist_time: Optional[int] = None, chat_data: Optional[list] = None):
        target_sid = sessionID if sessionID else self.sessionID
        target_pt = persist_time if persist_time is not None else self.persist_time
        target_data = chat_data if chat_data is not None else self.chat_data

        if not target_data:
            return self

        conn = self._get_conn()
        cur = conn.cursor()
        batch = []
        for item in target_data:
            try:
                role, content = item.split(":", maxsplit=1)
            except ValueError:
                print(f"【RAG警告】SQLite 消息格式异常，跳过: {str(item)[:50]}...")
                continue
            batch.append((target_sid, role, content))
        cur.executemany("INSERT INTO chat_records(session_id,role,content) VALUES (?,?,?)", batch)
        conn.commit()
        conn.close()
        return SaveChatSqlite(target_sid, target_pt, target_data)

    def sqlite_sessionID_delete(self, sessionID: str = ""):
        target_sid = sessionID if sessionID else self.sessionID
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_records WHERE session_id = ?", (target_sid,))
        conn.commit()
        conn.close()

    def sqlite_conversation_delete(self, sessionID: str = "", num: int = 5):
        target_sid = sessionID if sessionID else self.sessionID
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(id) AS total FROM chat_records WHERE session_id = ?", (target_sid,))
        total = cur.fetchone()["total"]
        if total == 0:
            print("此会话并不存在")
            conn.close()
            return
        keep = abs(num)
        if total <= keep:
            conn.close()
            return
        del_num = total - keep
        cur.execute('''
        DELETE FROM chat_records
        WHERE id IN (SELECT id FROM chat_records WHERE session_id=? ORDER BY id ASC LIMIT ?)
        ''', (target_sid, del_num))
        conn.commit()
        conn.close()

    def check_sqlite_info(self):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT session_id) AS session_cnt FROM chat_records")
        session_cnt = cur.fetchone()["session_cnt"]
        cur.execute("SELECT COUNT(id) AS msg_cnt FROM chat_records")
        msg_cnt = cur.fetchone()["msg_cnt"]
        conn.close()
        info = (
            f"数据库文件路径: {self.db_path}\n"
            f"总会话数量: {session_cnt}\n"
            f"全部消息总数: {msg_cnt}"
        )
        return info

    def get_langchain_history(self, sessionID: str = "") -> List:
        target_sid = sessionID if sessionID else self.sessionID
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute('''
        SELECT role, content FROM chat_records WHERE session_id=? ORDER BY id ASC
        ''', (target_sid,))
        rows = cur.fetchall()
        conn.close()
        history = []
        for r in rows:
            if r["role"] == "human":
                history.append(HumanMessage(content=r["content"]))
            elif r["role"] == "ai":
                history.append(AIMessage(content=r["content"]))
        return history
