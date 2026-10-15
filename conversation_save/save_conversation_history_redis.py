import os
import redis
from typing import Optional, List, Union
from langchain_core.messages import HumanMessage, AIMessage
try:
    from langchain.storage import RedisStore
except ImportError:
    from langchain_community.storage import RedisStore
from config.config_load import config_load

# ====================== 全局配置与连接池（基于 __file__ 计算绝对路径） ======================
# 修复：使用 config_load() 无参调用，自动基于 __file__ 计算 config/config.yaml 绝对路径
try:
    redis_config = config_load()["save_method"]["redis"]
    print(f"[RAG启动] Redis 配置加载成功: {redis_config['host']}:{redis_config['port']}")
except Exception as e:
    raise RuntimeError(f"【RAG系统】Redis 配置加载失败，服务无法启动: {e}")

# 全局只初始化一次连接池
POOL = redis.ConnectionPool(
    host=redis_config["host"],
    port=redis_config["port"],
    db=redis_config["db"],
    password=redis_config["password"] if redis_config["password"] else None,
    decode_responses=redis_config["decode_responses"],
    max_connections=redis_config["max_connections"]
)


class SaveChatRedis:
    """Redis 会话持久化管理器"""

    def __init__(self, sessionID: str = "",
                 persist_time: Optional[int] = None,
                 chat_data: Optional[list] = None):
        self.sessionID = sessionID
        self.persist_time = persist_time if persist_time is not None else redis_config.get("persist_time")
        self.chat_data = chat_data if chat_data is not None else []
        # 复用全局连接池
        try:
            self.rd = redis.StrictRedis(connection_pool=POOL)
        except Exception as e:
            raise RuntimeError(f"【RAG系统】Redis 客户端初始化失败: {e}")

    @staticmethod
    def get_langchain_redis_store(yaml_config: dict) -> RedisStore:
        """
        公用静态方法：获取 LangChain 专用 RedisStore 实例
        统一处理配置读取、url 拼接，外部文件直接调用，无需重复写逻辑
        """
        try:
            redis_conf = yaml_config["save_method"]["redis"]
            host = redis_conf["host"]
            port = redis_conf["port"]
            db = redis_conf["db"]
            pwd = redis_conf["password"]
            # 拼接带密码/无密码标准 redis url
            if pwd:
                redis_url = f"redis://:{pwd}@{host}:{port}/{db}"
            else:
                redis_url = f"redis://{host}:{port}/{db}"
            return RedisStore(redis_url=redis_url)
        except Exception as e:
            raise RuntimeError(f"【RAG系统】RedisStore 初始化失败: {e}")

    def redis_save(self, sessionID: str = "", persist_time: Optional[int] = None, chat_data: Optional[list] = None):
        """保存会话数据到 Redis"""
        target_sid = sessionID if sessionID else self.sessionID
        target_pt = persist_time if persist_time is not None else self.persist_time
        target_data = chat_data if chat_data is not None else self.chat_data

        if not target_sid:
            print("【RAG警告】redis_save 收到空 sessionID，跳过")
            return None

        try:
            key = f"chat_session:{target_sid}"
            if target_data:
                self.rd.lpush(key, *target_data)
            # 修复：persist_time 为空不执行 expire，避免传 None 报错
            if target_pt is not None:
                self.rd.expire(key, target_pt)
            return SaveChatRedis(target_sid, target_pt, target_data)
        except Exception as e:
            print(f"【RAG错误】Redis 保存会话失败 [{target_sid}]: {e}")
            return None

    def redis_sessionID_delete(self, sessionID: str = ""):
        """删除指定会话"""
        target_sid = sessionID if sessionID else self.sessionID
        if not target_sid:
            print("【RAG警告】redis_sessionID_delete 收到空 sessionID，跳过")
            return
        try:
            target_key = f"chat_session:{target_sid}"
            if self.rd.exists(target_key):
                self.rd.delete(target_key)
                print(f"【RAG会话】Redis 会话删除成功: {target_sid}")
        except Exception as e:
            print(f"【RAG错误】Redis 删除会话失败 [{target_sid}]: {e}")

    def redis_conversation_delete(self, sessionID: str = "", num: int = 5):
        """裁剪会话消息，保留最近 N 条"""
        target_sid = sessionID if sessionID else self.sessionID
        if not target_sid:
            return
        try:
            target_key = f"chat_session:{target_sid}"
            if not self.rd.exists(target_key):
                print("【RAG提示】此会话不存在，无需裁剪")
                return
            reserve_count = abs(num)
            self.rd.ltrim(target_key, 0, reserve_count - 1)
        except Exception as e:
            print(f"【RAG错误】Redis 裁剪会话失败 [{target_sid}]: {e}")

    def check_pool(self):
        """检查连接池状态（调试用）"""
        try:
            pool = self.rd.connection_pool
            pool_info = (
                f"最大连接数: {pool.max_connections},\n"
                f"当前连接数: {pool._num_connections},\n"
                f"最大空闲连接数: {pool._max_idle_connections},\n"
                f"最大空闲时间: {pool.max_idle_time}"
            )
            return pool_info
        except Exception as e:
            return f"【RAG错误】连接池状态检查失败: {e}"

    def get_langchain_history(self, sessionID: str = ""):
        """读取 Redis 对话，返回 LangChain 标准消息列表"""
        target_sid = sessionID if sessionID else self.sessionID
        if not target_sid:
            return []

        try:
            key = f"chat_session:{target_sid}"
            raw_list = self.rd.lrange(key, 0, -1)
            msg_history = []
            for line in raw_list:
                if not line:
                    continue
                try:
                    # 修复：split 增加异常捕获，单条损坏消息不影响整体加载
                    role, content = line.split(":", maxsplit=1)
                except ValueError:
                    print(f"【RAG警告】Redis 消息格式异常，跳过: {line[:50]}...")
                    continue
                if role == "human":
                    msg_history.append(HumanMessage(content=content))
                elif role == "ai":
                    msg_history.append(AIMessage(content=content))
            return msg_history
        except Exception as e:
            print(f"【RAG错误】Redis 读取历史消息失败 [{target_sid}]: {e}")
            return []

    @staticmethod
    def get_all_session_ids() -> List[str]:
        """
        静态方法：获取 Redis 内全部会话 ID
        直接复用全局连接池，无需实例化
        使用 scan_iter 安全遍历，不阻塞 Redis
        """
        session_id_list = []
        prefix = "chat_session:"
        try:
            client = redis.StrictRedis(connection_pool=POOL)
            for key in client.scan_iter(match=f"{prefix}*", count=100):
                sid = key.replace(prefix, "")
                session_id_list.append(sid)
            print(f"【RAG会话】扫描到 {len(session_id_list)} 个活跃会话")
        except Exception as e:
            print(f"【RAG错误】扫描 Redis 会话列表失败: {e}")
        return session_id_list

    @staticmethod
    def test_connection() -> bool:
        """静态方法：测试 Redis 连接是否正常"""
        try:
            client = redis.StrictRedis(connection_pool=POOL)
            client.ping()
            return True
        except Exception as e:
            print(f"【RAG错误】Redis 连接测试失败: {e}")
            return False