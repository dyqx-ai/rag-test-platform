import os
import pymysql
from typing import List, Dict, Optional
from config.config_load import config_load


class MysqlDataManager:
    """MySQL 数据管理器：知识库映射、会话、消息的持久化存储"""

    def __init__(self, yaml_path: str = ""):
        self.yaml_path = yaml_path
        # 修复：使用 config_load() 无参调用，自动基于 __file__ 计算绝对路径
        try:
            full_config = config_load(yaml_path) if yaml_path else config_load()
        except Exception as e:
            raise RuntimeError(f"【RAG系统】MySQL 配置加载失败: {e}")

        self.config = full_config["save_method"]["mysql"]
        self.conn = None
        self.cursor = None
        self._connect()
        self._init_tables()

    def _connect(self):
        """建立 MySQL 连接，异常捕获"""
        try:
            self.conn = pymysql.connect(
                host=self.config["host"],
                port=self.config["port"],
                user=self.config["user"],
                password=self.config["password"],
                database=self.config["database"],
                charset="utf8mb4",
                autocommit=False  # 手动事务控制
            )
            self.cursor = self.conn.cursor(pymysql.cursors.DictCursor)
            print(f"[RAG启动] MySQL 连接成功: {self.config['host']}:{self.config['port']}/{self.config['database']}")
        except Exception as e:
            print(f"【RAG系统】MySQL 连接失败: {e}")
            raise RuntimeError(f"数据库连接失败: {e}")

    def _ensure_connection(self):
        """断线自动重连：所有数据库操作前执行 ping(reconnect=True)"""
        try:
            if self.conn is None:
                self._connect()
                return
            self.conn.ping(reconnect=True)
        except Exception:
            print("【RAG警告】MySQL 连接断开，尝试重连...")
            try:
                self._connect()
                print("【RAG提示】MySQL 重连成功")
            except Exception as e:
                print(f"【RAG错误】MySQL 重连失败: {e}")
                raise

    def _init_tables(self):
        """自动创建三张核心表，不存在则创建（IF NOT EXISTS + utf8mb4）"""
        self._ensure_connection()
        create_kb_sql = """
        CREATE TABLE IF NOT EXISTS rag_kb_mapping (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            display_name VARCHAR(128) NOT NULL COMMENT '前端展示中文知识库名',
            collection_name VARCHAR(128) NOT NULL COMMENT '向量库底层英文集合名',
            support_methods VARCHAR(256) DEFAULT '' COMMENT '支持的检索方法，逗号分隔：normal,parent_child,preview_question,summary,enrich,hybrid_search',
            create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_deleted TINYINT DEFAULT 0,
            UNIQUE KEY uk_display_name (display_name),
            UNIQUE KEY uk_collection_name (collection_name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库名称映射表';
        """
        create_conv_sql = """
        CREATE TABLE IF NOT EXISTS rag_conversation (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            conversation_id VARCHAR(64) NOT NULL COMMENT '会话唯一ID',
            kb_display_name VARCHAR(128) NOT NULL COMMENT '绑定知识库名称',
            llm_model VARCHAR(64) DEFAULT '' COMMENT '使用模型',
            conversation_title VARCHAR(256) DEFAULT '新对话' COMMENT '会话标题',
            create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            update_time DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            is_deleted TINYINT DEFAULT 0,
            UNIQUE KEY uk_conv_id (conversation_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话主表';
        """
        create_msg_sql = """
        CREATE TABLE IF NOT EXISTS rag_conversation_message (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            conversation_id VARCHAR(64) NOT NULL,
            role VARCHAR(32) NOT NULL COMMENT 'user/assistant',
            content LONGTEXT NOT NULL COMMENT '消息内容',
            create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_conv_id (conversation_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='对话消息明细表';
        """
        try:
            self.cursor.execute(create_kb_sql)
            self.cursor.execute(create_conv_sql)
            self.cursor.execute(create_msg_sql)
            self.conn.commit()
            print("[RAG启动] MySQL 数据表初始化完成")
        except Exception as e:
            self.conn.rollback()
            print(f"【RAG系统】建表失败: {e}")
            raise

        self._migrate_support_methods_column()

    def _migrate_support_methods_column(self):
        """迁移：为已存在的 rag_kb_mapping 表添加 support_methods 字段（兼容旧数据库）"""
        self._ensure_connection()
        try:
            self.cursor.execute("DESCRIBE rag_kb_mapping")
            columns = [row[0] for row in self.cursor.fetchall()]
            if "support_methods" in columns:
                print("[RAG启动] MySQL support_methods 字段已存在，无需迁移")
                return
            
            alter_sql = """
            ALTER TABLE rag_kb_mapping 
            ADD COLUMN support_methods VARCHAR(256) DEFAULT '' 
            COMMENT '支持的检索方法，逗号分隔：normal,parent_child,preview_question,summary,enrich,hybrid_search'
            """
            self.cursor.execute(alter_sql)
            self.conn.commit()
            print("[RAG启动] MySQL support_methods 字段迁移完成")
        except Exception as e:
            self.conn.rollback()
            print(f"【RAG警告】support_methods 字段迁移失败（不影响系统运行）: {e}")

    # ====================== 知识库映射相关 ======================
    def add_kb_mapping(self, display_name: str, collection_name: str, support_methods: str = "") -> bool:
        """
        新增或恢复知识库映射
        修复：原 INSERT IGNORE 在软删除后再次上传会静默跳过，导致前端看不到知识库
        改用 ON DUPLICATE KEY UPDATE：新增时插入，已存在（含软删除）时恢复 is_deleted=0
        :param support_methods: 支持的检索方法，逗号分隔：normal,parent_child,preview_question,summary,enrich,hybrid_search
        """
        if not display_name or not collection_name:
            print("【RAG警告】add_kb_mapping 收到空参数，跳过")
            return False
        self._ensure_connection()
        try:
            sql = """
            INSERT INTO rag_kb_mapping(display_name, collection_name, support_methods)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE is_deleted=0, collection_name=VALUES(collection_name), support_methods=VALUES(support_methods)
            """
            self.cursor.execute(sql, (display_name, collection_name, support_methods))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"【RAG错误】新增知识库映射失败 [{display_name}]: {e}")
            return False

    def get_kb_by_display_name(self, display_name: str) -> Optional[str]:
        """根据中文展示名查询底层集合名"""
        if not display_name:
            return None
        self._ensure_connection()
        try:
            sql = "SELECT collection_name FROM rag_kb_mapping WHERE display_name=%s AND is_deleted=0"
            self.cursor.execute(sql, (display_name,))
            res = self.cursor.fetchone()
            return res["collection_name"] if res else None
        except Exception as e:
            print(f"【RAG错误】查询知识库失败 [{display_name}]: {e}")
            return None

    # 别名兼容：EmbeddingVectorManager 中使用此方法名
    def get_kb_mapping_by_display_name(self, display_name: str) -> Optional[str]:
        """get_kb_by_display_name 的别名，兼容不同调用方"""
        return self.get_kb_by_display_name(display_name)

    def get_all_kb_list(self) -> List[Dict]:
        """获取全部有效知识库列表（仅返回未删除的）"""
        self._ensure_connection()
        try:
            sql = """
            SELECT display_name, collection_name, support_methods, create_time
            FROM rag_kb_mapping WHERE is_deleted=0 ORDER BY create_time DESC
            """
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"【RAG错误】获取知识库列表失败: {e}")
            return []

    def delete_kb_mapping(self, display_name: str) -> bool:
        """软删除知识库映射"""
        if not display_name:
            print("【RAG警告】delete_kb_mapping 收到空参数，跳过")
            return False
        self._ensure_connection()
        try:
            sql = "UPDATE rag_kb_mapping SET is_deleted=1 WHERE display_name=%s"
            self.cursor.execute(sql, (display_name,))
            self.conn.commit()
            print(f"【RAG知识库】软删除成功: {display_name}")
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"【RAG错误】删除知识库失败 [{display_name}]: {e}")
            return False

    # ====================== 会话相关 ======================
    def create_conversation(self, conversation_id: str, kb_display_name: str,
                            llm_model: str = "", title: str = "新对话") -> bool:
        """创建新会话"""
        if not conversation_id or not kb_display_name:
            print("【RAG警告】create_conversation 收到空参数，跳过")
            return False
        self._ensure_connection()
        try:
            sql = """
            INSERT INTO rag_conversation(conversation_id, kb_display_name, llm_model, conversation_title)
            VALUES (%s, %s, %s, %s)
            """
            self.cursor.execute(sql, (conversation_id, kb_display_name, llm_model, title))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"【RAG错误】创建会话失败 [{conversation_id}]: {e}")
            return False

    def get_conversation_list(self) -> List[Dict]:
        """获取全部会话列表（按更新时间倒序，仅返回未删除的）"""
        self._ensure_connection()
        try:
            sql = """
            SELECT conversation_id, conversation_title, kb_display_name, create_time, update_time
            FROM rag_conversation WHERE is_deleted=0 ORDER BY update_time DESC
            """
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        except Exception as e:
            print(f"【RAG错误】获取会话列表失败: {e}")
            return []

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """更新会话标题"""
        if not conversation_id:
            return False
        self._ensure_connection()
        try:
            sql = "UPDATE rag_conversation SET conversation_title=%s WHERE conversation_id=%s AND is_deleted=0"
            self.cursor.execute(sql, (title, conversation_id))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"【RAG错误】更新会话标题失败 [{conversation_id}]: {e}")
            return False

    def delete_conversation(self, conversation_id: str) -> bool:
        """软删除会话"""
        if not conversation_id:
            print("【RAG警告】delete_conversation 收到空参数，跳过")
            return False
        self._ensure_connection()
        try:
            sql = "UPDATE rag_conversation SET is_deleted=1 WHERE conversation_id=%s"
            self.cursor.execute(sql, (conversation_id,))
            self.conn.commit()
            print(f"【RAG会话】软删除成功: {conversation_id}")
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"【RAG错误】删除会话失败 [{conversation_id}]: {e}")
            return False

    # ====================== 消息相关 ======================
    def add_message(self, conversation_id: str, role: str, content: str) -> bool:
        """新增一条消息"""
        if not conversation_id or not role or not content:
            print("【RAG警告】add_message 收到空参数，跳过")
            return False
        self._ensure_connection()
        try:
            sql = "INSERT INTO rag_conversation_message(conversation_id, role, content) VALUES (%s, %s, %s)"
            self.cursor.execute(sql, (conversation_id, role, content))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"【RAG错误】保存消息失败 [{conversation_id}]: {e}")
            return False

    def get_messages_by_conv_id(self, conversation_id: str) -> List[Dict]:
        """根据会话 ID 获取全部历史消息（按时间正序）"""
        if not conversation_id:
            return []
        self._ensure_connection()
        try:
            sql = """
            SELECT role, content, create_time FROM rag_conversation_message
            WHERE conversation_id=%s ORDER BY create_time ASC
            """
            self.cursor.execute(sql, (conversation_id,))
            return self.cursor.fetchall()
        except Exception as e:
            print(f"【RAG错误】获取历史消息失败 [{conversation_id}]: {e}")
            return []

    def close(self):
        """关闭连接"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            print("【RAG提示】MySQL 连接已关闭")
        except Exception as e:
            print(f"【RAG错误】关闭 MySQL 连接失败: {e}")

    @staticmethod
    def test_connection(yaml_path: str = "") -> bool:
        """静态方法：测试 MySQL 连接是否正常（用于启动自检）"""
        try:
            full_config = config_load(yaml_path) if yaml_path else config_load()
            mysql_config = full_config["save_method"]["mysql"]
            conn = pymysql.connect(
                host=mysql_config["host"],
                port=mysql_config["port"],
                user=mysql_config["user"],
                password=mysql_config["password"],
                database=mysql_config["database"],
                charset="utf8mb4",
                connect_timeout=5
            )
            conn.ping()
            conn.close()
            return True
        except Exception as e:
            print(f"【RAG错误】MySQL 连接测试失败: {e}")
            return False