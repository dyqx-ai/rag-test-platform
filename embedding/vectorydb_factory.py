import os
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from config.config_load import config_load


class VectorDbFactory:
    """向量数据库工厂：统一管理 Chroma/Milvus/Faiss/Redis 向量库初始化"""

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.config = config_load(yaml_path)

    def set_chroma(
            self,
            model=None,  # 外部注入的 embedding 模型对象
            persist_save: bool = False,
            persist_path: str = "",
            collection_name: str = "default_collection",
    ) -> Chroma:
        """
        初始化 Chroma 向量数据库
        :param model: 外部传入的 embedding 模型实例（必填，禁止内部 new）
        :param persist_save: 是否持久化存储
        :param persist_path: 持久化目录路径
        :param collection_name: 集合名称
        """
        if model is None:
            raise ValueError("【VectorDbFactory】embedding 模型不能为空，请外部注入 ModelManager 实例")

        # 持久化路径处理：支持相对路径 → 基于项目根目录计算绝对路径
        if persist_save and persist_path:
            persist_p = Path(persist_path)
            if not persist_p.is_absolute():
                from config.config_load import get_project_root
                persist_p = Path(get_project_root()) / persist_path
            persist_p = persist_p.resolve()
            persist_p.mkdir(parents=True, exist_ok=True)
            persist_dir = str(persist_p)
        else:
            persist_dir = None

        try:
            if persist_save and persist_dir:
                vector_store = Chroma(
                    collection_name=collection_name,
                    embedding_function=model,
                    persist_directory=persist_dir,
                )
            else:
                vector_store = Chroma(
                    collection_name=collection_name,
                    embedding_function=model,
                )
            print(f"[RAG向量库] Chroma 集合初始化成功: {collection_name}")
            return vector_store
        except Exception as e:
            raise RuntimeError(f"【VectorDbFactory】Chroma 初始化失败: {e}")

    def set_milvus(self):
        """初始化 Milvus 向量数据库"""
        raise NotImplementedError("【VectorDbFactory】Milvus 向量库逻辑尚未实现，请切换 db_type 为 chroma")

    def set_faiss(self):
        """初始化 Faiss 向量数据库"""
        raise NotImplementedError("【VectorDbFactory】Faiss 向量库逻辑尚未实现，请切换 db_type 为 chroma")

    def set_redis(self):
        """初始化 Redis 向量数据库"""
        raise NotImplementedError("【VectorDbFactory】Redis 向量库逻辑尚未实现，请切换 db_type 为 chroma")

    def init_vector_store(
            self,
            yaml_path: str = "",
            collection_name: str = "default_collection",
            model=None  # 新增：外部注入的 embedding 模型，避免内部重复创建
    ):
        """
        统一向量库初始化入口
        :param yaml_path: 配置文件路径（可选，默认使用实例化时传入的路径）
        :param collection_name: 集合名称
        :param model: 外部注入的 embedding 模型实例（推荐传入，避免内部重复创建 ModelManager）
        """
        target_yaml = yaml_path or self.yaml_path
        if not target_yaml:
            raise ValueError("【VectorDbFactory】配置文件路径不能为空")

        config = self.config if yaml_path == "" else config_load(target_yaml)
        db_type = config["vector_db"]["db_type"]

        if db_type == "chroma":
            # 优先使用外部传入的 model，兜底才从配置读取（但打印警告）
            if model is None:
                print("【警告】init_vector_store 未传入 model 参数，将内部创建 ModelManager（不推荐）")
                from myproject.models.models import ModelManager
                model = ModelManager(target_yaml).init_embedding_model()

            vector_store = self.set_chroma(
                persist_save=config["vector_db"]["persist_save"],
                persist_path=config["vector_db"]["persist_path"],
                collection_name=collection_name,
                model=model,
            )
            return vector_store
        elif db_type == "milvus":
            return self.set_milvus()
        elif db_type == "faiss":
            return self.set_faiss()
        elif db_type == "redis":
            return self.set_redis()
        else:
            raise ValueError(f"【VectorDbFactory】不支持的向量库类型: {db_type}，可选：chroma/milvus/faiss/redis")

    def get_chroma_client(self) -> Chroma:
        """
        获取 Chroma 原始客户端（用于管理操作：列出集合、删除集合等）
        不依赖具体集合，用于元数据操作
        """
        config = self.config
        persist_path = config["vector_db"]["persist_path"]
        persist_save = config["vector_db"]["persist_save"]

        from config.config_load import get_project_root
        persist_p = Path(persist_path)
        if not persist_p.is_absolute():
            persist_p = Path(get_project_root()) / persist_path
        persist_p = persist_p.resolve()

        try:
            from myproject.models.models import ModelManager
            embed_model = ModelManager(self.yaml_path).init_embedding_model()

            if persist_save:
                persist_p.mkdir(parents=True, exist_ok=True)
                client = Chroma(
                    collection_name="meta_temp",  # 修复：Chroma集合名不允许下划线开头，改为合法名称
                    embedding_function=embed_model,
                    persist_directory=str(persist_p),
                )
            else:
                client = Chroma(
                    collection_name="meta_temp",  # 同上
                    embedding_function=embed_model,
                )
            return client
        except Exception as e:
            raise RuntimeError(f"【VectorDbFactory】获取 Chroma 客户端失败: {e}")