import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
try:
    from langchain_dashscope import DashScopeEmbeddings
except ImportError:
    from langchain_community.embeddings import DashScopeEmbeddings

from config.config_load import config_load, get_project_root

# 修复：明确指定 .env 路径为项目根目录，避免 load_dotenv() 默认从调用者文件位置向上查找时找不到
load_dotenv(Path(get_project_root()) / ".env")


class ModelManager:
    """统一模型管理器：集中管理对话模型、Embedding 模型、Rerank 模型的初始化"""

    def __init__(self, yaml_path: str):
        if not yaml_path:
            raise ValueError("【RAG系统】请提供正确的 yaml 配置文件路径")
        self.yaml_path = yaml_path
        try:
            self.config = config_load(yaml_path)
        except Exception as e:
            raise RuntimeError(f"【RAG系统】ModelManager 配置加载失败: {e}")

        # 环境变量密钥缓存到实例，方便后续拓展自定义密钥
        self.ali_api_key = os.getenv("ALI_API_KEY")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    def init_model(
        self,
        base_url: str,
        model_name: str,
        api_key: str
    ) -> ChatOpenAI:
        """统一初始化兼容 OpenAI 接口的对话模型"""
        if not api_key:
            raise ValueError("【RAG系统】模型 API 密钥不能为空，请检查 .env 文件")
        if not base_url:
            raise ValueError("【RAG系统】模型 base_url 不能为空，请检查 config.yaml")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
        )

    def init_chat_model(self) -> ChatOpenAI:
        """根据配置加载对话大模型"""
        use_model = self.config["llm"]["use_model"]
        if use_model == "qwen":
            return self.init_model(
                base_url=self.config["llm"]["qwen_base_url"],
                model_name=self.config["llm"]["qwen_model"],
                api_key=self.ali_api_key or ""
            )
        elif use_model == "deepseek":
            return self.init_model(
                base_url=self.config["llm"]["deepseek_base_url"],
                model_name=self.config["llm"]["deepseek_model"],
                api_key=self.deepseek_api_key or ""
            )
        else:
            raise ValueError(f"【RAG系统】不支持的对话模型类型：{use_model}，可选 qwen / deepseek")

    def init_rerank_model(self):
        """
        加载重排模型（不走 LangChain，直接调 DashScope 原生 rerank HTTP 接口）
        配置来源：config.yaml 第5节 rerank.cloud
        返回一个提供 rerank(documents, query) -> List[float] 方法的封装对象
        """
        rerank_cfg = self.config.get("rerank", {})
        cloud_cfg = rerank_cfg.get("cloud", {})

        provider = cloud_cfg.get("provider", "aliyun")
        if provider != "aliyun":
            raise ValueError(f"【RAG系统】不支持的重排模型厂商：{provider}，目前仅支持 aliyun")

        # API key 优先级：配置文件 > .env 环境变量
        api_key = cloud_cfg.get("api_key") or self.ali_api_key or os.getenv("ALI_API_KEY") or ""
        model_name = cloud_cfg.get("model", "gte-rerank-v2")
        base_url = cloud_cfg.get("base_url", "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank")
        workspace_id = cloud_cfg.get("workspace_id", "")

        # qwen3-rerank 需要业务空间ID，切换到新的 MaaS API 地址
        if model_name == "qwen3-rerank" and workspace_id:
            base_url = f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks"
            print(f"【RAG启动】检测到 qwen3-rerank + workspace_id，切换到 MaaS API: {base_url}")

        print(f"【RAG启动】Rerank 模型加载: model={model_name}, url={base_url}")

        class DashScopeRerankWrapper:
            """DashScope rerank 原生接口封装（不走 LangChain）"""

            def __init__(self, api_key: str, model: str, base_url: str, model_name: str):
                self.api_key = api_key
                self.model = model
                self.base_url = base_url
                # qwen3-rerank 用兼容模式请求体（顶层 query/documents），gte-rerank-v2 用原生模式（input 嵌套）
                self.is_compatible_mode = (model_name == "qwen3-rerank")

            def rerank(self, documents: List[str], query: str) -> List[float]:
                """
                调用 DashScope rerank 接口，返回每篇文档的相关性分数（与输入顺序一致）
                :param documents: 文档文本列表
                :param query: 查询文本
                :return: 分数列表，长度与 documents 相同，顺序与输入一致
                """
                import requests

                if not documents:
                    return []
                if not self.api_key:
                    print("【RAG重排】API key 为空，跳过 rerank")
                    return [1.0] * len(documents)  # 全部给满分，不影响排序

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                # 根据模型选择请求体格式
                if self.is_compatible_mode:
                    # qwen3-rerank 兼容模式：query 和 documents 在顶层
                    payload = {
                        "model": self.model,
                        "query": query,
                        "documents": documents,  # 字符串数组
                        "top_n": len(documents),
                        "return_documents": False
                    }
                else:
                    # gte-rerank-v2 原生模式：input 嵌套，parameters 嵌套
                    payload = {
                        "model": self.model,
                        "input": {
                            "query": query,
                            "documents": documents  # 字符串数组
                        },
                        "parameters": {
                            "top_n": len(documents),
                            "return_documents": False
                        }
                    }

                try:
                    resp = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
                    if resp.status_code != 200:
                        print(f"【RAG重排】API调用失败: status={resp.status_code}, body={resp.text[:300]}")
                        return [1.0] * len(documents)

                    data = resp.json()
                    # 兼容两种响应格式：原生模式在 output.results，兼容模式在 results
                    if "output" in data:
                        results = data.get("output", {}).get("results", [])
                    else:
                        results = data.get("results", [])

                    # 初始化全0分数，按原始文档顺序填充
                    scores = [0.0] * len(documents)
                    for item in results:
                        idx = item.get("index", 0)
                        score = item.get("relevance_score", 0.0)
                        if 0 <= idx < len(scores):
                            scores[idx] = float(score)

                    avg_score = sum(scores) / len(scores) if scores else 0
                    print(f"【RAG重排】✅ 成功: {len(results)} 篇文档打分, 平均分={avg_score:.4f}")
                    return scores
                except Exception as e:
                    print(f"【RAG重排】HTTP调用异常: {e}")
                    return [1.0] * len(documents)

        return DashScopeRerankWrapper(api_key, model_name, base_url, model_name)

    def init_embedding_model(self):
        """加载嵌入模型（本地 Ollama / 云端 DashScope）"""
        if self.config["embedding"]["use_local"]:
            return OllamaEmbeddings(
                model=self.config["embedding"]["local"]["model"],
                base_url=self.config["embedding"]["local"]["base_url"],
            )
        else:
            return DashScopeEmbeddings(
                dashscope_api_key=self.ali_api_key or "",
                model=self.config["embedding"]["cloud"]["model"],
                base_url=self.config["embedding"]["cloud"]["base_url"],
            )
    def init_multimodal_model(self) -> ChatOpenAI:
        """加载多模态图文模型（兼容DashScope OpenAI兼容接口）"""
        mm_cfg = self.config.get("multimodal_model", {})
        if not mm_cfg:
            raise ValueError("【RAG系统】config.yaml 中缺少 multimodal_model 配置段，请检查配置")
        base_url = mm_cfg.get("base_url", "")
        model_name = mm_cfg.get("model_name", "")
        api_key = self.ali_api_key or ""
        if not model_name:
            raise ValueError("【RAG系统】multimodal_model.model_name 未配置，请检查 config.yaml")
        return self.init_model(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
        )