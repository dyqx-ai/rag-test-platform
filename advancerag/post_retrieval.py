import chromadb
from typing import Optional, List, Tuple, Dict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import (
    LLMChainExtractor,
    LLMChainFilter,
    DocumentCompressorPipeline,
    EmbeddingsFilter
)
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.load import dumps, loads

# 项目内部依赖（和pre_retrieval完全一致）
from config.config_load import config_load
from myproject.models.models import ModelManager
from myproject.embedding.vectorydb_factory import VectorDbFactory


class PostRetrieval:
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.config = config_load(yaml_path)

        # 统一通过ModelManager初始化所有模型
        model_manager = ModelManager(self.yaml_path)
        self.llm_client = model_manager.init_chat_model()
        self.embedding_client = model_manager.init_embedding_model()
        self.rerank_client = model_manager.init_rerank_model()

        # ========== 读取配置参数 ==========
        # RAG总开关
        self.enable_multi_query = self.config["rag"]["enable_multi_query"]
        self.enable_context_compress = self.config["rag"]["enable_context_compress"]
        self.enable_rerank = self.config["rerank"]["enable"]

        # 多路检索参数
        self.multi_query_count = self.config["rag"]["multi_query_count"]
        self.search_top_k = self.config["vector_db"]["search_top_k"]
        self.rrf_smooth_k = self.config["rag"].get("rrf_smooth_k", 60)

        # 上下文压缩参数
        self.compress_chunk_size = self.config["split"].get("compress_chunk_size", 300)
        self.embed_sim_threshold = self.config["vector_db"].get("embed_sim_threshold", 0.6)

        # 重排参数
        self.rerank_threshold = self.config["rerank"]["threshold"]
        self.rerank_top_n = self.config["rerank"]["top_n"]

        # 多路查询生成提示词
        self.multi_query_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是专业的查询扩展助手，基于用户原始问题生成多个不同表述的检索查询，仅输出{count}条，每行一条，无序号、无解释、无多余文字。"),
            ("human", "原始问题：{original_query}\n请输出{count}个检索查询：")
        ])
        self._build_multi_query_chain()

    def _build_multi_query_chain(self):
        """构建多路查询生成链路"""
        self.multi_query_chain = (
            self.multi_query_prompt.partial(count=self.multi_query_count)
            | self.llm_client
            | StrOutputParser()
            | RunnableLambda(lambda x: [q.strip() for q in x.split("\n") if q.strip()])
        )

    def _collection_exists(self, collection_name: str) -> bool:
        """
        判断Chroma集合是否存在
        修复：相对路径必须解析为绝对路径，与 vectorydb_factory.py 保持一致，
        否则 PersistentClient 会按 CWD 解析，指向与入库时不同的目录，导致始终查询不到集合
        """
        from pathlib import Path
        from config.config_load import get_project_root
        persist_path = self.config["vector_db"]["persist_path"]
        persist_p = Path(persist_path)
        if not persist_p.is_absolute():
            persist_p = Path(get_project_root()) / persist_path
        persist_p = persist_p.resolve()
        client = chromadb.PersistentClient(path=str(persist_p))
        # 兼容不同版本 chromadb：新版返回字符串，旧版返回带 .name 的集合对象
        raw_collections = client.list_collections()
        exist_names = []
        for c in raw_collections:
            if isinstance(c, str):
                exist_names.append(c)
            elif hasattr(c, 'name'):
                exist_names.append(c.name)
            else:
                exist_names.append(str(c))
        return collection_name in exist_names

    def _join_text_list(self, text_list: List[str]) -> str:
        """工具方法：文本列表拼接为单段规整纯文本，自动过滤空值，分段分隔"""
        valid_texts = [t.strip() for t in text_list if t.strip()]
        return "\n\n".join(valid_texts)

    # ====================== 重排序模块：2个独立私有函数 + 统一入口，输出拼接好的完整文本 ======================
    def _rrf_fusion_sort(self, results: List[List[Document]]) -> List[Document]:
        """私有函数1：RRF互逆排序融合，内部返回文档列表"""
        fused_scores: Dict[str, float] = {}
        for doc_group in results:
            for rank, doc in enumerate(doc_group, start=1):
                try:
                    doc_key = dumps(doc)
                except Exception:
                    doc_key = doc.page_content
                if doc_key not in fused_scores:
                    fused_scores[doc_key] = 0.0
                fused_scores[doc_key] += 1.0 / (rank + self.rrf_smooth_k)
        sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        doc_list = []
        for doc_key, _ in sorted_items:
            try:
                doc = loads(doc_key)
            except Exception:
                doc = Document(page_content=doc_key)
            doc_list.append(doc)
        return doc_list

    def _ali_rerank_sort(self, query: str, docs: List[Document], enable_model_rerank: bool = True) -> List[Document]:
        """私有函数2：调用 DashScope rerank 接口重排，带容错兜底"""
        if not enable_model_rerank or not docs:
            return docs[:self.rerank_top_n]
        try:
            # 优先使用 rerank 接口（DashScopeRerankWrapper 提供 rerank 方法）
            if hasattr(self.rerank_client, "rerank"):
                text_arr = [d.page_content for d in docs]
                score_arr = self.rerank_client.rerank(text_arr, query)
                pair = list(zip(docs, score_arr))
                pair.sort(key=lambda x: x[1], reverse=True)
                ranked_docs = []
                for doc, score in pair:
                    doc.metadata["relevance_score"] = float(score)
                    ranked_docs.append(doc)
            elif hasattr(self.rerank_client, "compress_documents"):
                ranked_docs = self.rerank_client.compress_documents(docs, query)
            else:
                # 兜底：无可用重排方法，直接返回原始文档前top_n
                print("【RAG重排】rerank_client 无可用重排方法，跳过重排")
                return docs[:self.rerank_top_n]

            # 防护1：rerank/compress 返回空，降级返回原始文档
            if not ranked_docs:
                print("【RAG重排】重排返回空列表，降级使用全部原始文档")
                return docs

            # 过滤低分文档并截断
            has_score = any("relevance_score" in d.metadata for d in ranked_docs)
            filter_docs = []
            if has_score:
                for d in ranked_docs:
                    score = d.metadata.get("relevance_score", 0.0)
                    if score >= self.rerank_threshold:
                        filter_docs.append(d)
            else:
                # 无分数信息：compress_documents 已做相关性判断，直接保留
                filter_docs = ranked_docs

            # 防护2：分数过滤后为空则降级保留全部重排文档，避免上下文丢失
            if not filter_docs:
                print(f"【RAG重排】threshold={self.rerank_threshold} 过滤后为空，降级保留全部重排文档")
                filter_docs = ranked_docs
            return filter_docs[:self.rerank_top_n]
        except Exception as e:
            print(f"【RAG重排】异常降级: {str(e)}，返回全部原始文档")
            return docs

    def doc_rerank_all(self, query: str, multi_retrieve_groups: Optional[List[List[Document]]] = None, raw_docs: Optional[List[Document]] = None, enable_rrf: bool = True, enable_model_rerank: bool = True) -> str:
        """
        重排序统一入口，对外直接返回拼接完成的纯文本字符串
        :param query: 用户提问
        :param multi_retrieve_groups: 多路检索文档组（用于RRF融合）
        :param raw_docs: 原始文档列表（单路检索时使用）
        :param enable_rrf: 是否启用RRF互逆排序融合（多路检索时有效）
        :param enable_model_rerank: 是否启用模型重排
        :return: 处理后的纯文本字符串
        """
        step_docs: List[Document]
        if multi_retrieve_groups and len(multi_retrieve_groups) > 0:
            if enable_rrf:
                step_docs = self._rrf_fusion_sort(multi_retrieve_groups)
            else:
                step_docs = []
                seen_docs = set()
                for group in multi_retrieve_groups:
                    for doc in group:
                        try:
                            doc_key = dumps(doc)
                        except Exception:
                            doc_key = doc.page_content
                        if doc_key not in seen_docs:
                            seen_docs.add(doc_key)
                            step_docs.append(doc)
        elif raw_docs and len(raw_docs) > 0:
            step_docs = raw_docs
        else:
            return ""

        sorted_docs = self._ali_rerank_sort(query, step_docs, enable_model_rerank=enable_model_rerank)

        text_list = [d.page_content for d in sorted_docs]
        return self._join_text_list(text_list)

    # ====================== 压缩模块：4个独立私有函数 + 统一入口，输出拼接好的完整文本 ======================
    def _compress_llm_extract(self, query: str, docs: List[Document]) -> List[Document]:
        """压缩方法1：LLM精准提取片段"""
        compressor = LLMChainExtractor.from_llm(self.llm_client)
        return compressor.compress_documents(docs, query)

    def _compress_llm_filter(self, query: str, docs: List[Document]) -> List[Document]:
        """压缩方法2：LLM过滤无关文档"""
        compressor = LLMChainFilter.from_llm(self.llm_client)
        return compressor.compress_documents(docs, query)

    def _compress_emb_filter(self, query: str, docs: List[Document]) -> List[Document]:
        """压缩方法3：轻量化向量相似度过滤"""
        compressor = EmbeddingsFilter(embeddings=self.embedding_client, similarity_threshold=self.embed_sim_threshold)
        return compressor.compress_documents(docs, query)

    def _compress_pipeline_mix(self, query: str, docs: List[Document]) -> List[Document]:
        """压缩方法4：完整混合流水线压缩"""
        splitter = CharacterTextSplitter(chunk_size=self.compress_chunk_size, chunk_overlap=0, separator=". ")
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embedding_client)
        rel_filter = EmbeddingsFilter(embeddings=self.embedding_client, similarity_threshold=self.embed_sim_threshold)
        llm_filter = LLMChainFilter.from_llm(self.llm_client)
        pipeline = DocumentCompressorPipeline(transformers=[splitter, redundant_filter, rel_filter, llm_filter])
        return pipeline.compress_documents(docs, query)

    def doc_compress_all(self, query: str, source_text: str, compress_type: str = "pipeline") -> str:
        """压缩统一入口：入参为一段完整文本，输出处理后的单段纯文本"""
        if not self.enable_context_compress or not source_text.strip():
            return source_text
        # 文本转Document内部处理
        temp_doc = Document(page_content=source_text)
        temp_docs = [temp_doc]
        try:
            compressed_docs: List[Document]
            if compress_type == "extract":
                compressed_docs = self._compress_llm_extract(query, temp_docs)
            elif compress_type == "llm_filter":
                compressed_docs = self._compress_llm_filter(query, temp_docs)
            elif compress_type == "emb_filter":
                compressed_docs = self._compress_emb_filter(query, temp_docs)
            elif compress_type == "pipeline":
                compressed_docs = self._compress_pipeline_mix(query, temp_docs)
            else:
                return source_text
            text_list = [d.page_content for d in compressed_docs]
            result = self._join_text_list(text_list)
            # 防护：压缩后为空（LLM/向量过滤过于激进可能丢弃全部文档），降级返回原始文本
            if not result.strip():
                print(f"【RAG压缩】compress_type={compress_type} 压缩后为空，降级返回原始文本（长度={len(source_text)}）")
                return source_text
            return result
        except Exception:
            return source_text

    # ====================== 多路检索底层方法 ======================
    def multi_query_retrieve(self, query: str, collection_name: str, init_texts: Optional[List[str]] = None) -> List[List[Document]]:
        """多路查询检索，返回多路Document二维列表（仅内部流转使用）"""
        if not self._collection_exists(collection_name):
            if not init_texts:
                raise ValueError(f"向量集合 {collection_name} 不存在，请传入 init_texts 初始化")
            init_docs = [Document(page_content=t) for t in init_texts]
            vector_store = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)
            vector_store.add_documents(init_docs)
        else:
            vector_store = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)
        base_retriever = vector_store.as_retriever(search_kwargs={"k": self.search_top_k})

        if not self.enable_multi_query:
            single_docs = base_retriever.invoke(query)
            return [single_docs]

        try:
            expand_queries = self.multi_query_chain.invoke({"original_query": query})
            if not expand_queries:
                expand_queries = [query]
        except Exception:
            expand_queries = [query]
        all_doc_groups = base_retriever.map().invoke(expand_queries)
        return all_doc_groups

    # ====================== 完整一站式总入口：输入输出全部为纯文本字符串 ======================
    def run_full_pipeline(
            self,
            query: str,
            collection_name: str,
            init_texts: Optional[List[str]] = None,
            compress_type: str = "pipeline",
            enable_rrf: bool = True,
            enable_model_rerank: bool = True
    ) -> str:
        """
        完整后置检索流程
        入参：普通字符串、文本列表；返回**单段纯文本**，可直接丢给LLM生成回答
        :param query: 用户提问文本
        :param collection_name: 向量集合名称
        :param init_texts: 集合不存在时，初始化向量库的文本列表
        :param compress_type: 压缩模式：extract / llm_filter / emb_filter / pipeline
        :param enable_rrf: 是否启用RRF互逆排序融合
        :param enable_model_rerank: 是否启用模型重排
        :return: 经过多路检索-RRF融合-重排-压缩后的完整上下文纯文本
        """
        # 1. 多路检索，获取文档组
        retrieve_groups = self.multi_query_retrieve(query, collection_name, init_texts)
        if len(retrieve_groups) == 0:
            return ""
        # 2. 重排序处理，输出规整纯文本
        sorted_context = self.doc_rerank_all(query, multi_retrieve_groups=retrieve_groups, enable_rrf=enable_rrf, enable_model_rerank=enable_model_rerank)
        if not sorted_context.strip():
            return ""
        # 3. 压缩精炼，输出最终可直接投喂大模型的文本
        final_context = self.doc_compress_all(query, source_text=sorted_context, compress_type=compress_type)
        return final_context
