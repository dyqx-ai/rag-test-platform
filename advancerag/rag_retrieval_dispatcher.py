import json
from typing import Optional, List, Dict, Any
from langchain_core.documents import Document

from config.config_load import config_load
from myproject.embedding.vectorydb_factory import VectorDbFactory
from myproject.advancerag.pre_retrieval import pre_retrieval


class RagRetrievalDispatcher:
    """检索调度器：统一管理向量集合初始化、8种预检索分发、基础检索兜底"""

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.cfg = config_load(yaml_path)
        self.pre_retrieve_tool = pre_retrieval(yaml_path)

        # 返回检索器对象的方法映射 + 对应文本参数名
        self.retriever_method_map: Dict[str, Dict[str, Any]] = {
            "parent_child": {
                "func": self.pre_retrieve_tool.parent_child,
                "text_param": "parent_child_text_list"
            },
            "self_query": {
                "func": self.pre_retrieve_tool.self_query,
                "text_param": "self_query_text_list"
            },
            "preview_question": {
                "func": self.pre_retrieve_tool.preview_question_index,
                "text_param": "preview_question_text_list"
            },
            "summary": {
                "func": self.pre_retrieve_tool.summary,
                "text_param": "need_summary_text_list"
            },
            "hybrid_search": {
                "func": self.pre_retrieve_tool.hybrid_search,
                "text_param": "hybrid_search_text_list"
            }
        }

        # 直接返回文档列表的方法映射 + 对应文本参数名
        self.direct_doc_method_map: Dict[str, Dict[str, Any]] = {
            "multi_query": {
                "func": self.pre_retrieve_tool.multi_query,
                "text_param": "multi_query_text_list"
            },
            "decomposition": {
                "func": self.pre_retrieve_tool.decomposition,
                "text_param": "decomposition_text_list"
            }
        }

        self.ENRICH_NAME = "enrich"

    def _ensure_collection_and_docs(self, collection_name: str, init_docs: Optional[List[Document]]) -> bool:
        """
        统一向量集合初始化入口
        规则：有文本就追加入库；无文本则校验集合是否存在
        返回True=可正常检索，返回False=无法检索
        """
        collection_exists = self.pre_retrieve_tool._collection_exists(collection_name)
        vs = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)

        if init_docs and len(init_docs) > 0:
            vs.add_documents(init_docs)
            return True
        else:
            return collection_exists

    def _get_base_single_retrieve_groups(self, query: str, collection_name: str, init_docs: Optional[List[Document]]) -> List[List[Document]]:
        """基础单路向量检索，包装为二维列表兼容后处理"""
        can_retrieve = self._ensure_collection_and_docs(collection_name, init_docs)
        if not can_retrieve:
            raise RuntimeError("COLLECTION_NOT_EXIST_NO_INIT_DOC")

        vs = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)
        top_k = self.pre_retrieve_tool.config["vector_db"]["search_top_k"]
        base_retriever = vs.as_retriever(search_kwargs={"k": top_k})
        single_docs = base_retriever.invoke(query)
        return [single_docs]

    def dispatch_pre_retrieval(self, query: str, pre_methods: List[str], collection_name: str, init_docs: Optional[List[Document]]) -> List[List[Document]]:
        """根据策略列表分发执行预检索，合并多路结果"""
        retrieve_groups: List[List[Document]] = []
        enrich_needed = self.ENRICH_NAME in pre_methods
        normal_methods = [m for m in pre_methods if m != self.ENRICH_NAME]

        # 先统一初始化基础集合
        can_retrieve = self._ensure_collection_and_docs(collection_name, init_docs)
        if not can_retrieve:
            raise RuntimeError("COLLECTION_NOT_EXIST_NO_INIT_DOC")

        # enrich 字段增强：最先执行，补全用户 query 中的结构化字段，补全后的完整 query 用于后续所有检索
        if enrich_needed:
            try:
                # enrich 入库时已通过 enrich_index_build 从文档提取字段模板存入 enrich_template_collection
                enrich_collection = f"{collection_name}_enrich_template_collection"
                enrich_result = self.pre_retrieve_tool.enrich(
                    user_query=query,
                    collection_name=enrich_collection
                )
                if enrich_result.get("isComplete") and enrich_result.get("content"):
                    query = str(enrich_result["content"])  # 用补全后的结构化数据替换原始 query
                    print(f"【RAG-enrich】字段增强完成，query 已补全为结构化数据，用于后续检索")
                else:
                    print(f"【RAG-enrich】字段增强未完成，使用原始 query 继续检索")
            except Exception as e:
                print(f"【RAG-enrich】字段增强执行失败，跳过 enrich 逻辑: {e}")

        # 执行直接返回文档列表的方法
        for method_name in self.direct_doc_method_map:
            if method_name not in normal_methods:
                continue
            try:
                method_info = self.direct_doc_method_map[method_name]
                func = method_info["func"]
                param_name = method_info["text_param"]
                kwargs = {
                    "query": query,
                    "collection_name": collection_name,
                    param_name: init_docs
                }
                docs = func(**kwargs)
                if docs:
                    retrieve_groups.append(docs)
            except Exception as e:
                print(f"【{method_name} 预检索执行失败】{str(e)}")
                continue

        # 执行返回检索器的方法
        for method_name in self.retriever_method_map:
            if method_name not in normal_methods:
                continue
            try:
                method_info = self.retriever_method_map[method_name]
                func = method_info["func"]
                param_name = method_info["text_param"]
                kwargs = {
                    "collection_name": collection_name,
                    param_name: init_docs
                }
                retriever = func(**kwargs)
                docs = retriever.invoke(query)
                if docs:
                    retrieve_groups.append(docs)
            except Exception as e:
                print(f"【{method_name} 预检索执行失败】{str(e)}")
                continue

        # 全部失败兜底到基础检索
        if not retrieve_groups:
            return self._get_base_single_retrieve_groups(query, collection_name, init_docs)
        return retrieve_groups

    def retrieve(self, query: str, use_pre_retrieval: bool, pre_methods: List[str], collection_name: str, init_docs: Optional[List[Document]]) -> List[List[Document]]:
        """对外统一检索入口"""
        if use_pre_retrieval:
            return self.dispatch_pre_retrieval(query, pre_methods, collection_name, init_docs)
        else:
            return self._get_base_single_retrieve_groups(query, collection_name, init_docs)
