from typing import Optional, List, Set
from langchain_core.documents import Document

from myproject.advancerag.post_retrieval import PostRetrieval


class RagPostProcessor:
    """后置检索优化处理器：封装重排+压缩流水线，对外输出纯文本"""

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.post_tool = PostRetrieval(yaml_path)

        # 合法压缩类型白名单
        self.valid_compress_types: Set[str] = {
            "extract",
            "llm_filter",
            "emb_filter",
            "pipeline"
        }
        self.default_compress_type = "pipeline"

    def process_context(self, query: str, retrieve_groups: List[List[Document]], custom_compress_type: Optional[str] = None, enable_rrf: bool = True, enable_model_rerank: bool = True) -> str:
        """
        输入多路检索文档组，输出处理后的纯净上下文文本
        流程：RRF融合 → 重排 → 上下文压缩
        :param query: 用户提问
        :param retrieve_groups: 多路检索文档组
        :param custom_compress_type: 自定义压缩类型
        :param enable_rrf: 是否启用RRF互逆排序融合
        :param enable_model_rerank: 是否启用模型重排
        """
        target_compress = self.default_compress_type
        if custom_compress_type and isinstance(custom_compress_type, str):
            clean_type = custom_compress_type.strip()
            if clean_type in self.valid_compress_types:
                target_compress = clean_type

        try:
            sorted_text = self.post_tool.doc_rerank_all(
                query=query,
                multi_retrieve_groups=retrieve_groups,
                enable_rrf=enable_rrf,
                enable_model_rerank=enable_model_rerank
            )
            final_text = self.post_tool.doc_compress_all(
                query=query,
                source_text=sorted_text,
                compress_type=target_compress
            )
            # 防护1：压缩后为空但重排结果非空，降级使用重排后文本
            if not final_text.strip() and sorted_text.strip():
                print(f"【后置优化】压缩后为空，降级使用重排后文本（长度={len(sorted_text)}）")
                return sorted_text
            # 防护2：重排后也为空，降级使用原始检索文本
            if not final_text.strip():
                print(f"【后置优化】重排+压缩均为空，降级使用原始检索文本")
                all_text = []
                for group in retrieve_groups:
                    for doc in group:
                        all_text.append(doc.page_content.strip())
                return "\n\n".join([t for t in all_text if t])
            return final_text
        except Exception as e:
            print(f"【后置优化异常】{str(e)}，降级使用原始检索文本")
            all_text = []
            for group in retrieve_groups:
                for doc in group:
                    all_text.append(doc.page_content.strip())
            return "\n\n".join([t for t in all_text if t])
