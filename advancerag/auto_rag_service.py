from typing import Optional, List, Union
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from config.config_load import config_load
from myproject.models.models import ModelManager

from myproject.advancerag.rag_decision_engine import RagDecisionEngine
from myproject.advancerag.rag_retrieval_dispatcher import RagRetrievalDispatcher
from myproject.advancerag.rag_post_processor import RagPostProcessor
from myproject.advancerag.rag_chat_saver import RagChatSaver
from myproject.datawash.data_wash import TextTanckle


class AutoRagAutoDispatchService:
    """
    全自动RAG调度顶层服务
    职责：仅做全链路流程编排，所有子能力由独立模块提供
    对外唯一入口：run_rag_task()
    """

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.cfg = config_load(yaml_path)
        self.model_manager = ModelManager(self.yaml_path)
        # 普通文本模型
        self.generate_llm_text = self.model_manager.init_chat_model()
        # 多模态图文模型：初始化失败时降级为None，不阻断服务启动
        try:
            self.multimodal_llm = self.model_manager.init_multimodal_model()
        except Exception as e:
            print(f"【RAG启动】多模态模型初始化失败，图文问答将降级为纯文本模式：{str(e)}")
            self.multimodal_llm = None
        self.output_parser = StrOutputParser()

        # 组合所有独立能力模块
        self.decision_engine = RagDecisionEngine(yaml_path)
        self.retrieval_dispatcher = RagRetrievalDispatcher(yaml_path)
        self.post_processor = RagPostProcessor(yaml_path)
        self.chat_saver = RagChatSaver(yaml_path)

        # 全局默认参数（第一层默认值）
        self.default_collection_name = "default_rag_collection"
        self.default_persist_time = None
        self.img_list_to_base64 = TextTanckle()
        # 单次最大传入图片数量，和配置对齐（配置字段名max_image_num）
        self.max_send_img = self.cfg.get("multimodal_model", {}).get("max_image_num", 3)

    def _build_generate_prompt(
            self,
            retrieve_groups: List[List[Document]],
            user_query: str,
            context_text: str
    ) -> Union[str, List[dict]]:
        """
        :param retrieve_groups: 原始检索返回的Document组（用于提取图片元数据）
        :param user_query: 用户提问
        :param context_text: 经过后置优化的上下文文本（用于填充prompt）
        :return: 无图片返回文本str；存在图片返回标准messages数组
        """
        # 从原始Document中提取图片元数据
        img_info_list: List[str] = []
        for group in retrieve_groups:
            for doc in group:
                # 容错：使用get，避免键不存在报错；兼容旧文档无img_id_list字段
                img_list = doc.metadata.get("img_id_list", [])
                if isinstance(img_list, list):
                    img_info_list.extend(img_list)

        prompt_template = """你是专业知识库问答助手，请严格遵循规则作答：
1. 答案仅允许来源于【检索参考文本上下文】和本次附带图片内可见内容，严禁编造不存在信息；
2. 若文字资料与图片均无相关内容，直接回复：知识库暂无相关信息，禁止随意猜测；
3. 需要结合文本上下文与全部图片综合推理，图片中的表格、示意图、标注文字都作为有效参考；
4. 回答逻辑清晰，如果结论主要来自图片内容，可以标注【参考附图信息】。

【检索参考文本上下文】
{context}

【用户提问】
{question}
"""
        prompt_text = prompt_template.format(context=context_text, question=user_query)

        # 无图片场景：直接返回纯文本字符串
        if not img_info_list:
            return prompt_text

        # 图片处理：去重 + 截断数量上限
        unique_img_paths = list(set(img_info_list))[:self.max_send_img]
        img_base64_list = self.img_list_to_base64.img_to_base64(unique_img_paths)
        # 过滤读取失败返回None的图片（img_to_base64内部已跳过失败文件，此过滤为双重保险）
        img_base64_list = [b64 for b64 in img_base64_list if b64]

        # 无有效图片可读取：降级为纯文本
        if not img_base64_list:
            return prompt_text

        # 组装标准多模态messages结构
        content_arr: List[dict] = [
            {
                "type": "text",
                "text": prompt_text
            }
        ]
        for b64_data in img_base64_list:
            content_arr.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
            })

        messages: List[dict] = [
            {
                "role": "user",
                "content": content_arr
            }
        ]
        return messages

    def run_full_auto_rag(
            self,
            user_query: str,
            collection_name: str,
            init_docs: Optional[List[Document]] = None,
            custom_compress_type: Optional[str] = None,
            session_id: Optional[str] = None,
            persist_time: Optional[int] = None,
            user_pre_methods: Optional[List[str]] = None,
            enable_rrf: bool = True,
            enable_model_rerank: bool = True
    ) -> str:
        # Step1：获取预检索策略（传入用户选择的预检索方法，作为最高优先级）
        decision = self.decision_engine.decide_strategy(user_query, user_selected_methods=user_pre_methods)
        print(f"【RAG流程-Step1决策】use_pre_retrieval={decision.get('use_pre_retrieval')}, pre_methods={decision.get('pre_methods')}, 用户指定={user_pre_methods}")

        # Step2：执行检索
        # 调试日志：打印传入的集合名称和实际可用的集合
        try:
            import chromadb
            from pathlib import Path
            persist_path = self.cfg["vector_db"]["persist_path"]  # 修复：self.cfg 而非 self.config
            persist_p = Path(persist_path)
            if not persist_p.is_absolute():
                from config.config_load import get_project_root
                persist_p = Path(get_project_root()) / persist_path
            persist_p = persist_p.resolve()
            client = chromadb.PersistentClient(path=str(persist_p))
            # 兼容不同版本 chromadb：新版返回字符串，旧版返回带 .name 的集合对象
            raw_collections = client.list_collections()
            available_collections = []
            for c in raw_collections:
                if isinstance(c, str):
                    available_collections.append(c)
                elif hasattr(c, 'name'):
                    available_collections.append(c.name)
                else:
                    available_collections.append(str(c))
            print(f"【RAG流程-Step2检索】传入集合名: {collection_name}")
            print(f"【RAG流程-Step2检索】可用集合: {available_collections}")
            print(f"【RAG流程-Step2检索】集合是否存在: {collection_name in available_collections}")
        except Exception as e:
            print(f"【RAG流程-Step2检索】检查集合失败: {e}")

        try:
            retrieve_groups = self.retrieval_dispatcher.retrieve(
                query=user_query,
                use_pre_retrieval=decision["use_pre_retrieval"],
                pre_methods=decision["pre_methods"],
                collection_name=collection_name,
                init_docs=init_docs
            )
        except RuntimeError as err:
            if "COLLECTION_NOT_EXIST_NO_INIT_DOC" in str(err):
                print(f"【RAG流程-Step2检索】❌ 集合不存在且无 init_docs，返回兜底答案")
                final_answer = "暂时没办法回答你的问题"
                self.chat_saver.save_chat(user_query, final_answer, session_id, persist_time)
                return final_answer
            else:
                print(f"【RAG流程-Step2检索】RuntimeError: {str(err)}")
                retrieve_groups = []
        except Exception as e:
            import traceback
            print(f"【RAG流程-Step2检索】异常: {str(e)}")
            print(traceback.format_exc())
            retrieve_groups = []

        # 检索结果统计
        total_docs = sum(len(g) for g in retrieve_groups) if retrieve_groups else 0
        print(f"【RAG流程-Step2检索】检索完成: {len(retrieve_groups) if retrieve_groups else 0} 组, 共 {total_docs} 篇文档")
        if retrieve_groups:
            for i, group in enumerate(retrieve_groups):
                if group:
                    print(f"  组{i}: {len(group)} 篇, 首篇预览: {group[0].page_content[:150]}...")

        # 检索无结果兜底
        if len(retrieve_groups) == 0 or all(len(g) == 0 for g in retrieve_groups):
            print(f"【RAG流程-Step2检索】❌ 检索无结果，返回兜底答案")
            final_answer = "暂时没办法回答你的问题"
            self.chat_saver.save_chat(user_query, final_answer, session_id, persist_time)
            return final_answer

        # Step3：后置优化
        print(f"【RAG流程-Step3后置】开始后置优化，compress_type={custom_compress_type}, enable_rrf={enable_rrf}, enable_model_rerank={enable_model_rerank}")
        final_context = self.post_processor.process_context(
            query=user_query,
            retrieve_groups=retrieve_groups,
            custom_compress_type=custom_compress_type,
            enable_rrf=enable_rrf,
            enable_model_rerank=enable_model_rerank
        )
        print(f"【RAG流程-Step3后置】final_context 长度: {len(final_context)} 字符")
        print(f"【RAG流程-Step3后置】final_context 内容预览(前500字):\n{final_context[:500]}")

        if not final_context.strip():
            print(f"【RAG流程-Step3后置】❌ final_context 为空，返回兜底答案")
            final_answer = "暂时没办法回答你的问题"
            self.chat_saver.save_chat(user_query, final_answer, session_id, persist_time)
            return final_answer

        # Step4：生成最终回答
        try:
            prompt_data = self._build_generate_prompt(retrieve_groups, user_query, final_context)
            # 调试信息：打印 prompt 类型和长度，便于定位上下文超长等问题
            if isinstance(prompt_data, str):
                print(f"【RAG生成】纯文本 prompt，长度: {len(prompt_data)} 字符")
                llm_resp = self.generate_llm_text.invoke(prompt_data)
            elif self.multimodal_llm is not None:
                print(f"【RAG生成】多模态 messages，调用多模态LLM")
                llm_resp = self.multimodal_llm.invoke(prompt_data)
            else:
                # 多模态模型不可用，降级为纯文本：提取messages中的text部分
                fallback_text = ""
                if isinstance(prompt_data, list) and len(prompt_data) > 0:
                    content_arr = prompt_data[0].get("content", [])
                    for part in content_arr:
                        if isinstance(part, dict) and part.get("type") == "text":
                            fallback_text = part.get("text", "")
                            break
                if not fallback_text:
                    fallback_text = str(prompt_data)
                print("【RAG提示】多模态模型不可用，降级为纯文本回答")
                llm_resp = self.generate_llm_text.invoke(fallback_text)
            # 防护：LLM 可能返回 None content（空响应/工具调用），避免 .strip() 崩溃
            raw_content = llm_resp.content if llm_resp is not None else None
            final_answer = (raw_content or "").strip()
            if not final_answer:
                print(f"【RAG生成警告】LLM 返回空内容，响应类型: {type(llm_resp).__name__}，原始响应: {str(llm_resp)[:300]}")
                final_answer = f"检索资料：\n{final_context}\n\n模型返回内容为空，请重新提问。"
        except Exception as e:
            import traceback
            print(f"【生成LLM调用异常】{str(e)}")
            print(f"【生成LLM调用异常-堆栈】\n{traceback.format_exc()}")
            print(f"【生成LLM调用异常-上下文】final_context 长度: {len(final_context)} 字符，user_query: {user_query[:50]}")
            final_answer = f"检索资料：\n{final_context}\n\n模型生成失败，请重新提问。"

        # Step5：持久化对话
        self.chat_saver.save_chat(user_query, final_answer, session_id, persist_time)

        return final_answer

    # ===================== 对外顶层统一入口 =====================
    def run_rag_task(
            self,
            vector_text_list: Optional[List[str]] = None,
            collection_name: Optional[str] = None,
            session_id: Optional[str] = None,
            user_question: str = "",
            persist_time: Optional[int] = None,
            custom_compress_type: Optional[str] = None,
            user_pre_methods: Optional[List[str]] = None,
            enable_rrf: bool = True,
            enable_model_rerank: bool = True
    ) -> str:
        if not user_question.strip():
            return "请输入有效的提问内容"

        # 第二层参数校验与默认值覆盖
        use_collection = self.default_collection_name
        if collection_name and collection_name.strip():
            use_collection = collection_name.strip()

        use_persist_time = self.default_persist_time
        if persist_time is not None:
            use_persist_time = persist_time

        # 用户选择的预检索方法校验：过滤空值和非法值
        use_user_pre_methods = None
        if user_pre_methods and isinstance(user_pre_methods, list):
            use_user_pre_methods = [m.strip() for m in user_pre_methods if isinstance(m, str) and m.strip()]
            if not use_user_pre_methods:
                use_user_pre_methods = None

        # 文本转换：非空才转为Document用于初始化
        init_docs = None
        if vector_text_list and isinstance(vector_text_list, list) and len(vector_text_list) > 0:
            init_docs = [Document(page_content=t) for t in vector_text_list if isinstance(t, str) and t.strip()]

        return self.run_full_auto_rag(
            user_query=user_question.strip(),
            collection_name=use_collection,
            init_docs=init_docs,
            custom_compress_type=custom_compress_type,
            session_id=session_id,
            persist_time=use_persist_time,
            user_pre_methods=use_user_pre_methods,
            enable_rrf=enable_rrf,
            enable_model_rerank=enable_model_rerank
        )
