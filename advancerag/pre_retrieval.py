import uuid
import chromadb
from typing import Optional, List, Set, Dict, Any

try:
    from langchain.retrievers import MultiVectorRetriever, EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import MultiVectorRetriever, EnsembleRetriever, ParentDocumentRetriever, SelfQueryRetriever
else:
    from langchain_classic.retrievers import ParentDocumentRetriever, SelfQueryRetriever
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.stores import InMemoryByteStore, InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config_load import config_load
from myproject.models.models import ModelManager
from myproject.embedding.vectorydb_factory import VectorDbFactory
from myproject.conversation_save.save_conversation_history_redis import SaveChatRedis

class pre_retrieval:
    def __init__(self, yaml_path: str,
                 need_summary_text_list: Optional[List] = None,
                 parent_child_text_list: Optional[List[Document]] = None,
                 preview_question_text_list: Optional[List[Document]] = None,
                 self_query_text_list: Optional[List[Document]] = None,
                 need_enrich_text_list: Optional[List[Document]] = None,
                 multi_query_text_list: Optional[List[Document]] = None,
                 decomposition_text_list: Optional[List[Document]] = None,
                 hybrid_search_text_list: Optional[List[Document]] = None,
                 ):
        self.yaml_path = yaml_path
        self.config = config_load(self.yaml_path)

        # 1. 摘要检索
        self.need_summary_text_list = need_summary_text_list if need_summary_text_list is not None else []
        self.summary_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage("你是一个专业的摘要生成器"),
                HumanMessage("请根据以下内容生成摘要：{text}")
            ]
        )

        # 2. 父子文档检索
        self.parent_child_text_list = parent_child_text_list if parent_child_text_list is not None else []

        # 3. 预览问题检索
        self.preview_question_text_list = preview_question_text_list if preview_question_text_list is not None else []
        self.preview_question_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("你是一个专业的预览问题生成器"),
                HumanMessagePromptTemplate.from_template(
                    """请根据以下内容生成{number}个预览问题：{text}
                    严格按照下面的格式输出（提示：问题末尾不要添加句号，所有问题一行回答，用逗号隔开）：
                    问题1，问题2，问题3......

                    格式参考示例：
                    用户输入文本：deepseek好用吗？
                    你的输出：deepseek有什么功能，deepseek的性能如何，deepseek的费用是多少
                    """
                )
            ]
        )

        # 4. 自查询检索
        self.self_query_text_list = self_query_text_list if self_query_text_list is not None else []
        self.self_query_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "你是专业代码生成工具，严格遵守输入输出规范，仅输出目标Python代码，无任何多余文字、注释、json包裹。禁止使用markdown代码块标记（即禁止输出```），直接以metadata_field_info开头，以右括号]结尾。"),
            HumanMessagePromptTemplate.from_template("""
任务规则：
输入一段Document对象代码，固定输出完整metadata_field_info Python列表代码，所有name、description、type文字一字不可修改。

标准示例：
输入Document样例：
Document(
    page_content="作者A团队开发出基于人工智能的自动驾驶决策系统，在复杂路况下的响应速度提升300%",
    metadata={{"year": 2024, "rating": 9.2, "genre": "AI", "author": "A"}},
)
你必须原样输出（仅此内容，不加任何符号包裹，不要输出```python或```标记）：
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="文章的技术领域，选项:['AI '，'区块链'，'云'，'大数据']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="文章的出版年份",
        type="integer",
    ),
    AttributeInfo(
        name="author",
        description="署名文章的作者姓名",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="技术价值评估得分（1-10分）",
        type="float"
    )
]

硬性约束：
1. 只输出上面固定的metadata_field_info完整代码，不要解释、不要markdown、不要json、不要```代码块标记；
2. 无论输入任何Document，输出内容完全固定，禁止修改任何描述文本；
3. 不输出除代码以外的任何字符；
4. 输出必须以metadata_field_info开头，以]结尾，中间不允许出现```符号。

用户输入Document：
{text}
""")
        ])

        # 5. 内容增强检索（默认模板，支持用户自定义字段）
        self.need_enrich_text_list = need_enrich_text_list if need_enrich_text_list is not None else []
        # 增强字段完整性校验模板，完全对齐你给的槽位补全逻辑
        self.enrich_check_prompt = ChatPromptTemplate.from_messages([
            SystemMessage("你是文档信息补全助手，根据原始文档和补充信息，判断是否覆盖了所有要求的增强字段。只输出JSON，不要多余内容，禁止markdown代码块标记，布尔值用小写true或小写false。"),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("""
### 原始文档内容
{doc_content}

### 必须覆盖的增强字段
{enrich_fields}

### 输出要求
如果信息已覆盖所有字段，isComplete设为true，content返回整理完整的增强文档；
如果有字段缺失，isComplete设为false，content返回友好的引导语，明确告知需要补充哪些字段的信息。

严格按以下JSON格式输出，不要加任何多余文字、注释、markdown标记（布尔值必须小写true或false）：
{{
    "isComplete": true,
    "content": "完整增强文档内容"
}}

或字段缺失时：
{{
    "isComplete": false,
    "content": "友好的补充引导语，明确告知需要补充哪些字段"
}}
""")
        ])

        # 6. 多查询扩展检索
        self.multi_query_text_list = multi_query_text_list if multi_query_text_list is not None else []
        self.multi_query_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage("你是一个专业的问题分析助手"),
                HumanMessage("""基于用户的问题，生成{number}个同义表述问题，所有问题放在一行，用中文逗号分隔，不要序号、不要解释、不要多余内容，结尾不要添加句号。
                用户问题：{question}""")
            ]
        )

        # 7. 问题分解检索
        self.decomposition_text_list = decomposition_text_list if decomposition_text_list is not None else []
        self.decomposition_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage("你是专业的问题分解助手，将用户的复杂问题拆解为{number}个可独立检索的子问题，子问题需完整覆盖原问题核心诉求，所有子问题放在一行，用中文逗号分隔，不要序号、不要解释、不要多余内容。"),
                HumanMessage("用户问题：{question}")
            ]
        )

        # 8. 混合检索
        self.hybrid_search_text_list = hybrid_search_text_list if hybrid_search_text_list is not None else []

    def _collection_exists(self, collection_name: str) -> bool:
        """
        内部工具方法：判断Chroma中指定集合是否存在，避免重复入库
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

    def summary(self, need_summary_text_list: Optional[List[Document]] = None,
                collection_name: str = "default_collection"):
        """预检索摘要"""
        if need_summary_text_list is None:
            need_summary_text_list = self.need_summary_text_list

        chat_client = ModelManager(self.yaml_path).init_chat_model()
        out_parser = StrOutputParser()
        chain = self.summary_prompt | chat_client | out_parser
        vector_store = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)

        # 集合已存在则直接返回检索器，避免重复入库
        if self._collection_exists(collection_name):
            return MultiVectorRetriever(
                vectorstore=vector_store,
                byte_store=SaveChatRedis.get_langchain_redis_store(self.config) if self.config["save_method"]["if_save"] else InMemoryByteStore(),
                id_key="connection_id",
            )

        save_type = self.config["save_method"]["if_save"]
        if save_type:
            store = SaveChatRedis.get_langchain_redis_store(self.config)
        else:
            store = InMemoryByteStore()

        batch_inputs = [
            {"text": doc.page_content if isinstance(doc, Document) else doc}
            for doc in need_summary_text_list
        ]
        summary_text_list = chain.batch(batch_inputs)
        uuid_list = [str(uuid.uuid4()) for _ in range(len(need_summary_text_list))]

        document_summary_list = [
            Document(page_content=data, metadata={"connection_id": uuid_list[i]})
            for i, data in enumerate(summary_text_list)
        ]
        for i, data in enumerate(need_summary_text_list):
            data.metadata["connection_id"] = uuid_list[i]
        primary_data_list = need_summary_text_list

        retriever = MultiVectorRetriever(
            vectorstore=vector_store,
            byte_store=store,
            id_key="connection_id",
        )
        retriever.vectorstore.add_documents(document_summary_list)
        retriever.byte_store.mset(list(zip(uuid_list, primary_data_list)))
        return retriever

    def parent_child(self, parent_child_text_list: Optional[List[Document]] = None, collection_name: str = ""):
        """父子文档检索"""
        if parent_child_text_list is None:
            parent_child_text_list = self.parent_child_text_list
        if not collection_name:
            collection_name = "parent_child_collection"

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["parent_splitter"]["chunk_size"],
            chunk_overlap=self.config["parent_splitter"]["chunk_overlap"],
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["child_splitter"]["chunk_size"],
            chunk_overlap=self.config["child_splitter"]["chunk_overlap"],
        )
        chat_client = ModelManager(self.yaml_path).init_chat_model()

        save_type = self.config["save_method"]["if_save"]
        if save_type:
            store = SaveChatRedis.get_langchain_redis_store(self.config)
        else:
            store = InMemoryStore()

        vector_store = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)
        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=store,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter,
            search_kwargs={"k": self.config["parent_splitter_k"]["k"]},
        )
        # 集合已存在则跳过添加入库
        if not self._collection_exists(collection_name):
            retriever.add_documents(parent_child_text_list)
        return retriever

    def preview_question_index(self, preview_question_text_list: Optional[List[Document]] = None,
                               collection_name: str = "preview_question_collection"):
        """预览问题索引检索"""
        if preview_question_text_list is None:
            preview_question_text_list = self.preview_question_text_list

        vector_store = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)
        # 集合已存在则直接返回检索器
        if self._collection_exists(collection_name):
            return MultiVectorRetriever(
                vectorstore=vector_store,
                byte_store=SaveChatRedis.get_langchain_redis_store(self.config) if self.config["save_method"]["if_save"] else InMemoryByteStore(),
                id_key="connection_id",
            )

        batch_preview_question_text_list = [{"text": q.page_content} for q in preview_question_text_list]
        chat_client = ModelManager(self.yaml_path).init_chat_model()
        out_parser = StrOutputParser()
        partial_prompt = self.preview_question_prompt.partial(number=self.config["preview_question"]["number"])
        chain = partial_prompt | chat_client | out_parser

        preview_question_list = chain.batch(batch_preview_question_text_list)
        new_question_list = [question.split("，") for question in preview_question_list]

        save_type = self.config["save_method"]["if_save"]
        if save_type:
            store = SaveChatRedis.get_langchain_redis_store(self.config)
        else:
            store = InMemoryByteStore()

        retriever = MultiVectorRetriever(
            vectorstore=vector_store,
            byte_store=store,
            id_key="connection_id",
        )
        uuid_list = [str(uuid.uuid4()) for _ in range(len(preview_question_text_list))]
        question_document_list = []
        for i, question_group in enumerate(new_question_list):
            docs = [Document(page_content=q.strip(), metadata={"connection_id": uuid_list[i]}) for q in question_group
                    if q.strip()]
            question_document_list.extend(docs)

        retriever.vectorstore.add_documents(question_document_list)
        retriever.byte_store.mset(list(zip(uuid_list, preview_question_text_list)))
        return retriever

    def self_query(self, self_query_text_list: Optional[List[Document]] = None,
                   collection_name: str = "self_query_collection", document_content_description: str = ""):
        """自查询元数据检索"""
        if self_query_text_list is None:
            self_query_text_list = self.self_query_text_list
        if not self_query_text_list:
            raise ValueError("self_query_text_list 不能为空，请传入至少一条Document")

        chat_client = ModelManager(self.yaml_path).init_chat_model()
        str_parser = StrOutputParser()
        gen_chain = self.self_query_prompt | chat_client | str_parser

        try:
            raw_code = gen_chain.invoke({"text": self_query_text_list[0]})
            # 清理大模型可能输出的markdown代码块标记
            raw_code = raw_code.strip()
            if raw_code.startswith("```python"):
                raw_code = raw_code[len("```python"):]
            elif raw_code.startswith("```"):
                raw_code = raw_code[3:]
            if raw_code.endswith("```"):
                raw_code = raw_code[:-3]
            raw_code = raw_code.strip()

            local_vars: Dict[str, Any] = {}
            exec(raw_code, globals(), local_vars)
            metadata_field_info = local_vars.get("metadata_field_info")
            if metadata_field_info is None:
                raise ValueError("LLM输出代码中未找到metadata_field_info变量，请检查提示词或重试")
        except Exception as e:
            print(f"[RAG-self_query] 代码生成或执行失败：{str(e)}")
            raise ValueError(f"self_query元数据生成失败：{str(e)}，请检查LLM输出或重试")

        vectorstore = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)
        if not self._collection_exists(collection_name):
            vectorstore.add_documents(self_query_text_list)

        retriever = SelfQueryRetriever.from_llm(
            llm=chat_client,
            vectorstore=vectorstore,
            document_contents=document_content_description,
            metadata_field_info=metadata_field_info,
            verbose=False
        )
        return retriever

    def enrich(self, user_query: str, template_list: Optional[List[str]] = None, collection_name: str = "enrich_template_collection") -> Dict[str, Any]:
        """
        模板向量检索+多轮信息补全（本方法完全重写，其余代码未改动）
        :param user_query: 用户原始提问/需求
        :param template_list: 业务字段模板列表，如["起点，终点，出行时间，座位等级", "城市，入住日期，房型"]
        :param collection_name: 存储业务模板的向量集合名称
        :return: 补全完成的结构化数据字典
        """
        chat_client = ModelManager(self.yaml_path).init_chat_model()
        json_parser = JsonOutputParser()

        # ========== 步骤1：业务模板存入向量库，集合已存在则跳过 ==========
        vector_store = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)
        if not self._collection_exists(collection_name):
            template_docs = [Document(page_content=tmp) for tmp in template_list]
            vector_store.add_documents(template_docs)

        # ========== 步骤2：用户问题向量召回匹配度最高的模板 ==========
        base_retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        match_template = base_retriever.invoke(user_query)[0].page_content

        # ========== 步骤3：构建补全专用prompt + 带历史的执行链路 ==========
        fill_prompt = ChatPromptTemplate.from_messages([
            SystemMessage("""你是信息填充助手，严格遵守以下规则：
1. 读取【标准字段模板】里所有必填字段；
2. 对比用户提问和历史对话，判断字段是否全部补齐；
3. 存在缺失字段：isComplete设为false，content返回友好引导语，告知需要补充的信息；
4. 全部字段齐全：isComplete设为true，content返回完整结构化JSON数据；
只输出标准JSON，禁止任何多余文字、注释、解释、markdown代码块标记。

输出格式必须严格参照以下JSON结构（布尔值用小写true/false）：
{"isComplete": true, "content": "完整的结构化JSON数据"}
或
{"isComplete": false, "content": "友好的补充引导语"}"""),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        chat_history = ChatMessageHistory()
        fill_chain = fill_prompt | chat_client | json_parser
        with_history_chain = RunnableWithMessageHistory(
            fill_chain,
            lambda session_id: chat_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        # ========== 步骤4：首轮校验：拼接用户问题+模板规则 ==========
        first_input = f"""### 用户原始提问
{user_query}

### 匹配到的标准字段模板
{match_template}

请按规则判断信息是否完整，输出指定JSON格式。"""
        try:
            result = with_history_chain.invoke(
                input={"input": first_input},
                config={"configurable": {"session_id": "enrich_fill"}}
            )
        except Exception as e:
            print(f"[RAG-enrich] 首轮LLM调用或JSON解析失败：{str(e)}")
            return {"isComplete": False, "content": f"信息解析失败，请重试。错误：{str(e)}"}

        # 防御性检查：确保result是字典
        if not isinstance(result, dict):
            print(f"[RAG-enrich] LLM返回格式异常，期望dict实际{type(result).__name__}：{str(result)[:200]}")
            return {"isComplete": False, "content": "信息解析格式异常，请重试"}

        # ========== 步骤5：LLM 自动推断补全，循环直到字段完整或达到最大轮数 ==========
        max_rounds = 5  # 防止无限循环兜底
        round_count = 0
        while not result.get("isComplete", False) and round_count < max_rounds:
            round_count += 1
            try:
                # 让 LLM 根据已有上下文自动推断补充缺失字段，不再阻塞等待用户输入
                auto_fill_prompt = (
                    f"根据之前对话中你提出的补充引导语「{result.get('content', '请补充对应信息')}」，"
                    f"请结合【原始用户提问】和【标准字段模板】中能推断出的所有信息，"
                    f"自动补充缺失字段的合理推断值，直接输出完整JSON格式，禁止再要求补充。"
                )
                result = with_history_chain.invoke(
                    input={"input": auto_fill_prompt},
                    config={"configurable": {"session_id": "enrich_fill"}}
                )
                if not isinstance(result, dict):
                    print(f"[RAG-enrich] 第{round_count}轮自动推断LLM返回格式异常，继续下一轮")
                    continue
            except Exception:
                print(f"[RAG-enrich] 第{round_count}轮自动推断格式解析失败，继续下一轮")
                continue

        # ========== 步骤6：返回补全完成的最终结构化数据 ==========
        return result

    def enrich_index_build(self, enrich_text_list: List[Document], collection_name: str = "enrich_template_collection") -> None:
        """
        文档字段增强索引构建：从文档中提取业务字段模板并存储到向量库
        :param enrich_text_list: 文档列表，从中提取字段模板
        :param collection_name: 存储业务模板的向量集合名称
        """
        if not enrich_text_list:
            print("[RAG-enrich] enrich_index_build 收到空文档列表，跳过")
            return

        chat_client = ModelManager(self.yaml_path).init_chat_model()
        str_parser = StrOutputParser()

        extract_prompt = ChatPromptTemplate.from_messages([
            SystemMessage("你是一个专业的字段提取助手。请从以下文档内容中提取关键业务字段，生成标准字段模板。"),
            HumanMessagePromptTemplate.from_template("""
请从以下文档内容中提取关键业务字段，每行一个字段模板：

文档内容：
{document_content}

输出格式示例：
起点，终点，出行时间，座位等级
城市，入住日期，房型
姓名，身份证号，联系方式

请提取3-5个字段模板。""")
        ])

        extract_chain = extract_prompt | chat_client | str_parser

        all_templates = set()
        for doc in enrich_text_list:
            try:
                result = extract_chain.invoke({"document_content": doc.page_content[:2000]})
                for line in result.split("\n"):
                    line = line.strip()
                    if line and len(line) > 5:
                        all_templates.add(line)
            except Exception as e:
                print(f"[RAG-enrich] 从文档提取字段模板失败: {e}")
                continue

        if not all_templates:
            print("[RAG-enrich] 未提取到任何字段模板，使用默认模板")
            all_templates = {"姓名，联系方式，时间，地点"}

        vector_store = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)
        if not self._collection_exists(collection_name):
            template_docs = [Document(page_content=tmp) for tmp in all_templates]
            vector_store.add_documents(template_docs)
            print(f"[RAG-enrich] ✅ 字段模板索引构建完成，共 {len(template_docs)} 个模板")
        else:
            print(f"[RAG-enrich] ⚠️ 集合 {collection_name} 已存在，跳过索引构建")

    def multi_query(self, query: str,
                    multi_query_text_list: Optional[List[Document]] = None,
                    collection_name: str = "multi_query_collection") -> List[Document]:
        """多查询扩展检索：生成同义查询批量召回，合并去重返回结果"""
        if not query:
            raise ValueError("query 不能为空，请传入查询问题")

        # 集合已存在则直接复用，不存在则新建入库
        if not self._collection_exists(collection_name):
            if multi_query_text_list is None and not self.multi_query_text_list:
                raise ValueError("集合不存在时，必须传入 multi_query_text_list 构建索引")
            use_docs = multi_query_text_list if multi_query_text_list is not None else self.multi_query_text_list
            vectorstore = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)
            vectorstore.add_documents(use_docs)
        else:
            vectorstore = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)

        chat_client = ModelManager(self.yaml_path).init_chat_model()
        partial_prompt = self.multi_query_prompt.partial(number=self.config["rag"]["multi_query_count"])
        chain = partial_prompt | chat_client | StrOutputParser()

        # 生成多个同义查询
        generated_queries = [q.strip() for q in chain.invoke({"question": query}).split("，") if q.strip()]
        base_retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.config["vector_db"]["search_top_k"]}
        )

        # 批量检索并去重（修复Document不可哈希的报错，改用内容去重）
        result_map: Dict[str, Document] = {}
        for q in generated_queries:
            chunks = base_retriever.invoke(q)
            for doc in chunks:
                if doc.page_content not in result_map:
                    result_map[doc.page_content] = doc
        return list(result_map.values())

    def decomposition(self, query: str,
                      decomposition_text_list: Optional[List[Document]] = None,
                      collection_name: str = "decomposition_collection") -> List[Document]:
        """查询分解检索：拆分复杂问题为独立子问题，分别召回后合并去重"""
        if not query:
            raise ValueError("query 不能为空，请传入查询问题")

        # 集合已存在则直接复用，不存在则新建入库
        if not self._collection_exists(collection_name):
            if decomposition_text_list is None and not self.decomposition_text_list:
                raise ValueError("集合不存在时，必须传入 decomposition_text_list 构建索引")
            use_docs = decomposition_text_list if decomposition_text_list is not None else self.decomposition_text_list
            vectorstore = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)
            vectorstore.add_documents(use_docs)
        else:
            vectorstore = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)

        chat_client = ModelManager(self.yaml_path).init_chat_model()
        partial_prompt = self.decomposition_prompt.partial(number=self.config["rag"]["decomposition_count"])
        chain = partial_prompt | chat_client | StrOutputParser()

        # 拆解为多个独立子问题
        sub_queries = [q.strip() for q in chain.invoke({"question": query}).split("，") if q.strip()]
        base_retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.config["vector_db"]["search_top_k"]}
        )

        # 每个子问题分别检索，合并去重（修复Document不可哈希的报错）
        result_map: Dict[str, Document] = {}
        for sub_q in sub_queries:
            chunks = base_retriever.invoke(sub_q)
            for doc in chunks:
                if doc.page_content not in result_map:
                    result_map[doc.page_content] = doc
        return list(result_map.values())

    def hybrid_search(self, hybrid_search_text_list: Optional[List[Document]] = None,
                      collection_name: str = "hybrid_search_collection"):
        """混合检索：BM25关键词检索 + 向量语义检索加权融合"""
        vectorstore = VectorDbFactory(self.yaml_path).init_vector_store(collection_name=collection_name)
        vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.config["vector_db"]["search_top_k"]}
        )

        # 如果集合已存在，从向量库获取已存储的文档来构建 BM25
        if self._collection_exists(collection_name):
            try:
                all_docs = vectorstore._collection.get(include=["documents", "metadatas"])
                stored_docs = []
                for i, content in enumerate(all_docs["documents"]):
                    metadata = all_docs["metadatas"][i] if i < len(all_docs["metadatas"]) else {}
                    stored_docs.append(Document(page_content=content, metadata=metadata))
                bm25_retriever = BM25Retriever.from_documents(stored_docs)
                bm25_retriever.k = self.config["vector_db"]["search_top_k"]
                retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=[
                        self.config["vector_db"]["rrf_weight_bm25"],
                        self.config["vector_db"]["rrf_weight_vector"]
                    ]
                )
                return retriever
            except Exception as e:
                print(f"【RAG混合检索】从向量库重建BM25失败，降级为纯向量检索: {e}")
                return vector_retriever
        
        # 集合不存在，需要传入文档来构建索引
        if hybrid_search_text_list is None:
            hybrid_search_text_list = self.hybrid_search_text_list
        if not hybrid_search_text_list:
            raise ValueError("hybrid_search_text_list 不能为空，请传入至少一条Document")

        vectorstore.add_documents(hybrid_search_text_list)
        bm25_retriever = BM25Retriever.from_documents(hybrid_search_text_list)
        bm25_retriever.k = self.config["vector_db"]["search_top_k"]

        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[
                self.config["vector_db"]["rrf_weight_bm25"],
                self.config["vector_db"]["rrf_weight_vector"]
            ]
        )
        return retriever
