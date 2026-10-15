import os
import re
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional, Callable
from langchain_core.documents import Document
from pypinyin import lazy_pinyin

from config.config_load import config_load, get_project_root
from myproject.embedding.vectorydb_factory import VectorDbFactory
from myproject.datawash.muti_catolog_file_wash import data_wash
from myproject.datawash.data_wash import TextTanckle

# 加载 .env 环境变量（修复：明确指定项目根目录路径，与 models.py 保持一致）
load_dotenv(Path(get_project_root()) / ".env")

# 集合名称合法字符正则（仅允许小写字母、数字、下划线、连字符，长度3-63）
_COLLECTION_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,61}[a-z0-9]$")


class EmbeddingVectorManager:
    """嵌入向量管理器：统一管理文档向量化入库、集合名称映射"""

    # 开关：是否启用中文自动转拼音集合名
    ENABLE_AUTO_PINYIN_CONVERT = True
    # 预留：默认兜底集合
    DEFAULT_COLLECTION = "default_rag_collection"
    
    # 支持的入库类型及其对应的检索方法
    UPLOAD_TYPES = {
        "normal": {"name": "普通向量入库", "support_methods": ["normal"]},
        "parent_child": {"name": "父子文档入库", "support_methods": ["parent_child", "normal"]},
        "preview_question": {"name": "预览问题索引", "support_methods": ["preview_question", "normal"]},
        "summary": {"name": "摘要多向量入库", "support_methods": ["summary", "normal"]},
        "enrich": {"name": "文档字段增强", "support_methods": ["enrich", "normal"]},
        "hybrid_search": {"name": "混合检索入库", "support_methods": ["hybrid_search", "normal"]},
    }

    def __init__(
        self,
        yaml_path: str,
        dashscope_api_key: Optional[str] = None,
        kb_meta_service=None,  # 外部注入 MySQL 映射操作类
        model_manager=None      # 外部注入 ModelManager 实例
    ):
        self.yaml_path = yaml_path
        self.config = config_load(yaml_path)
        self.vector_factory = VectorDbFactory(yaml_path)
        self.kb_meta_service = kb_meta_service
        self.model_manager = model_manager
        # 密钥优先级：传入参数 > .env 环境变量
        self.dashscope_api_key = dashscope_api_key or os.getenv("ALI_API_KEY")

    def _normalize_collection_name(self, input_name: str) -> str:
        """
        规范化向量集合名称【动态拼音转换版本】
        流程：
        1. 去除首尾空白，判空
        2. 优先查询 MySQL 映射表：中文展示名是否存在已有映射
        3. 存在 → 返回数据库存储的底层英文名
        4. 不存在 → 中文转拼音 + 清洗非法字符生成合法集合名
        5. 新增映射关系写入 MySQL，下次直接复用
        """
        if not isinstance(input_name, str):
            print(f"【RAG警告】知识库名称类型异常: {type(input_name)}，使用默认集合")
            return self.DEFAULT_COLLECTION

        input_name = input_name.strip()
        if not input_name:
            print(f"【RAG警告】知识库名称为空，使用默认集合 {self.DEFAULT_COLLECTION}")
            return self.DEFAULT_COLLECTION

        # --------------------------
        # 第一步：查询 MySQL 是否已有这条知识库映射
        # --------------------------
        if self.kb_meta_service is not None:
            try:
                exist_coll_name = self.kb_meta_service.get_kb_by_display_name(input_name)
                if exist_coll_name:
                    print(f"【RAG映射】数据库命中: {input_name} → {exist_coll_name}")
                    return exist_coll_name
            except Exception as e:
                print(f"【RAG警告】查询知识库映射失败: {e}，继续拼音转换")

        # --------------------------
        # 第二步：动态拼音转换
        # --------------------------
        valid_pattern = re.compile(r"^[a-z0-9][a-z0-9_-]*[a-z0-9]$")
        # 如果输入本身已经是合法英文名，直接使用
        if valid_pattern.fullmatch(input_name.lower()):
            real_name = input_name.lower()
        elif self.ENABLE_AUTO_PINYIN_CONVERT:
            try:
                # 中文转拼音
                pinyin_list = lazy_pinyin(input_name)
                raw_str = "_".join(pinyin_list)
                # 全部小写 + 非法字符替换为下划线
                raw_str = raw_str.lower()
                real_name = re.sub(r"[^a-z0-9_]", "_", raw_str)
                # 合并连续多个下划线
                real_name = re.sub(r"_+", "_", real_name)
                # 去除首尾下划线
                real_name = real_name.strip("_")
                # 长度校验：截断过长名称
                if len(real_name) > 63:
                    real_name = real_name[:63].rstrip("_")
                # 防止清洗后全是下划线或为空
                if not real_name or len(real_name) < 3:
                    real_name = self.DEFAULT_COLLECTION
                print(f"【RAG映射】拼音转换: {input_name} → {real_name}")
            except Exception as e:
                print(f"【RAG警告】拼音转换失败: {e}，使用默认集合")
                return self.DEFAULT_COLLECTION
        else:
            # 关闭拼音转换且无匹配项，返回默认集合
            print(f"【RAG警告】知识库名称[{input_name}]无映射，使用默认集合 {self.DEFAULT_COLLECTION}")
            return self.DEFAULT_COLLECTION

        # --------------------------
        # 第三步：新增映射写入数据库（不存在时持久化）
        # --------------------------
        if self.kb_meta_service is not None:
            try:
                self.kb_meta_service.add_kb_mapping(display_name=input_name, collection_name=real_name)
            except Exception as e:
                print(f"【RAG警告】写入知识库映射失败: {e}")

        return real_name

    def _validate_collection_name(self, name: str) -> bool:
        """校验集合名称是否合法（仅允许小写字母/数字/下划线/连字符，3-63字符）"""
        if not isinstance(name, str) or not name:
            return False
        return bool(_COLLECTION_NAME_PATTERN.fullmatch(name))

    def text_to_embedding_ollama(
        self,
        documents: List[Document],
        collection_name: str = DEFAULT_COLLECTION
    ):
        """本地 Ollama Embedding，文档向量入库"""
        if not documents:
            print("【RAG警告】待入库文档为空，跳过向量化")
            return None

        try:
            embed_client = self.model_manager.init_embedding_model()
            vector_store = self.vector_factory.init_vector_store(
                collection_name=collection_name,
                model=embed_client,
            )
            vector_store.add_documents(documents)
            print(f"【RAG向量】Ollama 本地入库成功: {collection_name}，文档数: {len(documents)}")
            return vector_store
        except Exception as e:
            print(f"【RAG错误】Ollama 向量入库失败 [{collection_name}]: {e}")
            raise

    def text_to_embedding_dashscope(
        self,
        documents: List[Document],
        collection_name: str = DEFAULT_COLLECTION
    ):
        """阿里云 DashScope 云端 Embedding，文档向量入库"""
        if not documents:
            print("【RAG警告】待入库文档为空，跳过向量化")
            return None

        try:
            embed_client = self.model_manager.init_embedding_model()
            vector_store = self.vector_factory.init_vector_store(
                collection_name=collection_name,
                model=embed_client,
            )
            vector_store.add_documents(documents)
            print(f"【RAG向量】DashScope 云端入库成功: {collection_name}，文档数: {len(documents)}")
            return vector_store
        except Exception as e:
            print(f"【RAG错误】DashScope 向量入库失败 [{collection_name}]: {e}")
            raise

    def embedding_manager_choice(
        self,
        documents: List[Document],
        collection_name: str = DEFAULT_COLLECTION
    ):
        """
        统一入口：自动读取配置，选择本地/云端 Embedding 执行入库
        """
        if not documents:
            print("【RAG警告】embedding_manager_choice 收到空文档列表，跳过")
            return None

        # 边界防护：清洗 metadata 中的空列表，Chroma 不允许 metadata 列表值为空
        for doc in documents:
            if doc.metadata:
                for k, v in list(doc.metadata.items()):
                    if isinstance(v, list) and len(v) == 0:
                        doc.metadata[k] = ""  # 空字符串代替空列表，下游 .get() 默认值兼容

        if self.config["embedding"]["use_local"]:
            return self.text_to_embedding_ollama(documents, collection_name)
        else:
            return self.text_to_embedding_dashscope(documents, collection_name)

    def simple_text_list_insert(
        self,
        text_list: List[str],
        collection_name: str = DEFAULT_COLLECTION,
        upload_type: str = "normal"
    ):
        """
        便捷接口：直接传入原始字符串列表，自动转为 Document 入库
        支持传入【中文知识库名称】内部自动规范化
        :param upload_type: 入库类型，可选：normal, parent_child, preview_question, summary, enrich, hybrid_search
        """
        if not text_list or not isinstance(text_list, list):
            print("【RAG警告】simple_text_list_insert 收到空列表，跳过")
            return None

        real_coll_name = self._normalize_collection_name(collection_name)
        docs = [Document(page_content=txt.strip()) for txt in text_list if txt and txt.strip()]
        if not docs:
            print("【RAG警告】所有文本为空，跳过入库")
            return None
        
        # 修复：parent_child/preview_question/summary/enrich 需要跳过基础向量入库，
        # 否则集合已存在会导致预检索索引构建被跳过
        need_skip_base = upload_type in ("parent_child", "preview_question", "summary", "enrich")
        result = None
        if not need_skip_base:
            result = self.embedding_manager_choice(docs, real_coll_name)
        
        # 如果不是普通入库，执行对应预检索索引构建
        if upload_type != "normal" and upload_type in self.UPLOAD_TYPES:
            try:
                print(f"【RAG入库-进阶】开始构建 {self.UPLOAD_TYPES[upload_type]['name']} 索引")
                from myproject.advancerag.pre_retrieval import pre_retrieval
                pre_retriever = pre_retrieval(self.yaml_path)
                
                if upload_type == "parent_child":
                    pre_retriever.parent_child(parent_child_text_list=docs, collection_name=real_coll_name)
                elif upload_type == "preview_question":
                    pre_retriever.preview_question_index(preview_question_text_list=docs, collection_name=real_coll_name)
                elif upload_type == "summary":
                    pre_retriever.summary(need_summary_text_list=docs, collection_name=real_coll_name)
                elif upload_type == "enrich":
                    pre_retriever.enrich_index_build(enrich_text_list=docs, collection_name=real_coll_name)
                elif upload_type == "hybrid_search":
                    pre_retriever.hybrid_search(hybrid_search_text_list=docs, collection_name=real_coll_name)
                
                print(f"【RAG入库-进阶】✅ {self.UPLOAD_TYPES[upload_type]['name']} 索引构建完成")
            except Exception as e:
                print(f"【RAG入库-进阶】⚠️ {self.UPLOAD_TYPES[upload_type]['name']} 索引构建失败（不影响基础向量入库）: {e}")
        
        return result
    
    def get_support_methods(self, upload_type: str) -> str:
        """根据入库类型获取支持的检索方法列表（逗号分隔）"""
        if upload_type in self.UPLOAD_TYPES:
            return ",".join(self.UPLOAD_TYPES[upload_type]["support_methods"])
        return "normal"

    # ===== 文件直接向量化入库 =====
    def file_to_embedding(
            self,
            file_path: str,
            collection_name: str = DEFAULT_COLLECTION,
            split_method: Optional[Callable[[str, bool, str], Optional[List[Document]]]] = None,
            upload_type: str = "normal"
    ):
        """
        【顶层接口】支持传入文件路径 / 文件夹路径
        自动执行：文件遍历 → 清洗解析 → 文本分片 → 循环嵌套列表执行向量化
        :param file_path: 文件/文件夹路径
        :param collection_name: 支持【中文知识库名称 / 标准英文集合名】
        :param split_method: 【可选】自定义文本分片函数；不传则内置默认分片
        :param upload_type: 入库类型，可选：normal, parent_child, preview_question, summary, enrich, hybrid_search
        """
        # 入参校验
        if not file_path or not isinstance(file_path, str):
            print("【RAG错误】file_to_embedding 收到无效文件路径")
            return None

        try:
            real_collection = self._normalize_collection_name(collection_name)
        except Exception as e:
            print(f"【RAG错误】集合名称规范化失败 [{collection_name}]: {e}")
            return None

        target_path = Path(file_path).resolve()
        valid_file_paths: List[str] = []
        allow_suffix = {".txt", ".md", ".pdf", ".docx"}

        if not target_path.exists():
            print(f"【RAG警告】路径不存在 {file_path}，跳过向量化")
            return None

        try:
            if target_path.is_file():
                if target_path.suffix.lower() in allow_suffix:
                    valid_file_paths.append(str(target_path))
                else:
                    print(f"【RAG警告】不支持的文件类型: {target_path.suffix}，允许: {allow_suffix}")
                    return None
            elif target_path.is_dir():
                for child in target_path.iterdir():
                    if child.is_file() and child.suffix.lower() in allow_suffix:
                        valid_file_paths.append(str(child))
            else:
                print(f"【RAG警告】路径类型未知: {file_path}")
                return None
        except Exception as e:
            print(f"【RAG错误】文件遍历失败: {e}")
            return None

        if not valid_file_paths:
            print("【RAG警告】未找到任何支持解析的有效文件")
            return None

        # 1、批量解析文件（增加异常捕获，单文件失败不影响整体）
        try:
            file_tackle = data_wash(
                file_path_list=valid_file_paths,
                image_save_path=self.config["path"]["output_img"],
                markdown_save_dir=self.config["path"]["output_md"],
                outdir=self.config["path"]["mineru_output_dir"]
            )
            wash_result_dict = file_tackle.multi_data_wash()
        except Exception as e:
            print(f"【RAG错误】批量文件解析异常: {e}")
            return None

        if not wash_result_dict:
            print("【RAG警告】文件解析后无有效文本内容")
            return None

        print(f"【RAG入库-Step1解析】成功解析 {len(wash_result_dict)} 个文件")
        for file_key, info in wash_result_dict.items():
            md_text = info.get("markdown_text", "")
            print(f"  文件[{file_key}]: 文本长度 {len(md_text)} 字符, 图片 {len(info.get('image_paths', []))} 张")

        # 2、文本与源路径整理
        raw_text_list: List[str] = []
        source_path_list: List[str] = []
        for _, file_info in wash_result_dict.items():
            raw_text_list.append(file_info["markdown_text"])
            source_path_list.append(file_info["original_file_path"])

        # 3、文本分片，产出文档列表 List[Document]
        try:
            document_tackle = TextTanckle()
            document_list = document_tackle.text_batch_tanckle(
                method=split_method,
                text_list=raw_text_list,
                text_source_list=source_path_list,
                is_markdown=True
            )
        except Exception as e:
            print(f"【RAG错误】文本分片异常: {e}")
            return None

        if not document_list:
            print("【RAG警告】文本分片后无可用文档，跳过入库")
            return None

        # 过滤空内容文档（page_content 为空字符串的文档会导致 Chroma 静默跳过）
        valid_docs = [d for d in document_list if d.page_content and d.page_content.strip()]
        skipped_count = len(document_list) - len(valid_docs)
        if skipped_count > 0:
            print(f"【RAG入库-Step2分片】过滤 {skipped_count} 篇空内容文档")
        document_list = valid_docs

        print(f"【RAG入库-Step2分片】分片完成: 共 {len(document_list)} 篇有效文档")

        # 4、执行向量入库（异常不中断，打印日志）
        # 修复：parent_child/preview_question/summary/enrich 需要跳过基础向量入库，
        # 否则集合已存在会导致预检索索引构建被跳过
        need_skip_base = upload_type in ("parent_child", "preview_question", "summary", "enrich")
        success_count = 0
        fail_count = 0
        if not need_skip_base:
            try:
                self.embedding_manager_choice(document_list, real_collection)
                success_count = 1
                # 写入验证：查询集合中实际文档数，确认真的写入成功
                try:
                    from langchain_chroma import Chroma
                    from config.config_load import get_project_root
                    persist_path = self.config["vector_db"]["persist_path"]
                    persist_p = Path(persist_path)
                    if not persist_p.is_absolute():
                        persist_p = Path(get_project_root()) / persist_path
                    persist_p = persist_p.resolve()
                    embed_model = self.model_manager.init_embedding_model()
                    verify_store = Chroma(
                        collection_name=real_collection,
                        embedding_function=embed_model,
                        persist_directory=str(persist_p),
                    )
                    actual_count = verify_store._collection.count()
                    print(f"【RAG入库-Step3验证】✅ 集合[{real_collection}] 实际写入 {actual_count} 条向量记录")
                    if actual_count == 0:
                        print(f"【RAG入库-Step3验证】⚠️ 警告：写入后集合仍为空，可能 embedding 模型未正确生成向量")
                except Exception as verify_err:
                    print(f"【RAG入库-Step3验证】验证失败（不影响入库结果）: {verify_err}")
            except Exception as e:
                fail_count = 1
                print(f"【RAG错误】文档入库失败: {e}")

        print(f"【RAG向量】基础入库: {real_collection}，成功 {success_count} 批，失败 {fail_count} 批")
        
        # 如果不是普通入库，执行对应预检索索引构建
        if upload_type != "normal" and upload_type in self.UPLOAD_TYPES:
            try:
                print(f"【RAG入库-进阶】开始构建 {self.UPLOAD_TYPES[upload_type]['name']} 索引")
                from myproject.advancerag.pre_retrieval import pre_retrieval
                pre_retriever = pre_retrieval(self.yaml_path)
                
                if upload_type == "parent_child":
                    pre_retriever.parent_child(parent_child_text_list=document_list, collection_name=real_collection)
                elif upload_type == "preview_question":
                    pre_retriever.preview_question_index(preview_question_text_list=document_list, collection_name=real_collection)
                elif upload_type == "summary":
                    pre_retriever.summary(need_summary_text_list=document_list, collection_name=real_collection)
                elif upload_type == "enrich":
                    pre_retriever.enrich_index_build(enrich_text_list=document_list, collection_name=real_collection)
                elif upload_type == "hybrid_search":
                    pre_retriever.hybrid_search(hybrid_search_text_list=document_list, collection_name=real_collection)
                
                print(f"【RAG入库-进阶】✅ {self.UPLOAD_TYPES[upload_type]['name']} 索引构建完成")
            except Exception as e:
                print(f"【RAG入库-进阶】⚠️ {self.UPLOAD_TYPES[upload_type]['name']} 索引构建失败（不影响基础向量入库）: {e}")
        
        return True if success_count > 0 else None