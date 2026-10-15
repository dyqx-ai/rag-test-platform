import os
import sys
import uuid
import logging
from typing import Optional, List

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ====================== 基础路径配置（基于 __file__ 绝对路径，杜绝运行目录问题）======================
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # myproject/
_PARENT_DIR = os.path.dirname(_BASE_DIR)  # myproject 的父目录（project1/）
# 将父目录加入 sys.path，确保 `from myproject.xxx` 在任何运行目录下都能正确导入
sys.path.insert(0, _PARENT_DIR)
sys.path.insert(0, _BASE_DIR)

CONFIG_PATH = os.path.join(_BASE_DIR, "config", "config.yaml")
TEMPLATE_FOLDER = os.path.join(_BASE_DIR, "html")
STATIC_FOLDER = os.path.join(TEMPLATE_FOLDER, "static")
UPLOAD_FOLDER = os.path.join(_BASE_DIR, "upload_files")

# ====================== 日志配置（统一标识 [RAGxxx]，方便排查）======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("RAG_System")

# ====================== 文件上传白名单 ======================
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".xml"}
MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024  # 100MB

# ====================== 目录自动创建 ======================
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
logger.info(f"[RAG启动] 上传目录: {UPLOAD_FOLDER}")
logger.info(f"[RAG启动] 模板目录: {TEMPLATE_FOLDER}")

# ====================== 启动前自检函数 ======================
def startup_self_check():
    """FastAPI 启动阶段自检：配置文件、目录、Redis、MySQL 连通性"""
    logger.info("=" * 50)
    logger.info("[RAG启动] 开始启动前自检...")

    # 1. 检查配置文件
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"[RAG启动] 配置文件不存在: {CONFIG_PATH}")
        raise FileNotFoundError(f"配置文件不存在: {CONFIG_PATH}，请确认 config/config.yaml 已创建")
    logger.info(f"[RAG启动] ✅ 配置文件存在: {CONFIG_PATH}")

    # 2. 检查目录自动创建
    for d in [UPLOAD_FOLDER, TEMPLATE_FOLDER, STATIC_FOLDER]:
        os.makedirs(d, exist_ok=True)
        logger.info(f"[RAG启动] ✅ 目录就绪: {d}")

    # 3. 测试 Redis 连通性
    try:
        from myproject.conversation_save.save_conversation_history_redis import SaveChatRedis
        if SaveChatRedis.test_connection():
            logger.info("[RAG启动] ✅ Redis 连接正常")
        else:
            logger.error("[RAG启动] ❌ Redis 连接失败，请检查 Redis 服务是否启动")
            raise RuntimeError("Redis 连接失败，请检查 config.yaml 中 Redis 配置并确保 Redis 服务已启动")
    except Exception as e:
        logger.error(f"[RAG启动] ❌ Redis 自检异常: {e}")
        raise

    # 4. 测试 MySQL 连通性
    try:
        from myproject.mysql_save_data.save_data_to_mysql import MysqlDataManager
        if MysqlDataManager.test_connection(CONFIG_PATH):
            logger.info("[RAG启动] ✅ MySQL 连接正常")
        else:
            logger.error("[RAG启动] ❌ MySQL 连接失败，请检查 MySQL 服务是否启动")
            raise RuntimeError("MySQL 连接失败，请检查 config.yaml 中 MySQL 配置并确保 MySQL 服务已启动")
    except Exception as e:
        logger.error(f"[RAG启动] ❌ MySQL 自检异常: {e}")
        raise

    logger.info("[RAG启动] 自检全部通过，服务启动中...")
    logger.info("=" * 50)


# ====================== 全局单例初始化（启动时只创建一次，避免每个请求重复实例化）======================
# 自检（连通性失败直接终止启动）
startup_self_check()

# 模型管理器
from myproject.models.models import ModelManager
model_manager = ModelManager(yaml_path=CONFIG_PATH)
logger.info("[RAG启动] ModelManager 初始化完成")

# 向量管理器（注入 model_manager，禁止内部 new 模型）
from myproject.embedding.text_embedding import EmbeddingVectorManager
embedding_manager = EmbeddingVectorManager(
    yaml_path=CONFIG_PATH,
    model_manager=model_manager
)
logger.info("[RAG启动] EmbeddingVectorManager 初始化完成")

# MySQL 数据管理器
from myproject.mysql_save_data.save_data_to_mysql import MysqlDataManager
mysql_manager = MysqlDataManager(yaml_path=CONFIG_PATH)
logger.info("[RAG启动] MysqlDataManager 初始化完成")

# 将 MySQL 管理器注入向量管理器（用于知识库名称映射）
embedding_manager.kb_meta_service = mysql_manager

# 向量数据库工厂
from myproject.embedding.vectorydb_factory import VectorDbFactory
vector_db_factory = VectorDbFactory(yaml_path=CONFIG_PATH)
logger.info("[RAG启动] VectorDbFactory 初始化完成")

# RAG 自动调度服务（全局单例，不在每个请求里重复创建）
from myproject.advancerag.auto_rag_service import AutoRagAutoDispatchService
rag_service = AutoRagAutoDispatchService(yaml_path=CONFIG_PATH)
logger.info("[RAG启动] AutoRagAutoDispatchService 初始化完成")


# ====================== FastAPI 应用（使用 lifespan 管理启动/关闭） ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("[RAG启动] RAG 智能知识库系统启动完成，监听 0.0.0.0:8000")
    yield
    # 关闭时清理资源
    logger.info("[RAG关闭] 正在关闭资源...")
    try:
        mysql_manager.close()
    except Exception:
        pass
    logger.info("[RAG关闭] 系统已关闭")


app = FastAPI(title="RAG智能知识库系统", lifespan=lifespan)

# ====================== CORS 跨域中间件（方便前端调试） ======================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请替换为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("[RAG启动] CORS 跨域中间件已启用")

# 挂载静态资源
app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")
templates = Jinja2Templates(directory=TEMPLATE_FOLDER)


# ====================== 统一标准返回工具 ======================
def success_response(data=None, message: str = "操作成功") -> dict:
    return {"code": 200, "message": message, "data": data}


def error_response(message: str = "操作失败", code: int = 500) -> dict:
    return {"code": code, "message": message, "data": None}


# ====================== 入参校验工具 ======================
def validate_required_fields(**kwargs) -> Optional[str]:
    """校验必填字段，返回第一个为空的字段名，全部通过返回 None"""
    for field_name, value in kwargs.items():
        if value is None:
            return field_name
        if isinstance(value, str) and not value.strip():
            return field_name
        if isinstance(value, (list, tuple, dict)) and len(value) == 0:
            return field_name
    return None


# ====================== 页面路由 ======================
@app.get('/')
def read_root(request: Request):
    return templates.TemplateResponse(request, "muti_function.html")


# ====================== 知识库相关接口 ======================
@app.post("/api/upload_file")
async def api_upload_file(
    collection_name: str = Form(..., description='知识库中文名称'),
    file: UploadFile = File(..., description='上传文件'),
    upload_type: str = Form("normal", description='入库类型：normal, parent_child, preview_question, summary, enrich, hybrid_search')
):
    """文件上传并向量化入库（保留原始文件名 + 流式写入 + 后缀白名单）"""
    # 入参校验
    empty_field = validate_required_fields(collection_name=collection_name)
    if empty_field:
        return error_response(f"参数 '{empty_field}' 不能为空", code=400)

    if not file.filename:
        return error_response("请选择有效文件", code=400)

    # 文件后缀白名单校验
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return error_response(
            f"不支持的文件类型 '{file_ext}'，允许: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            code=400
        )

    # 入库类型校验
    valid_types = ["normal", "parent_child", "preview_question", "summary", "enrich", "hybrid_search"]
    if upload_type not in valid_types:
        return error_response(f"不支持的入库类型 '{upload_type}'，允许: {', '.join(valid_types)}", code=400)

    save_filename = ""
    save_path = ""
    try:
        # 1. 保留原始文件名保存（MinerU输出目录就是中文名，方便用户识别）；重名自动加序号
        safe_name = os.path.basename(file.filename).replace('/', '_').replace('\\', '_')
        save_filename = safe_name
        save_path = os.path.join(UPLOAD_FOLDER, save_filename)
        # 重名处理：文件存在则加 _1, _2 ... 后缀
        if os.path.exists(save_path):
            stem, ext = os.path.splitext(safe_name)
            counter = 1
            while os.path.exists(os.path.join(UPLOAD_FOLDER, f"{stem}_{counter}{ext}")):
                counter += 1
            save_filename = f"{stem}_{counter}{ext}"
            save_path = os.path.join(UPLOAD_FOLDER, save_filename)

        with open(save_path, "wb") as buffer:
            # 流式读取，每次最多 64KB，防止大文件撑爆内存
            while chunk := await file.read(64 * 1024):
                buffer.write(chunk)

        logger.info(f"[RAG上传] 文件写入成功: {save_filename}")

        # 2. 执行向量化入库（传入入库类型）
        result = embedding_manager.file_to_embedding(
            file_path=save_path,
            collection_name=collection_name,
            upload_type=upload_type
        )

        if result is None:
            return error_response("文件向量化入库失败，请检查文件内容或日志", code=500)

        # 3. 同步写入 MySQL 知识库映射（包含支持的检索方法）
        real_collection = embedding_manager._normalize_collection_name(collection_name)
        support_methods = embedding_manager.get_support_methods(upload_type)
        display_name = f"{collection_name}{upload_type}"
        mysql_manager.add_kb_mapping(display_name, real_collection, support_methods)

        logger.info(f"[RAG上传] 向量入库成功: {display_name} → {real_collection}，入库类型: {upload_type}")
        return success_response(message="文件上传并入库成功")

    except Exception as e:
        logger.error(f"[RAG上传] 文件入库失败 [{collection_name}]: {e}")
        return error_response(f"文件入库失败: {str(e)}", code=500)


@app.post("/api/upload_text")
async def api_upload_text(
    collection_name: str = Form(..., description='知识库中文名称'),
    text: str = Form(..., description='文本内容'),
    upload_type: str = Form("normal", description='入库类型：normal, parent_child, preview_question, summary, enrich, hybrid_search'),
):
    """纯文本向量化入库"""
    # 入参校验
    empty_field = validate_required_fields(collection_name=collection_name, text=text)
    if empty_field:
        return error_response(f"参数 '{empty_field}' 不能为空", code=400)

    if not text.strip():
        return error_response("文本内容不能为空", code=400)

    # 入库类型校验
    valid_types = ["normal", "parent_child", "preview_question", "summary", "enrich", "hybrid_search"]
    if upload_type not in valid_types:
        return error_response(f"不支持的入库类型 '{upload_type}'，允许: {', '.join(valid_types)}", code=400)

    try:
        from myproject.datawash.data_wash import TextTanckle
        text_tackle = TextTanckle()
        doc_list = text_tackle.text_batch_tanckle(text_list=[text])
        if not doc_list:
            return error_response("文本解析后无有效内容", code=400)

        # 修复：中文知识库名称必须先规范化为底层英文集合名，否则 Chroma 报错
        real_collection = embedding_manager._normalize_collection_name(collection_name)
        
        # 执行向量化入库（传入入库类型）
        embedding_manager.simple_text_list_insert(
            text_list=[doc.page_content for doc in doc_list],
            collection_name=real_collection,
            upload_type=upload_type
        )

        # 同步映射（包含支持的检索方法）
        support_methods = embedding_manager.get_support_methods(upload_type)
        display_name = f"{collection_name}{upload_type}"
        mysql_manager.add_kb_mapping(display_name, real_collection, support_methods)

        logger.info(f"[RAG文本] 文本入库成功: {display_name}，入库类型: {upload_type}")
        return success_response(message="文本入库成功")

    except Exception as e:
        logger.error(f"[RAG文本] 文本入库失败 [{collection_name}]: {e}")
        return error_response(f"文本入库失败: {str(e)}", code=500)


@app.get("/api/get_knowledge_list")
async def api_get_knowledge_list():
    """获取全部知识库列表（从 MySQL 获取，元数据统一管理）"""
    try:
        kb_list = mysql_manager.get_all_kb_list()
        return success_response(data=kb_list)
    except Exception as e:
        logger.error(f"[RAG知识库] 获取列表失败: {e}")
        return error_response(f"获取知识库列表失败: {str(e)}", code=500)


@app.get("/api/get_collections")
async def api_get_collections():
    """
    获取 Chroma 向量库中实际存在的集合名称列表
    修复：GET 方法 + 提取名称列表（非原始对象，可 JSON 序列化）
    """
    try:
        chroma_client = vector_db_factory.get_chroma_client()
        # Chroma 原始集合对象无法 JSON 序列化，提取名称列表
        # 兼容不同版本 chromadb：旧版返回集合对象（有 .name），新版可能返回字符串
        raw_collections = chroma_client._client.list_collections()
        collection_names = []
        for c in raw_collections:
            if isinstance(c, str):
                collection_names.append(c)
            elif hasattr(c, 'name'):
                collection_names.append(c.name)
            else:
                collection_names.append(str(c))
        logger.info(f"[RAG集合] 获取到 {len(collection_names)} 个集合")
        return success_response(data=collection_names)
    except Exception as e:
        logger.error(f"[RAG集合] 获取集合列表失败: {e}")
        return error_response(f"获取集合列表失败: {str(e)}", code=500)


@app.post("/api/collection_delete")
async def api_collection_delete(
    collection_name: str = Form(..., description='知识库中文名称'),
):
    """删除知识库（向量库 + MySQL 映射同步删除）"""
    empty_field = validate_required_fields(collection_name=collection_name)
    if empty_field:
        return error_response(f"参数 '{empty_field}' 不能为空", code=400)

    try:
        real_collection = embedding_manager._normalize_collection_name(collection_name)
        chroma_client = vector_db_factory.get_chroma_client()

        try:
            # 兼容不同版本 chromadb 的 delete_collection 参数名
            try:
                chroma_client._client.delete_collection(name=real_collection)
            except TypeError:
                chroma_client._client.delete_collection(real_collection)
            logger.info(f"[RAG知识库] 向量集合删除: {real_collection}")
        except Exception as e:
            logger.warning(f"[RAG知识库] 向量集合删除失败（可能不存在）: {e}")

        mysql_manager.delete_kb_mapping(collection_name)
        return success_response(message="知识库删除成功")

    except Exception as e:
        logger.error(f"[RAG知识库] 删除失败 [{collection_name}]: {e}")
        return error_response(f"删除失败: {str(e)}", code=500)


# ====================== 会话相关接口 ======================
@app.get("/api/get_conversation_list")
async def api_get_conversation_list():
    """获取全部历史会话列表（MySQL 为准）"""
    try:
        conv_list = mysql_manager.get_conversation_list()
        return success_response(data=conv_list)
    except Exception as e:
        logger.error(f"[RAG会话] 获取列表失败: {e}")
        return error_response(f"获取会话列表失败: {str(e)}", code=500)


@app.get("/api/get_redis_session_ids")
async def api_get_redis_session_ids():
    """获取 Redis 中现存的会话 ID（调试用）"""
    try:
        from myproject.conversation_save.save_conversation_history_redis import SaveChatRedis
        ids = SaveChatRedis.get_all_session_ids()
        return success_response(data=ids)
    except Exception as e:
        logger.error(f"[RAG会话] 获取 Redis 会话失败: {e}")
        return error_response(f"获取 Redis 会话失败: {str(e)}", code=500)


@app.post("/api/create_conversation")
async def api_create_conversation(
    kb_display_name: str = Form(..., description="绑定的知识库名称"),
    llm_model: str = Form("", description="使用模型")
):
    """创建新会话，返回会话 ID"""
    empty_field = validate_required_fields(kb_display_name=kb_display_name)
    if empty_field:
        return error_response(f"知识库名称不能为空", code=400)

    try:
        conv_id = uuid.uuid4().hex
        mysql_manager.create_conversation(
            conversation_id=conv_id,
            kb_display_name=kb_display_name,
            llm_model=llm_model or "",
            title="新对话"
        )
        logger.info(f"[RAG会话] 创建成功: {conv_id}")
        return success_response(data={"conversation_id": conv_id}, message="会话创建成功")

    except Exception as e:
        logger.error(f"[RAG会话] 创建失败: {e}")
        return error_response(f"创建会话失败: {str(e)}", code=500)


@app.get("/api/get_conversation_history")
async def api_get_conversation_history(conversation_id: str = ""):
    """加载单条会话的全部历史消息（Redis 优先，MySQL 兜底）"""
    if not conversation_id or not conversation_id.strip():
        return error_response("会话 ID 不能为空", code=400)

    try:
        from myproject.conversation_save.save_conversation_history_redis import SaveChatRedis

        # 1. 先查 Redis
        redis_chat = SaveChatRedis(sessionID=conversation_id)
        history = redis_chat.get_langchain_history()
        if history:
            # 修复：统一 role 字段为 user/assistant，前端只识别这两种值
            msg_list = []
            for msg in history:
                role = "user" if msg.type == "human" else "assistant"
                msg_list.append({"role": role, "content": msg.content})
            return success_response(data=msg_list)

        # 2. Redis 没有，查 MySQL 兜底
        msg_list = mysql_manager.get_messages_by_conv_id(conversation_id)
        # 同步加载回 Redis 缓存
        if msg_list:
            try:
                # 修复：MySQL 中 role 是 user/assistant，Redis 需要 human:/ai: 前缀
                role_map = {"user": "human", "assistant": "ai"}
                format_msgs = [f"{role_map.get(item['role'], item['role'])}:{item['content']}" for item in msg_list]
                redis_chat.redis_save(chat_data=format_msgs)
            except Exception as e:
                logger.warning(f"[RAG会话] 回写 Redis 缓存失败: {e}")

        return success_response(data=msg_list)

    except Exception as e:
        logger.error(f"[RAG会话] 加载历史消息失败 [{conversation_id}]: {e}")
        return error_response(f"加载历史消息失败: {str(e)}", code=500)


@app.post("/api/conversation_delete")
async def api_conversation_delete(
    conversation_id: str = Form(..., description='会话ID'),
):
    """删除会话（Redis + MySQL 同步删除）"""
    empty_field = validate_required_fields(conversation_id=conversation_id)
    if empty_field:
        return error_response(f"会话 ID 不能为空", code=400)

    try:
        from myproject.conversation_save.save_conversation_history_redis import SaveChatRedis
        redis_chat = SaveChatRedis(sessionID=conversation_id)
        redis_chat.redis_sessionID_delete()
        mysql_manager.delete_conversation(conversation_id)
        logger.info(f"[RAG会话] 删除成功: {conversation_id}")
        return success_response(message="会话删除成功")

    except Exception as e:
        logger.error(f"[RAG会话] 删除失败 [{conversation_id}]: {e}")
        return error_response(f"删除失败: {str(e)}", code=500)


# ====================== RAG 核心问答接口 ======================
@app.post("/api/rag_chat")
async def api_rag_chat(
    user_query: str = Form(..., description='用户问题'),
    collection_name: str = Form(..., description='知识库中文名称'),
    session_id: Optional[str] = Form(None, description='会话ID'),
    persist_time: Optional[int] = Form(None, description='Redis持久化时间'),
    embedding_text: Optional[str] = Form(None, description='嵌入文本'),
    custom_compress_type: Optional[str] = Form(None, description='自定义压缩类型'),
    pre_retrieval_methods: Optional[str] = Form(None, description='用户选择的预检索方法，逗号分隔'),
    enable_rrf: Optional[str] = Form("true", description='是否启用RRF互逆排序融合（true/false）'),
    enable_model_rerank: Optional[str] = Form("true", description='是否启用模型重排（true/false）'),
):
    """RAG 智能问答（使用全局单例 rag_service，不再每次请求重复实例化）"""
    # 入参校验
    empty_field = validate_required_fields(user_query=user_query, collection_name=collection_name)
    if empty_field:
        return error_response(f"参数 '{empty_field}' 不能为空", code=400)

    if not user_query.strip():
        return error_response("问题不能为空", code=400)

    try:
        # 1. 执行 RAG 问答（使用全局单例）
        # 修复：中文知识库名称必须先规范化为底层英文集合名，否则检索时集合名不匹配
        real_collection = embedding_manager._normalize_collection_name(collection_name)

        # 解析用户选择的预检索方法（前端传逗号分隔字符串，转为列表）
        user_pre_methods = None
        if pre_retrieval_methods and pre_retrieval_methods.strip():
            user_pre_methods = [m.strip() for m in pre_retrieval_methods.split(',') if m.strip()]

        answer = rag_service.run_rag_task(
            user_question=user_query,
            collection_name=real_collection,
            vector_text_list=[embedding_text] if embedding_text else [],
            custom_compress_type=custom_compress_type,
            session_id=session_id,
            persist_time=persist_time,
            user_pre_methods=user_pre_methods,
            enable_rrf=enable_rrf.lower() == "true" if enable_rrf else True,
            enable_model_rerank=enable_model_rerank.lower() == "true" if enable_model_rerank else True,
        )

        # 2. 如果传了会话 ID，同步持久化到 MySQL
        if session_id:
            try:
                mysql_manager.add_message(session_id, "user", user_query)
                mysql_manager.add_message(session_id, "assistant", str(answer))
                # 自动用第一个问题更新会话标题
                conv_list = mysql_manager.get_messages_by_conv_id(session_id)
                if len(conv_list) == 2:
                    title = user_query[:20] if len(user_query) > 20 else user_query
                    mysql_manager.update_conversation_title(session_id, title)
            except Exception as e:
                logger.warning(f"[RAG问答] MySQL 持久化失败: {e}")

        logger.info(f"[RAG问答] 问答成功 [{collection_name}]: {user_query[:30]}...")
        return success_response(data={"answer": answer}, message="问答成功")

    except Exception as e:
        logger.error(f"[RAG问答] 问答失败 [{collection_name}]: {e}")
        return error_response(f"问答失败: {str(e)}", code=500)


# ====================== 启动入口 ======================
if __name__ == '__main__':
    import uvicorn
    import socket

    # 从配置文件读取端口（支持环境变量覆盖）
    try:
        from config.config_load import config_load
        cfg = config_load(CONFIG_PATH)
        host = os.getenv("RAG_HOST", cfg.get("api", {}).get("host", "0.0.0.0"))
        base_port = int(os.getenv("RAG_PORT", cfg.get("api", {}).get("port", 8000)))
    except Exception:
        host = "0.0.0.0"
        base_port = 8000

    # 查找可用端口
    port = base_port
    max_attempts = 10
    for i in range(max_attempts):
        test_port = base_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, test_port))
                port = test_port
                break
        except socket.error:
            logger.warning(f"[RAG启动] 端口 {test_port} 被占用，尝试下一个...")
    else:
        raise RuntimeError(f"[RAG启动] 端口 {base_port}-{base_port + max_attempts - 1} 全部被占用")
    
    logger.info(f"[RAG启动] 服务启动: {host}:{port}")
    uvicorn.run(app, host=host, port=port)