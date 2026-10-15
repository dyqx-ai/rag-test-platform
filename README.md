## 项目定位

**这不是一个问答产品，是一台 RAG 策略实验台。**

传统 RAG 项目往往是"一条固定流水线写死"，用户无法感知不同检索、压缩、重排策略对最终答案的影响。本项目把 RAG 拆成**五个可插拔的层**，每一层都提供多个可选实现，用户可以自由组合、横向对比、沉淀出适合自己场景的最佳方案。

**核心价值**：

- 🧪 **策略实验**：8 种预检索 × 2 种重排 × 4 种压缩，自由组合
- 🔌 **可插拔架构**：每层统一接口，新增插件成本低
- 🔍 **中间过程可见**：详细日志打印，方便观察每一步
- 📊 **多策略对比**：同一问题不同参数组合，效果横向对比
- 🖼️ **多模态支持**：文档图片提取 + 多模态模型图文联合问答
- 💾 **持久化保障**：向量化一次性落盘，Redis + MySQL 双写

------

## 核心特性

### 📄 文档解析

- **多格式支持**：PDF / DOCX / TXT / MD
- **双轨解析**：PyMuPDF / mammoth 原生解析优先，MinerU 兜底
- **图片提取**：图片落盘 + 文本占位符 + metadata 关联
- **专业场景**：药品说明书、器械说明书专用分片策略
- **排版还原**：PDF 按坐标混排图文表，保持阅读顺序

### 🧮 向量化入库

- **多种入库类型**：普通向量 / 父子文档 / 预览问题 / 摘要多向量 / 字段增强 / 混合检索
- **中文知识库名**：自动拼音转换 + MySQL 映射持久化
- **本地/云端 Embedding**：Ollama bge-m3 / DashScope text-embedding-v3
- **写后验证**：入库后自动校验，避免静默失败

### 🔍 多策略检索

**8 种预检索策略**：

| 策略               | 适用场景                              |
| :----------------- | :------------------------------------ |
| `multi_query`      | 宽泛开放式问题，多视角召回            |
| `decomposition`    | 复合问题，拆分为独立子问题            |
| `parent_child`     | 需要完整长原文上下文                  |
| `self_query`       | 带时间/分类/数值筛选条件的精准查询    |
| `preview_question` | FAQ 类知识库，匹配同类衍生问题        |
| `summary`          | 摘要类问题，匹配浓缩语义              |
| `enrich`           | 结构化业务字段补全                    |
| `hybrid_search`    | 含专业专有名词的查询，BM25 + 向量混合 |

**决策引擎**：LLM 驱动的"因问施策"智能路由，也支持用户手动指定。

### 🎛️ 后置优化

- **RRF 融合**：多路检索结果倒数排名融合
- **模型重排**：DashScope gte-rerank-v2 / qwen3-rerank
- **4 种压缩模式**：LLM 提取 / LLM 过滤 / 向量过滤 / 混合流水线

### 🖼️ 图文联合问答

- 文档解析时图片落盘 + 文本埋 `[图片:img_n]` 占位符
- 分片时正则提取 `img_id_list` 存入 metadata
- 生成时按 ID 读盘 → base64 → 喂多模态模型

### 💾 持久化

- **Redis**：会话缓存，TTL 可配
- **MySQL**：知识库映射 + 会话主表 + 消息明细
- **SQLite**：轻量备选存储
- **Chroma**：向量库持久化

------

## 整体架构

text

```
                                    ┌─────────────────────────────────┐
                                    │  前端 muti_function.html         │
                                    │  ┌───────────────────────────┐  │
                                    │  │ 对话列表 / 知识库管理       │  │
                                    │  │ 高级参数面板：              │  │
                                    │  │   • 8 种预检索 checkbox    │  │
                                    │  │   • 压缩类型下拉           │  │
                                    │  │   • RRF 开关              │  │
                                    │  │   • 模型重排开关          │  │
                                    │  └───────────────────────────┘  │
                                    └──────────────┬──────────────────┘
                                                   │ HTTP
                                    ┌──────────────▼──────────────────┐
                                    │  main_fastapi.py                 │
                                    │  ┌───────────────────────────┐  │
                                    │  │ 启动自检：配置文件/Redis/  │  │
                                    │  │          MySQL 连通性     │  │
                                    │  └───────────────────────────┘  │
                                    │  全局单例：                      │
                                    │   model_manager                  │
                                    │   embedding_manager              │
                                    │   mysql_manager                  │
                                    │   vector_db_factory              │
                                    │   rag_service                    │
                                    └──────┬────────────────────┬──────┘
                                           │                    │
                        入库链路 ──────────┘                    └────── 问答链路
                              │                                          │
              ┌───────────────▼─────────────┐            ┌───────────────▼─────────────┐
              │ EmbeddingVectorManager       │            │ AutoRagAutoDispatchService  │
              │  file_to_embedding           │            │  run_rag_task               │
              │  simple_text_list_insert     │            │  run_full_auto_rag          │
              └───────────────┬─────────────┘            └───────────────┬─────────────┘
                              │                                          │
                              ▼                                          ▼
                    数据处理链路                                问答处理链路
              （详见下方数据链路章节）                      （详见下方数据链路章节）
```



------

## 五大插拔层

这是项目的灵魂。每一层都是「统一接口 + 可替换插件」，层与层之间独立，用户自由组合。

text

```
┌─────────────────────────────────────────────────────────────┐
│  插槽 1  模型层      config 驱动      （全局，重启生效）        │
│  插槽 2  入库类型    建库时定一次      （决定知识库"长什么样"） │
│  插槽 3  预检索方法  每次提问多选      （决定"怎么召回"）      │
│  插槽 4  重排        每次提问可开关    （决定"多路怎么合并"）   │
│  插槽 5  压缩        每次提问可换      （决定"怎么瘦身"）      │
└─────────────────────────────────────────────────────────────┘
```



### 插槽 1：模型层

| 模型类型  | 可选实现                                  | 切换位置                     |
| :-------- | :---------------------------------------- | :--------------------------- |
| 对话模型  | qwen / deepseek                           | `config.llm.use_model`       |
| Embedding | Ollama 本地 / DashScope 云端              | `config.embedding.use_local` |
| Rerank    | DashScope（gte-rerank-v2 / qwen3-rerank） | `config.rerank.cloud.model`  |
| 多模态    | qwen-vl-max                               | `config.multimodal_model`    |

**扩展方式**：`models.py` 里加分支 + config 加配置段 + `.env` 加 key。

### 插槽 2：入库类型

| 入库类型           | 建什么索引                     | 支持哪些检索              | 跳过基础入库 |
| :----------------- | :----------------------------- | :------------------------ | :----------- |
| `normal`           | 基础向量                       | normal                    | 否           |
| `parent_child`     | 父子双索引                     | parent_child / normal     | ✅            |
| `preview_question` | 预览问题向量 + 原文 byte_store | preview_question / normal | ✅            |
| `summary`          | 摘要向量 + 原文 byte_store     | summary / normal          | ✅            |
| `enrich`           | 字段模板向量                   | enrich / normal           | ✅            |
| `hybrid_search`    | 基础向量 + BM25 索引           | hybrid_search / normal    | 否           |

**注册表**：`EmbeddingVectorManager.UPLOAD_TYPES`

### 插槽 3：预检索方法

**两个注册表**：

python

```
# A 类：返回 retriever 对象，需要再 .invoke(query)
retriever_method_map = {
    "parent_child":     {...},
    "self_query":       {...},
    "preview_question": {...},
    "summary":          {...},
    "hybrid_search":    {...},
}

# B 类：直接返回 List[Document]
direct_doc_method_map = {
    "multi_query":   {...},
    "decomposition": {...},
}

# 特殊公民
ENRICH_NAME = "enrich"
```



**决策优先级**：用户选择 > LLM 决策 > 默认 `multi_query`

### 插槽 4：重排

| 插件     | 开关                  | 实现               |
| :------- | :-------------------- | :----------------- |
| RRF 融合 | `enable_rrf`          | `_rrf_fusion_sort` |
| 模型重排 | `enable_model_rerank` | `_ali_rerank_sort` |

**RRF 公式**：`score(doc) = Σ 1 / (rank + k)`，k 默认 60。

### 插槽 5：压缩

| 压缩类型     | 实现类                       | 行为                                                |
| :----------- | :--------------------------- | :-------------------------------------------------- |
| `extract`    | `LLMChainExtractor`          | LLM 从每篇抽取相关片段                              |
| `llm_filter` | `LLMChainFilter`             | LLM 判断每篇留或丢                                  |
| `emb_filter` | `EmbeddingsFilter`           | 向量相似度过滤                                      |
| `pipeline`   | `DocumentCompressorPipeline` | 四连击：splitter → 冗余过滤 → 相似度过滤 → LLM 过滤 |

------

## 完整数据链路

### 文件入库链路

text

```
【用户上传文件】
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ main_fastapi.api_upload_file                                        │
│  ① 校验参数 → ② 校验后缀 → ③ 保存文件到 UPLOAD_FOLDER                │
│  ④ 调 embedding_manager.file_to_embedding                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ EmbeddingVectorManager.file_to_embedding                            │
│  Step 1  集合名规范化 _normalize_collection_name                     │
│          ├── 查 MySQL：中文名已有映射？ → 直接用现成英文集合名         │
│          ├── 没命中 → 拼音转换 + 清洗（只留 a-z0-9_-，长度 3-63）      │
│          └── 转换后写回 MySQL：add_kb_mapping                        │
│                                                                     │
│  Step 2  文件遍历（文件/目录）                                       │
│                                                                     │
│  Step 3  批量解析 data_wash().multi_data_wash()                     │
│          ├── .txt/.md  → 文本读取                                    │
│          ├── .docx    → mammoth 原生解析 → 失败降级 MinerU           │
│          └── .pdf     → PyMuPDF 原生解析 → 失败降级 MinerU           │
│                                                                     │
│  Step 4  分片 TextTanckle.text_batch_tanckle                        │
│          ├── 清洗 → 全文 sha256 → 按 markdown 标题切分               │
│          ├── 提取图片 ID → 分片 sha256 → 封装 Document               │
│          └── 三种切割策略：通用 / 药品 / 器械                        │
│                                                                     │
│  Step 5  过滤空内容 Document                                        │
│                                                                     │
│  Step 6  判断 need_skip_base                                        │
│          ├── parent_child/preview_question/summary/enrich → 跳过    │
│          └── normal/hybrid_search → embedding_manager_choice 入库   │
│                                                                     │
│  Step 7  写后验证（重新打开 Chroma count 一下）                      │
│                                                                     │
│  Step 8  非 normal 时构建进阶索引 pre_retrieval.xxx                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ main_fastapi 继续：                                                  │
│  ⑤ mysql_manager.add_kb_mapping(display_name, real_collection, ...) │
└─────────────────────────────────────────────────────────────────────┘
```



### 问答链路

text

```
【用户提问 + 勾选参数】
      │
      ▼
┌──────────────────────────────────────────────────────────────────────┐
│ main_fastapi.api_rag_chat                                            │
│  ① 校验参数 → ② 集合名规范化 → ③ 解析预检索方法列表                   │
│  ④ 调 rag_service.run_rag_task                                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ AutoRagAutoDispatchService.run_rag_task                              │
│  ① 空问题拦截 ② 默认值覆盖 ③ 用户方法列表清洗 ④ 文本 → Document       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ run_full_auto_rag  五步主流程                                         │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Step 1  决策                                                    │  │
│  │   decision_engine.decide_strategy(user_query, user_pre_methods) │  │
│  │   ├── 用户勾了 normal → 直接返回 {False, []}                     │  │
│  │   ├── 用户勾了其他 → 拼 user_hint 注入 prompt                    │  │
│  │   ├── 用户没勾 → 纯 LLM 决策                                     │  │
│  │   ├── LLM 结果空但有用户选择 → 降级用用户选择                    │  │
│  │   └── LLM 异常 → 有用户选择用用户选择，无则默认 multi_query      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                               │                                       │
│  ┌────────────────────────────▼───────────────────────────────────┐  │
│  │ Step 2  检索                                                    │  │
│  │   retrieval_dispatcher.retrieve(...)                            │  │
│  │   ├── use_pre_retrieval=True → dispatch_pre_retrieval           │  │
│  │   │     ├── 遍历 direct_doc_method_map                          │  │
│  │   │     ├── 遍历 retriever_method_map                           │  │
│  │   │     ├── enrich 跳过                                         │  │
│  │   │     └── 全部失败兜底 _get_base_single_retrieve_groups       │  │
│  │   └── use_pre_retrieval=False → 基础单路检索                    │  │
│  │   → List[List[Document]]                                        │  │
│  └────────────────────────────┬───────────────────────────────────┘  │
│                               │                                       │
│  ┌────────────────────────────▼───────────────────────────────────┐  │
│  │ Step 3  后置处理                                                 │  │
│  │   post_processor.process_context(...)                           │  │
│  │   ├── doc_rerank_all                                            │  │
│  │   │     ├── enable_rrf=True → _rrf_fusion_sort                  │  │
│  │   │     └── enable_model_rerank=True → _ali_rerank_sort         │  │
│  │   └── doc_compress_all                                          │  │
│  │         ├── extract / llm_filter / emb_filter / pipeline        │  │
│  │         └── 压缩后空 → 降级原文                                  │  │
│  └────────────────────────────┬───────────────────────────────────┘  │
│                               │                                       │
│  ┌────────────────────────────▼───────────────────────────────────┐  │
│  │ Step 4  生成                                                     │  │
│  │   _build_generate_prompt(retrieve_groups, user_query, context)   │  │
│  │   ├── 从 retrieve_groups 收集 img_id_list                       │  │
│  │   ├── 无图片 → 纯文本 prompt                                    │  │
│  │   └── 有图片 → messages 数组 → 多模态模型                       │  │
│  │   ├── 多模态可用 → multimodal_llm.invoke                        │  │
│  │   └── 多模态不可用 → 抠出 text 部分 → 纯文本 LLM                │  │
│  └────────────────────────────┬───────────────────────────────────┘  │
│                               │                                       │
│  ┌────────────────────────────▼───────────────────────────────────┐  │
│  │ Step 5  持久化                                                   │  │
│  │   chat_saver.save_chat(user_query, answer, session_id, ...)      │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ main_fastapi 继续：                                                   │
│  ⑤ mysql_manager.add_message × 2 + update_conversation_title         │
│  ⑥ 返回 success_response(data={"answer": answer})                     │
└──────────────────────────────────────────────────────────────────────┘
```



### 图片链路

text

```
解析时：图片落盘 img_n.png + 文本埋 [图片:img_n] 占位符
    ↓
分片时：正则提取 img_id_list → 塞进 metadata
    ↓
入库时：metadata 跟着 Document 存进向量库
    ↓
检索时：命中的 doc 从 metadata 拿 img_id_list
    ↓
生成时：按 ID 读盘 → base64 → 喂多模态模型
```



------

## 项目结构

text

```
myproject/
├── config/
│   ├── config.yaml              # 全局配置文件
│   └── config_load.py           # 配置加载（相对路径 → 绝对路径）
│
├── models/
│   └── models.py                # 统一模型管理器（chat / embedding / rerank / 多模态）
│
├── datawash/                    # 文档解析与清洗
│   ├── mineru_wash.py           # MinerU CLI 封装（兜底解析）
│   ├── muti_catolog_file_wash.py # 解析调度（原生 + MinerU 双轨）
│   └── data_wash.py             # 文本切割（三种策略 + hash + 图片提取）
│
├── embedding/                   # 向量化入库
│   ├── vectorydb_factory.py     # 向量库工厂（Chroma 实现）
│   └── text_embedding.py        # 嵌入向量管理器（6 种入库类型）
│
├── advancerag/                  # 高级 RAG
│   ├── pre_retrieval.py         # 9 种预检索插件
│   ├── rag_retrieval_dispatcher.py # 检索调度器
│   ├── rag_decision_engine.py   # 策略决策引擎
│   ├── post_retrieval.py        # 后置处理（重排 + 压缩）
│   ├── rag_post_processor.py    # 后置处理器封装
│   ├── rag_chat_saver.py        # 对话保存器
│   └── auto_rag_service.py      # 顶层编排服务
│
├── conversation_save/           # 会话持久化
│   ├── choose_conversation_save_method.py
│   ├── save_conversation_history_redis.py
│   └── save_conversation_history_sqlite.py
│
├── mysql_save_data/             # MySQL 持久化
│   └── save_data_to_mysql.py
│
├── html/
│   └── muti_function.html       # 前端页面
│
├── main_fastapi.py              # FastAPI 入口
├── requirements.txt             # 依赖清单
└── README.md                    # 本文档
```



------

## 环境要求

- **Python**：3.10+
- **Chroma**：向量数据库（内置，无需额外部署）
- **Redis**：会话缓存（可选，但不装会影响 summary/preview_question 检索）
- **MySQL**：持久化存储（必须）
- **Docker**：可选，用于沙箱场景
- **MinerU**：可选，PDF/DOCX 兜底解析
- **Ollama**：可选，本地 Embedding

------

## 快速开始

### 1. 安装依赖

bash

```
pip install -r requirements.txt
```



### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

bash

```
ALI_API_KEY=your_dashscope_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```



### 3. 修改配置文件

编辑 `config/config.yaml`，重点配置：

yaml

```
# MySQL 配置
save_method:
  mysql:
    host: localhost
    port: 3306
    user: root
    password: "your_password"
    database: myRAG_Info

# LLM 模型选择
llm:
  use_model: "qwen"  # 或 deepseek

# Embedding 本地/云端
embedding:
  use_local: true  # 本地 Ollama
```



### 4. 启动服务

bash

```
python main_fastapi.py
```



服务默认监听 `0.0.0.0:8000`，访问 `http://localhost:8000` 即可。

### 5. 使用流程

text

```
① 上传文档 → 选入库类型 → 建知识库
② 进入对话 → 选知识库 → 勾选预检索方法
③ 提问 → 观察结果
④ 换一组参数 → 再提问 → 横向对比
⑤ 记录最佳组合
```



------

## 配置说明

### 全局路径配置

yaml

```
path:
  mineru_exe: "C:/pyProject/mineru/pdf_env/Scripts/mineru.exe"
  output_img: "../output/img"
  output_md: "../output/md"
  storage_vector: "../storage/vector_db"
```



### 文本分片规则

yaml

```
split:
  chunk_size: 300
  chunk_overlap: 30

parent_splitter:
  chunk_size: 1000
  chunk_overlap: 100

child_splitter:
  chunk_size: 200
  chunk_overlap: 20
```



### LLM 模型

yaml

```
llm:
  use_model: "qwen"
  qwen_model: "qwen-max"
  qwen_base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  deepseek_model: "deepseek-v4-pro"
  deepseek_base_url: "https://api.deepseek.com"
```



### Rerank 配置

yaml

```
rerank:
  enable: true
  use_local: false
  threshold: 0.35
  top_n: 4
  cloud:
    provider: "aliyun"
    model: "gte-rerank-v2"
    base_url: "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
```



### RAG 控制开关

yaml

```
rag:
  enable_multi_retrieval: true
  enable_context_compress: true
  enable_multi_query: true
  enable_decomposition: true
  max_context_window: 2800
  multi_query_count: 3
  decomposition_count: 3
  rrf_smooth_k: 60
```



### 存储配置

yaml

```
save_method:
  if_save: true
  method: "redis"  # 或 sqlite
  redis:
    host: localhost
    port: 6379
    db: 0
    persist_time: 60
```



------

## API 接口

### 知识库管理

| 方法 | 路径                      | 说明                 |
| :--- | :------------------------ | :------------------- |
| POST | `/api/upload_file`        | 文件上传并向量化入库 |
| POST | `/api/upload_text`        | 纯文本向量化入库     |
| GET  | `/api/get_knowledge_list` | 获取知识库列表       |
| GET  | `/api/get_collections`    | 获取 Chroma 集合列表 |
| POST | `/api/collection_delete`  | 删除知识库           |

### 会话管理

| 方法 | 路径                            | 说明         |
| :--- | :------------------------------ | :----------- |
| GET  | `/api/get_conversation_list`    | 获取会话列表 |
| POST | `/api/create_conversation`      | 创建新会话   |
| GET  | `/api/get_conversation_history` | 加载会话历史 |
| POST | `/api/conversation_delete`      | 删除会话     |

### RAG 问答

| 方法 | 路径            | 说明         |
| :--- | :-------------- | :----------- |
| POST | `/api/rag_chat` | 核心问答接口 |

**`/api/rag_chat` 参数**：

| 参数                    | 类型 | 说明                   |
| :---------------------- | :--- | :--------------------- |
| `user_query`            | str  | 用户问题               |
| `collection_name`       | str  | 知识库中文名           |
| `session_id`            | str  | 会话 ID                |
| `pre_retrieval_methods` | str  | 预检索方法（逗号分隔） |
| `custom_compress_type`  | str  | 压缩类型               |
| `enable_rrf`            | bool | 是否启用 RRF 融合      |
| `enable_model_rerank`   | bool | 是否启用模型重排       |
| `embedding_text`        | str  | 附加向量化文本         |

------

## 核心模块详解

### `config_load.py` —— 配置地基

**核心目的**：相对路径 → 绝对路径，保证项目可移植性 + 运行目录无关性。

python

```
_CONFIG_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))  # config/
_PROJECT_ROOT = os.path.dirname(_CONFIG_MODULE_DIR)              # myproject/
```



`__file__` 永远是这个文件自己的位置，跟从哪启动无关。

### `models.py` —— 模型管理

**核心思路**：所有模型的统一出口，配置驱动插拔。

**DIY 新模型**：在对应方法里加分支 + config 加配置段 + `.env` 加 key。

### `data_wash.py` (TextTanckle) —— 文本切割

**三种切割策略同签名**：

python

```
(text, is_markdown, text_source) -> Optional[List[Document]]
```



- `split_text_to_document` —— 通用
- `special_text_medical` —— 药品说明书（父子两层 + mark 标记）
- `special_text_device` —— 器械说明书（按类别拆）

**扩展方式**：类里加方法，通过 `text_batch_tanckle(method=...)` 或 `file_to_embedding(split_method=...)` 传入。

### `text_embedding.py` —— 入库调度

**两个并列入口**：

- `file_to_embedding` —— 文件进来 → 解析 → 分片 → 入库
- `simple_text_list_insert` —— 纯文本进来 → 包 Document → 入库

**`_normalize_collection_name` 三步走**：

1. 先查 MySQL（有缓存效果）
2. 拼音转换 + 清洗
3. 写回 MySQL 映射

### `pre_retrieval.py` —— 9 种检索插件

**两个注册表**：

- `retriever_method_map` —— 返回 retriever 对象
- `direct_doc_method_map` —— 直接返回文档列表

**关键方法逐个说明**：

- `summary` / `preview_question_index`：LLM 生成内容进向量库，原文进 byte_store
- `parent_child`：父块进 docstore，子块进向量库
- `self_query`：LLM 生成元数据字段定义 → `exec` 执行
- `enrich`：含 `input()` 交互，自动链路跳过
- `multi_query` / `decomposition`：LLM 生成查询 → 分别检索 → 去重
- `hybrid_search`：BM25 + 向量加权融合

### `rag_decision_engine.py` —— 决策引擎

**决策优先级**：用户选择 > LLM 决策 > 默认 `multi_query`

**五条路径**：

1. 用户勾 `normal` → 直接返回，不调 LLM
2. 用户勾了其他 → 拼 `user_hint` 注入 prompt
3. 用户没勾 → 纯 LLM 决策
4. LLM 结果空但有用户选择 → 用用户选择
5. LLM 异常 → 降级

### `post_retrieval.py` —— 重排 + 压缩

**重排两个插件**：

- `_rrf_fusion_sort`：用 `dumps(doc)` 序列化做 key
- `_ali_rerank_sort`：六层容错逻辑

**压缩四个插件**：

- `extract` / `llm_filter` / `emb_filter` / `pipeline`

------

## 扩展指南

### 添加新的预检索方法

需要改 4 个地方：

1. `pre_retrieval.py` 加方法
2. `rag_retrieval_dispatcher.py` 注册表加一项
3. `rag_decision_engine.valid_pre_methods` 加白名单
4. 前端 checkbox 加一个

### 添加新的压缩类型

1. `post_retrieval.py` 加 `_compress_xxx`
2. `doc_compress_all` 加分支
3. `RagPostProcessor.valid_compress_types` 加白名单
4. 前端下拉加一项

### 添加新的入库类型

1. `text_embedding.UPLOAD_TYPES` 加一项
2. `file_to_embedding` / `simple_text_list_insert` 加分支
3. 前端下拉加一项

### 添加新的切割策略

在 `TextTanckle` 类里加方法，遵守签名：

python

```
def special_text_xxx(self, text: str, is_markdown: bool, text_source: str) -> Optional[List[Document]]:
    ...
```



然后通过 `text_batch_tanckle(method=...)` 或 `file_to_embedding(split_method=...)` 传入。

### 添加新的模型厂商

在 `models.py` 对应 `init_*_model` 加分支 + `config.yaml` 加配置段 + `.env` 加 key。

------

## 已知限制

### 🔴 严重

- **去重缺失**：hash 生成了但没用起来，同一文件重复上传会产生重复 chunk
- **图片与压缩后文本可能错配**：图片从原始检索结果提取，文本是压缩后的
- **`_collection_exists` 粒度太粗**：同名不同版本会误判为"已存在"
- **Redis 消息顺序可能反了**：`lpush` 头插 + `lrange` 正序读 = 整体反序

### 🟠 中等

- **`enrich` 前端勾了没反应**：含 `input()` 交互，自动链路跳过
- **压缩的"多文档 → 一段文本"降维**：文档级过滤只有 1 篇可判断
- **`run_full_pipeline` / `multi_query_retrieve` 是孤儿**：主链路不走
- **`self.enable_rerank` 死配置**：读了但没用
- **`summary` / `preview_question` 依赖 Redis**：Redis 关了用 InMemoryByteStore，重启就丢
- **`self_query` 用 `exec` 执行 LLM 代码**：理论风险
- **BM25 每次重建**：大库慢

### 🟡 轻度

- 加插件要动多处（4 个地方）
- 清洗策略未暴露到前端
- 重复代码（两个 embedding 方法完全一样等）
- `data:image/jpeg` 前缀固定
- `run_full_auto_rag` 每次问答都查 Chroma 集合列表
- 参数名不一致（`user_question` vs `user_query`）

------

## 后续规划

按 ROI 排序：

| 优先级 | 事项                                               | 投入 | 收益             |
| :----- | :------------------------------------------------- | :--- | :--------------- |
| 🥇      | 补 chunk 级去重（`add_documents(ids=[chunk_id])`） | 一行 | 检索结果干净     |
| 🥇      | 修图片与压缩的错配                                 | 小   | 图文精度提升     |
| 🥈      | 补文件级去重（MySQL 文件 hash 表）                 | 小   | 省解析时间       |
| 🥈      | 验证 Redis 消息顺序                                | 小   | 修 bug           |
| 🥈      | 接通 `enrich`（前端单独交互入口）                  | 中   | 释放插件价值     |
| 🥉      | 清洗策略暴露到前端                                 | 中   | 扩展场景         |
| 🥉      | 统一加插件的注册点                                 | 中   | 降低扩展成本     |
| 🥉      | 清理死代码 / 重复代码                              | 中   | 可维护性         |
| 📌      | 把 `print` 换成 logging                            | 中   | 生产可用         |
| 📌      | 前端加"过程 trace"面板                             | 大   | 测试平台体验提升 |
| 📌      | 加批量对比接口                                     | 大   | 实验效率提升     |

------

## 技术栈

- **后端**：FastAPI / Flask
- **Agent 编排**：LangGraph / LangChain
- **向量库**：Chroma（主）/ Milvus（部分项目）/ Redis
- **关系存储**：MySQL / SQLite
- **文档存储**：MongoDB（部分项目）
- **Embedding**：Ollama bge-m3 / DashScope text-embedding-v3
- **Rerank**：DashScope gte-rerank-v2 / qwen3-rerank
- **LLM**：通义千问 / DeepSeek
- **多模态**：qwen-vl-max
- **文档解析**：PyMuPDF / mammoth / MinerU
- **容器化**：Docker
- **测试**：pytest

------

## 贡献指南

本项目为个人项目，欢迎提 Issue 和 PR。

**添加新插件前请先阅读"扩展指南"章节**，了解需要改动的位置。

**开发规范**：

- 新增检索方法：遵守 `pre_retrieval.py` 里的方法签名约定
- 新增切割策略：遵守 `TextTanckle` 类的方法签名
- 提交前跑一遍 pytest

------

## License

MIT License

------

**作者**：李龙
**邮箱**：3490765269@qq.com

如有问题或建议，欢迎通过邮箱联系。