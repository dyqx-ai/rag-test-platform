好，这次重写，把聊天里的细节全带上，流程图画详细，问题风险单列。篇幅会很长，但这就是你要的完整版。

---

# RAG 策略测试平台 · 完整代码梳理与架构文档

---

## 〇、项目定位（一句话）

**这是一台 RAG 策略实验台，不是问答产品。**

用户拿自己的文档当试验田，横向试各种检索方式、压缩方式、重排方式，对比结果，最后沉淀出适合自己场景的一套组合。前端那些 checkbox 和下拉框是**主操作台**，AI 自动决策只是一个可选的基线对照。

---

## 一、五大插拔层

这是项目的灵魂。每一层都是「统一接口 + 可替换插件」，层与层之间独立，用户自由组合。

```
┌─────────────────────────────────────────────────────────────┐
│  插槽 1  模型层      config 驱动      （全局，重启生效）        │
│  插槽 2  入库类型    建库时定一次      （决定知识库"长什么样"） │
│  插槽 3  预检索方法  每次提问多选      （决定"怎么召回"）      │
│  插槽 4  重排        每次提问可开关    （决定"多路怎么合并"）   │
│  插槽 5  压缩        每次提问可换      （决定"怎么瘦身"）      │
└─────────────────────────────────────────────────────────────┘
```

### 插槽 1：模型层 🧠

| 模型类型  | 可选实现                                  | 切换位置                     | 备注                                      |
| --------- | ----------------------------------------- | ---------------------------- | ----------------------------------------- |
| 对话模型  | qwen / deepseek                           | `config.llm.use_model`       | 都走 OpenAI 兼容协议                      |
| Embedding | Ollama 本地 / DashScope 云端              | `config.embedding.use_local` | 本地走 11434 端口                         |
| Rerank    | DashScope（gte-rerank-v2 / qwen3-rerank） | `config.rerank.cloud.model`  | 两种请求体格式，`is_compatible_mode` 切换 |
| 多模态    | qwen-vl-max                               | `config.multimodal_model`    | 初始化失败降级为纯文本                    |

**DIY 新模型的方式**：`models.py` 里加 `elif` 分支 → config 加配置段 → `.env` 加 key。

**目前不"插拔"的地方**：`init_chat_model` 是 if/elif 硬编码，加多了会变长。理想是注册表化（类似 `retriever_method_map`）。

### 插槽 2：入库类型 📦

**注册表**：`EmbeddingVectorManager.UPLOAD_TYPES`

| 入库类型           | 建什么索引                     | 支持哪些检索              | 跳过基础入库？ |
| ------------------ | ------------------------------ | ------------------------- | -------------- |
| `normal`           | 基础向量                       | normal                    | 否             |
| `parent_child`     | 父子双索引                     | parent_child / normal     | **是**         |
| `preview_question` | 预览问题向量 + 原文 byte_store | preview_question / normal | **是**         |
| `summary`          | 摘要向量 + 原文 byte_store     | summary / normal          | **是**         |
| `enrich`           | 字段模板向量                   | enrich / normal           | **是**         |
| `hybrid_search`    | 基础向量 + BM25 索引           | hybrid_search / normal    | 否             |

**为什么某些类型要跳过基础入库？**
因为这几种走的是自己的存储结构（byte_store / docstore），如果先做一遍基础向量入库，集合就"已存在"了，后续索引构建会被 `_collection_exists` 拦掉，导致整个流程失效。所以 `need_skip_base` 是必须的。

### 插槽 3：预检索方法 🔍

**两个注册表**（这是关键设计）：

```python
# A 类：返回 retriever 对象，需要再 .invoke(query)
retriever_method_map = {
    "parent_child":     {"func": ..., "text_param": "parent_child_text_list"},
    "self_query":       {"func": ..., "text_param": "self_query_text_list"},
    "preview_question": {"func": ..., "text_param": "preview_question_text_list"},
    "summary":          {"func": ..., "text_param": "need_summary_text_list"},
    "hybrid_search":    {"func": ..., "text_param": "hybrid_search_text_list"},
}

# B 类：直接返回 List[Document]
direct_doc_method_map = {
    "multi_query":   {"func": ..., "text_param": "multi_query_text_list"},
    "decomposition": {"func": ..., "text_param": "decomposition_text_list"},
}

# 特殊公民
ENRICH_NAME = "enrich"   # 不在任何注册表里，含 input() 交互
```

**9 种方法一览**：

| 方法               | 返回类型                  | 是否需要 LLM | 核心机制                                    |
| ------------------ | ------------------------- | ------------ | ------------------------------------------- |
| `normal`           | 走兜底路径                | 否           | 基础单路向量检索                            |
| `multi_query`      | `List[Document]`          | 是           | LLM 生成 N 个同义查询 → 分别检索 → 内容去重 |
| `decomposition`    | `List[Document]`          | 是           | LLM 拆 N 个子问题 → 分别检索 → 内容去重     |
| `parent_child`     | `ParentDocumentRetriever` | 否           | 子块进向量库，父块进 docstore               |
| `self_query`       | `SelfQueryRetriever`      | 是           | LLM 生成元数据字段定义代码 → exec 执行      |
| `preview_question` | `MultiVectorRetriever`    | 是           | 生成预览问题向量，原文存 byte_store         |
| `summary`          | `MultiVectorRetriever`    | 是           | 摘要向量化，原文存 byte_store               |
| `enrich`           | `Dict`                    | 是           | 模板匹配 + 多轮 input() 补全                |
| `hybrid_search`    | `EnsembleRetriever`       | 否           | BM25 + 向量加权融合                         |

**决策优先级**：用户选择 > LLM 决策 > 默认 `multi_query`

### 插槽 4：重排 🎛️

| 插件     | 开关                  | 实现               | 只在多路时有效？ |
| -------- | --------------------- | ------------------ | ---------------- |
| RRF 融合 | `enable_rrf`          | `_rrf_fusion_sort` | ✅ 单路开不开一样 |
| 模型重排 | `enable_model_rerank` | `_ali_rerank_sort` | ❌ 单路也能用     |

**RRF 公式**：`score(doc) = Σ 1 / (rank + k)`，k 默认 60（config 里的 `rrf_smooth_k`）

**为什么要 RRF？** 不同检索器的分数尺度不一样，不能直接比较。RRF 只看排名，不看分数，天然融合多路。

### 插槽 5：压缩 🗜️

| 压缩类型     | 实现类                       | 行为                                              |
| ------------ | ---------------------------- | ------------------------------------------------- |
| `extract`    | `LLMChainExtractor`          | LLM 从每篇抽取相关片段                            |
| `llm_filter` | `LLMChainFilter`             | LLM 判断每篇留或丢（文档级）                      |
| `emb_filter` | `EmbeddingsFilter`           | 向量相似度过滤，不调 LLM                          |
| `pipeline`   | `DocumentCompressorPipeline` | 四连：splitter → 冗余过滤 → 相似度过滤 → LLM 过滤 |

**总开关**：`config.rag.enable_context_compress`，关掉后所有压缩跳过。

---

## 二、完整数据链路（超详细流程图）

### 2.1 整体架构图

```
                                    ┌─────────────────────────────────┐
                                    │  前端 muti_function.html         │
                                    │  ┌───────────────────────────┐  │
                                    │  │ 左侧：对话列表 / 知识库面板 │  │
                                    │  │ 主区：消息流 / 输入框       │  │
                                    │  │ 高级参数面板：              │  │
                                    │  │   • 9 个预检索 checkbox    │  │
                                    │  │   • 压缩类型下拉           │  │
                                    │  │   • RRF 开关              │  │
                                    │  │   • 模型重排开关          │  │
                                    │  └───────────────────────────┘  │
                                    └──────────────┬──────────────────┘
                                                   │ HTTP
                                    ┌──────────────▼──────────────────┐
                                    │  main_fastapi.py                 │
                                    │  ┌───────────────────────────┐  │
                                    │  │ 启动自检：                 │  │
                                    │  │   配置文件 → 目录 → Redis  │  │
                                    │  │   → MySQL                 │  │
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
                    （见 2.2 入库链路）                          （见 2.3 问答链路）
```

### 2.2 文件入库链路（超详细）

```
【用户上传文件】
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ main_fastapi.api_upload_file                                        │
│  ① 校验 collection_name 非空                                         │
│  ② 校验文件名后缀在 ALLOWED_EXTENSIONS 里                             │
│  ③ 校验 upload_type 合法（6 选 1）                                    │
│  ④ 保存文件到 UPLOAD_FOLDER（重名自动加 _1 _2 后缀）                   │
│  ⑤ 调 embedding_manager.file_to_embedding                            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ EmbeddingVectorManager.file_to_embedding                            │
│  Step 1  集合名规范化                                                │
│          _normalize_collection_name(collection_name)                │
│          ├── 查 MySQL：中文名已有映射？ → 直接用现成英文集合名         │
│          ├── 没命中 → 拼音转换 + 清洗（只留 a-z0-9_-，长度 3-63）      │
│          ├── 本身是合法英文名 → 直接小写使用                          │
│          └── 转换后写回 MySQL：add_kb_mapping                        │
│                                                                     │
│  Step 2  文件遍历                                                    │
│          ├── 单文件 → 校验后缀 → 加入 valid_file_paths                │
│          └── 目录 → 遍历子文件 → 过滤后缀 → 加入 valid_file_paths      │
│                                                                     │
│  Step 3  批量解析                                                    │
│          data_wash(valid_file_paths, image_save_path, ...)          │
│                    .multi_data_wash()                               │
│          （详见 2.2.1 解析子流程）                                    │
│                                                                     │
│  Step 4  分片                                                        │
│          TextTanckle.text_batch_tanckle(                            │
│              method=split_method,     ← None 走通用                  │
│              text_list=raw_text_list,                               │
│              text_source_list=source_path_list,                     │
│              is_markdown=True                                       │
│          )                                                          │
│          （详见 2.2.2 分片子流程）                                    │
│                                                                     │
│  Step 5  过滤空内容 Document                                         │
│                                                                     │
│  Step 6  判断 need_skip_base                                        │
│          ├── parent_child / preview_question / summary / enrich     │
│          │     → 跳过基础向量入库                                     │
│          └── normal / hybrid_search                                 │
│                → embedding_manager_choice(docs, real_collection)    │
│                  （详见 2.2.3 向量化子流程）                          │
│                                                                     │
│  Step 7  写后验证                                                    │
│          重新打开 Chroma，count() 一下，确认真的写进去了              │
│                                                                     │
│  Step 8  非 normal 时构建进阶索引                                    │
│          pre_retrieval.xxx(docs, collection_name)                   │
│          （详见 2.2.4 进阶索引子流程）                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ main_fastapi 继续：                                                  │
│  ⑥ mysql_manager.add_kb_mapping(display_name, real_collection, ...) │
│     display_name = f"{中文名}{upload_type}"                          │
│     support_methods 从 embedding_manager.get_support_methods 拿      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 2.2.1 解析子流程（data_wash.multi_data_wash）

```
对每个文件按后缀分发：

  ┌─ .txt ──→ text_wash()            UTF-8 读，失败 GBK 兜底
  │
  ├─ .md ───→ md_wash()              直接复用 text_wash
  │
  ├─ .docx ─→ docx_wash_method()
  │            ├── docx_wash()        mammoth 原生解析
  │            │     ├── 图片回调 save_image() → img_n.ext 落盘
  │            │     └── md 图片语法 ![img_n](path) → [图片:img_n] 占位
  │            └── 失败降级 → mineru_wash()
  │
  └─ .pdf ──→ pdf_wash_method()
               ├── pdf_wash()         PyMuPDF 原生解析
               │     ├── blocks 按 y、x 坐标排序
               │     ├── find_tables() 抽表格 → md 表格语法
               │     ├── get_images() 抽图片 → img_n.png 落盘
               │     │     └── 采样像素判断是否空白图 → 过滤
               │     ├── 按 y 坐标把 图/表/文 混排
               │     └── 图片也埋 [图片:img_n] 占位
               └── 失败降级 → mineru_wash()

mineru_wash（兜底）：
  ├── 校验 MinerU exe 存在
  ├── 校验文件后缀 .pdf / .docx
  ├── subprocess.run([mineru_exe, "-p", file, "-o", out_dir, ...])
  ├── 解码 stdout/stderr（utf-8 失败 → gbk）
  ├── 找 out_dir/{file_stem}/auto/ 下的 md 文件
  │     └── 优先取带 "_content" 的
  ├── 读 md 内容
  └── 扫描同级 images/ 目录，收集图片路径

产出 dict：
  {
    "文件名": {
      "markdown_text": "...",
      "image_paths": [...],
      "mineru_md_file": "路径 或 空字符串",
      "original_file_path": "..."
    }
  }
```

#### 2.2.2 分片子流程（TextTanckle）

```
TextTanckle.text_batch_tanckle(method, text_list, text_source_list, is_markdown)

  对每条文本：
    method(raw_text, is_markdown, text_source)
    │
    ├── split_text_to_document（默认 / 通用）
    │     ├── 入参校验（非空字符串）
    │     ├── _clean_text(raw_text, is_markdown)
    │     │     ├── 去控制字符 \x00-\x1F\x7F
    │     │     ├── 全角空格 → 半角
    │     │     ├── 连续空白 → 单空格
    │     │     ├── 连续空行 → \n\n
    │     │     └── md 额外：去 <!-- page n -->、去超链接保留文字、去 [1] 引用标记
    │     ├── 全文 sha256 = doc_full_hash
    │     ├── RecursiveCharacterTextSplitter 切分
    │     │     separators=["\n## ", "\n### ", "\n\n", "\n", " "]
    │     ├── 对每个 chunk：
    │     │     ├── 过滤空 / <10 字符碎片
    │     │     ├── _extract_img_list(chunk)
    │     │     │     ├── sha256 命名图片
    │     │     │     ├── img_n.ext 格式
    │     │     │     └── [图片:img_n] 占位符
    │     │     ├── chunk sha256 = chunk_id
    │     │     └── 封装 Document(metadata={
    │     │           doc_full_hash, chunk_id, img_id_list, source
    │     │         })
    │     └── 返回 List[Document]
    │
    ├── special_text_medical（药品）
    │     ├── 提取 核准日期 / 修改日期 / 警告段落
    │     ├── 按 【章节名】拆正文
    │     ├── 两层切分：父块 → 子块
    │     └── metadata 额外含：father_id / grandfather_id / mark
    │
    └── special_text_device（器械）
          ├── 提取 设备型号 / 修订日期 / 禁忌
          ├── 按 1、2、3 拆操作步骤
          ├── 拆故障与解决方案
          └── metadata 含：mark（类别标记）

来源列表对齐：
  长度不足 → 补 "未知来源"
  长度超出 → 截断
  单条失败 → 打印警告继续
```

#### 2.2.3 向量化子流程（embedding_manager_choice）

```
embedding_manager_choice(documents, collection_name)
  │
  ├── 边界防护：metadata 里的空列表 → 转空字符串
  │     （Chroma 不允许 metadata 存空列表）
  │
  ├── 判断 use_local：
  │     ├── True  → text_to_embedding_ollama
  │     └── False → text_to_embedding_dashscope
  │
  └── 两者内部代码完全一样：
        ├── model_manager.init_embedding_model()
        ├── vector_factory.init_vector_store(collection_name, model)
        └── vector_store.add_documents(documents)
              ⚠️ 没传 ids → Chroma 生成随机 UUID
                   → 重复入库会产生重复 chunk
```

#### 2.2.4 进阶索引子流程（按 upload_type）

```
parent_child:
  ├── 父 splitter（chunk_size=1000, overlap=100）
  ├── 子 splitter（chunk_size=200, overlap=20）
  ├── docstore = RedisStore 或 InMemoryStore
  ├── ParentDocumentRetriever(vectorstore, docstore, ...)
  └── 集合不存在才 add_documents

preview_question:
  ├── LLM 批量生成预览问题（N 条，逗号分隔）
  ├── MultiVectorRetriever(vectorstore, byte_store, id_key="connection_id")
  ├── 每个原文分配 UUID
  ├── 生成的问题向量带 connection_id 进向量库
  └── 原文进 byte_store（key=UUID）

summary:
  ├── LLM 批量生成摘要
  ├── MultiVectorRetriever(...)
  ├── 摘要向量带 connection_id 进向量库
  └── 原文进 byte_store

enrich_index_build:
  ├── LLM 从每篇文档抽字段模板（每行一个）
  ├── 过滤长度 >5 的行
  ├── 抽不到 → 用默认 {"姓名，联系方式，时间，地点"}
  └── 模板进向量库（集合不存在才建）

hybrid_search:
  ├── 基础向量入库（不跳过）
  ├── BM25Retriever.from_documents(docs)
  └── EnsembleRetriever([bm25, vector], weights=[...])
        ⚠️ BM25 是内存索引，不落盘
            每次调用都要从向量库捞文档重建
```

### 2.3 问答链路（超详细）

```
【用户提问 + 勾选参数】
      │
      ▼
┌──────────────────────────────────────────────────────────────────────┐
│ main_fastapi.api_rag_chat                                            │
│  ① 校验 user_query / collection_name 非空                            │
│  ② embedding_manager._normalize_collection_name（中文→拼音）          │
│  ③ 解析 pre_retrieval_methods（逗号分隔 → list）                      │
│  ④ 调 rag_service.run_rag_task                                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ AutoRagAutoDispatchService.run_rag_task                              │
│  ① 空问题拦截 → "请输入有效的提问内容"                                │
│  ② 默认值覆盖：collection_name / persist_time                        │
│  ③ user_pre_methods 清洗（去空、去非字符串）                          │
│  ④ vector_text_list → List[Document]                                │
│  ⑤ 调 run_full_auto_rag                                             │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ run_full_auto_rag  五步主流程                                         │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Step 1  决策                                                    │  │
│  │   decision_engine.decide_strategy(user_query, user_pre_methods) │  │
│  │   （详见 2.3.1 决策五分支）                                       │  │
│  │   → {use_pre_retrieval, pre_methods}                            │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                               │                                       │
│  ┌────────────────────────────▼───────────────────────────────────┐  │
│  │ Step 2  检索                                                    │  │
│  │   try:                                                          │  │
│  │     retrieval_dispatcher.retrieve(...)                          │  │
│  │   except RuntimeError err:                                      │  │
│  │     if "COLLECTION_NOT_EXIST_NO_INIT_DOC" in str(err):          │  │
│  │        → 兜底答案 "暂时没办法回答你的问题" + 存历史 + return     │  │
│  │     else:                                                       │  │
│  │        retrieve_groups = []                                     │  │
│  │   except Exception:                                             │  │
│  │     retrieve_groups = []                                        │  │
│  │   （详见 2.3.2 检索子流程）                                       │  │
│  │   → List[List[Document]]                                        │  │
│  │                                                                 │  │
│  │   空结果兜底：                                                   │  │
│  │     检索组数=0 或 全空 → 兜底答案 + 存历史 + return              │  │
│  └────────────────────────────┬───────────────────────────────────┘  │
│                               │                                       │
│  ┌────────────────────────────▼───────────────────────────────────┐  │
│  │ Step 3  后置处理                                                 │  │
│  │   post_processor.process_context(                               │  │
│  │       query, retrieve_groups, custom_compress_type,             │  │
│  │       enable_rrf, enable_model_rerank)                          │  │
│  │   （详见 2.3.3 后置处理子流程）                                   │  │
│  │   → final_context（纯文本）                                       │  │
│  │                                                                 │  │
│  │   空结果兜底：final_context 空 → 兜底答案 + 存历史 + return       │  │
│  └────────────────────────────┬───────────────────────────────────┘  │
│                               │                                       │
│  ┌────────────────────────────▼───────────────────────────────────┐  │
│  │ Step 4  生成                                                     │  │
│  │   _build_generate_prompt(retrieve_groups, user_query,            │  │
│  │                          final_context)                         │  │
│  │   （详见 2.3.4 prompt 组装）                                      │  │
│  │   → str 或 messages 数组                                        │  │
│  │                                                                 │  │
│  │   if isinstance(prompt_data, str):                              │  │
│  │      llm_resp = generate_llm_text.invoke(prompt_data)           │  │
│  │   elif multimodal_llm is not None:                              │  │
│  │      llm_resp = multimodal_llm.invoke(prompt_data)              │  │
│  │   else:                                                         │  │
│  │      # 多模态不可用 → 从 messages 抠 text 部分 → 纯文本模型      │  │
│  │                                                                 │  │
│  │   final_answer = (llm_resp.content or "").strip()               │  │
│  │   if 空：final_answer = "检索资料：... + 模型返回内容为空"        │  │
│  │   except Exception:                                             │  │
│  │      final_answer = "检索资料：... + 模型生成失败"               │  │
│  └────────────────────────────┬───────────────────────────────────┘  │
│                               │                                       │
│  ┌────────────────────────────▼───────────────────────────────────┐  │
│  │ Step 5  持久化                                                   │  │
│  │   chat_saver.save_chat(user_query, final_answer,                 │  │
│  │                        session_id, persist_time)                │  │
│  │   （详见 2.4 持久化链路）                                          │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ main_fastapi 继续：                                                   │
│  ⑤ 如果 session_id 存在：                                             │
│     mysql_manager.add_message(session_id, "user", user_query)        │
│     mysql_manager.add_message(session_id, "assistant", answer)       │
│  ⑥ 如果会话消息数 == 2（第一轮）：                                     │
│     mysql_manager.update_conversation_title(session_id, query[:20])  │
│  ⑦ 返回 success_response(data={"answer": answer})                     │
└──────────────────────────────────────────────────────────────────────┘
```

#### 2.3.1 决策五分支（decide_strategy）

```
输入：user_query, user_selected_methods

  ① 用户勾了 normal
     ├── 直接返回 {use_pre_retrieval: False, pre_methods: []}
     └── ⚠️ 不调 LLM，其他勾选全被忽略（互斥）
         │
  ② 用户勾了其他方法（含多个）
     ├── valid_user_methods = 白名单过滤
     ├── 拼 user_hint："用户已手动选择... 请优先采用"
     ├── 动态拼 prompt（system 从原 chain 反解 + human 带 user_hint）
     ├── 调 LLM → _parse_and_verify_decision 校验
     └── 如果 LLM 返回空方法列表 → 降级用用户选择
         │
  ③ 用户什么都没勾
     ├── 纯 LLM 决策（_build_decision_chain 自由判断）
     └── _parse_and_verify_decision 校验
         │
  ④ LLM 调用异常
     ├── 有用户选择 → 用用户选择
     └── 无用户选择 → 默认 {True, ["multi_query"]}
         │
  ⑤ _parse_and_verify_decision 内部分层校验
     ├── 字段缺失 → 补默认
     ├── 方法名白名单过滤 → 剔除非法
     ├── use_pre_retrieval=False → 清空方法列表
     ├── use_pre_retrieval=True 但方法空 → 兜底 ["multi_query"]
     └── JSON 解析失败 → 返回默认 {True, ["multi_query"]}
```

#### 2.3.2 检索子流程（retrieval_dispatcher.retrieve）

```
if use_pre_retrieval:
    dispatch_pre_retrieval(...)
else:
    _get_base_single_retrieve_groups(...)   ← 兜底单路

dispatch_pre_retrieval 流程：
  ① normal_methods = pre_methods 剔除 enrich
  ② _ensure_collection_and_docs(collection_name, init_docs)
       ├── 有 init_docs → vs.add_documents(init_docs) → True
       └── 无 init_docs → 检查集合存在 → True/False
       └── False → raise RuntimeError("COLLECTION_NOT_EXIST_NO_INIT_DOC")
  ③ 遍历 direct_doc_method_map
       for method_name in 注册表:      ← 注意：遍历的是注册表，不是用户选择
         if method_name not in normal_methods: continue
         try:
           kwargs = {query, collection_name, text_param: init_docs}
           docs = func(**kwargs)
           if docs: retrieve_groups.append(docs)
         except: print 警告; continue
  ④ 遍历 retriever_method_map
       for method_name in 注册表:
         if method_name not in normal_methods: continue
         try:
           kwargs = {collection_name, text_param: init_docs}
           retriever = func(**kwargs)
           docs = retriever.invoke(query)   ← 这里才真正检索
           if docs: retrieve_groups.append(docs)
         except: print 警告; continue
  ⑤ 如果 enrich 在 pre_methods 里：打印"自动调度链路暂不执行"
  ⑥ 全部失败兜底：_get_base_single_retrieve_groups

返回：List[List[Document]]（二维数组，每组一路检索结果）
```

#### 2.3.3 后置处理子流程（process_context）

```
process_context(query, retrieve_groups, custom_compress_type,
                enable_rrf, enable_model_rerank)

  ① 压缩类型校验
     target_compress = "pipeline"
     if custom_compress_type 在白名单:
        target_compress = custom_compress_type
  ② doc_rerank_all(query, multi_retrieve_groups=retrieve_groups,
                    enable_rrf, enable_model_rerank)

     ├── 多路：enable_rrf=True → _rrf_fusion_sort
     │       enable_rrf=False → 简单拼接去重（dumps(doc) 做 key）
     ├── 单路：raw_docs 直接用
     ├── step_docs 空 → 返回 ""
     │
     ├── _ali_rerank_sort(query, step_docs, enable_model_rerank)
     │     ├── 关或空 → docs[:top_n]
     │     ├── 有 rerank 方法 → rerank_client.rerank(text_arr, query)
     │     │     ├── 返回 score_arr，按序填回 docs 的 metadata["relevance_score"]
     │     │     ├── 按分数降序排列
     │     │     ├── 阈值过滤（>= rerank_threshold）
     │     │     ├── 过滤后空 → 降级保留全部重排文档
     │     │     └── 截断到 top_n
     │     ├── 有 compress_documents 方法 → 用它
     │     ├── 都没有 → 跳过重排
     │     └── 异常 → 返回全部原始文档
     │
     └── 拼接 page_content → "\n\n".join(...)
  ③ doc_compress_all(query, source_text=sorted_text,
                     compress_type=target_compress)

     ├── config.rag.enable_context_compress=False 或 source_text 空 → 直接返回
     ├── 把整段文本包成一个 Document
     ├── 分发：
     │     ├── extract → LLMChainExtractor
     │     ├── llm_filter → LLMChainFilter
     │     ├── emb_filter → EmbeddingsFilter
     │     ├── pipeline → DocumentCompressorPipeline（四连）
     │     └── 其他 → 返回原文
     ├── 压缩后空 → 降级返回原文
     └── 异常 → 返回原文
  ④ process_context 两级降级
     ├── final_text 空 but sorted_text 非空 → 返回 sorted_text
     └── 都空 → 把 retrieve_groups 原文拼接返回
```

#### 2.3.4 prompt 组装（_build_generate_prompt）

```
_build_generate_prompt(retrieve_groups, user_query, context_text)

  ① 从 retrieve_groups 收集 img_id_list
       ⚠️ 注意：从原始检索结果收集，不是从压缩后的 context_text
  ② 拼 prompt 模板（4 条规则 + context + question）
  ③ 无图片 → 返回 str
  ④ 图片去重 + 截断（max_send_img 默认 3）
  ⑤ img_list_to_base64 读盘转 base64
  ⑥ 过滤掉 None 的 base64
  ⑦ 无有效图片 → 降级返回 str
  ⑧ 组装 messages：
     [
       {"role": "user", "content": [
         {"type": "text", "text": prompt_text},
         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
         ...
       ]}
     ]
```

### 2.4 持久化链路（两条并行）

```
【保存对话】auto_rag_service Step 5
      │
      ▼
┌────────────────────────────────────────────────────────────────────┐
│ RagChatSaver.save_chat(user_query, answer, session_id, persist_time)│
│  ① config.save_method.if_save=False → 直接 return                  │
│  ② session_id 空 → 直接 return                                     │
│  ③ chat_pair = [f"human:{user_query}", f"ai:{answer}"]              │
│  ④ choose_conversation_save_method(...)                            │
│      ├── method="redis"  → SaveChatRedis                           │
│      └── method="sqlite" → SaveChatSqlite                          │
│  ⑤ hasattr 判断 → 调 redis_save 或 sqlite_save                      │
└────────────┬──────────────────────┬─────────────────────────────────┘
             │                      │
             ▼                      ▼
    ┌─────────────────┐   ┌──────────────────┐
    │ SaveChatRedis    │   │ SaveChatSqlite   │
    │ 全局 POOL 连接池  │   │ 每次新建连接      │
    │ lpush + expire   │   │ executemany 批量  │
    │ key=chat_session:│   │ 表 chat_records  │
    │ TTL 默认 60 秒   │   │ ORDER BY id ASC  │
    └─────────────────┘   └──────────────────┘

【同时，main_fastapi 里另一条线】
      │
      ▼
┌────────────────────────────────────────────────────────────────────┐
│ MysqlDataManager.add_message(session_id, role, content) × 2        │
│   rag_conversation_message 表                                       │
│ 如果消息数 == 2 → update_conversation_title                          │
└────────────────────────────────────────────────────────────────────┘

【加载历史】main_fastapi.api_get_conversation_history
      │
      ▼
 ① 先查 Redis（get_langchain_history）
      │
      ├── 有 → 解析 human:/ai: → 转 user/assistant → 返回
      │
      └── 没有
            ↓
 ② 查 MySQL（get_messages_by_conv_id，ORDER BY create_time ASC）
      │
      └── 有 → 回写 Redis（role_map: user→human, assistant→ai）→ 返回
```

---

## 三、逐文件深度梳理（含我们聊过的所有细节）

### 3.1 `config_load.py` —— 配置地基 🧱

**核心目的**：相对路径 → 绝对路径，保证项目可移植性 + 运行目录无关性。

**为什么必须这么做？**
- 可移植：拷到别的机器 / 目录，不用改配置
- 运行目录无关：从任何地方 `python xxx.py`，路径都算得对

**关键代码**：

```python
_CONFIG_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))  # config/
_PROJECT_ROOT = os.path.dirname(_CONFIG_MODULE_DIR)              # myproject/
```

`__file__` 永远是这个文件自己的位置，跟从哪启动无关。所有相对路径都基于 `_PROJECT_ROOT` 拼。

**支持环境变量覆盖** `RAG_CONFIG_PATH`（Docker 部署场景）。

**坑点**：`yaml.safe_load` 遇到空文件返回 `None`，不是空字典，代码里堵住了。

### 3.2 `models.py` —— 模型管理 🧠

**核心思路**：所有模型的统一出口，配置驱动插拔。

**四个方法**：

| 方法                    | 返回                                   | 配置驱动                         |
| ----------------------- | -------------------------------------- | -------------------------------- |
| `init_chat_model`       | ChatOpenAI                             | `llm.use_model`（qwen/deepseek） |
| `init_embedding_model`  | OllamaEmbeddings / DashScopeEmbeddings | `embedding.use_local`            |
| `init_rerank_model`     | DashScopeRerankWrapper                 | `rerank.cloud.model`             |
| `init_multimodal_model` | ChatOpenAI                             | `multimodal_model`               |

**`DashScopeRerankWrapper` 的巧思**：
- `gte-rerank-v2` 和 `qwen3-rerank` 请求体格式不同
- 用 `is_compatible_mode = (model_name == "qwen3-rerank")` 在内部切换
- 对外统一 `.rerank(documents, query) -> List[float]`
- 返回分数与输入顺序一致，让调用方自己排序

**容错设计**：API key 空 / HTTP 失败 / 状态码非 200 → 返回全 1.0 分数（不排序，不影响链路）

**DIY 新模型**：
- chat：加 `elif` 分支 + config + .env
- embedding：加分支
- rerank：本地 bge-reranker 还没实现，DIY 空间最大

**当前约束**：`init_chat_model` 是硬编码 if/elif，加多了会变长。理想是注册表化。

**已知问题**：`load_dotenv(Path(get_project_root()) / ".env")` 只在模块导入时跑一次，中途改环境变量不感知。

### 3.3 `mineru_wash.py` —— MinerU 兜底 🛟

**定位**：MinerU CLI 的薄封装，只在原生解析失败时被调用。

**为什么是兜底？** MinerU 很重（模型做版面分析、公式识别、表格识别），在 CPU 上跑一个 PDF 能等半天。个人电脑上用着累。

**关键设计**：
- 用 `file_p.stem` 而不是 `split(".")` —— 兼容 `v1.2说明书.pdf` 这种带点文件名
- 输出目录：`out_dir/{file_stem}/auto/`
- 优先取带 `_content` 的 md 文件
- 编码兜底：utf-8 失败 → gbk
- 异常全吞，返回 `(None, None)`
- **不改变工作目录**（注释里专门说明，避免影响其他模块）

### 3.4 `muti_catolog_file_wash.py` —— 解析调度 🚗

**核心思路**：**原生解析优先 + MinerU 兜底**，双轨策略。

**双轨的优势**：
- 大部分情况走快路（PyMuPDF / mammoth），速度快
- 只有遇到原生搞不定的（复杂排版、扫描件）才降级到 MinerU
- 个人电脑上用着不累

**图片处理的关键设计**：
- 图片落盘（`img_n.png`）
- 文本流里埋 `[图片:img_n]` 占位符
- **不同解析路径统一成同一个占位符格式**，下游 `TextTanckle._extract_img_list` 只需认一种

**PDF 的坐标混排**（精华）：
- 按 y 坐标把文字块 / 图片 / 表格混排
- 逻辑：遍历每个文字块，看当前块上方有没有未放置的图/表，先插它们再放文字
- 用 `insert_imgs` / `remain_imgs` 模式（遍历一次分拣到两个桶），避免边遍历边删除
- 空白图过滤：采样一个像素看是不是全白

**一个隐患**：图片名 `img_n.png` 是全局编号，多文件解析时**可能互相覆盖**（个人测试问题不大）。

### 3.5 `data_wash.py` (TextTanckle) —— 文本切割 ✂️

**核心价值**：文本进向量库前的最后一站。

**三种切割策略同签名**：
```python
(text, is_markdown, text_source) -> Optional[List[Document]]
```

- `split_text_to_document` —— 通用
- `special_text_medical` —— 药品说明书（父子两层 + mark 标记）
- `special_text_device` —— 器械说明书（按类别拆）

**扩展方式**：类里加方法，通过 `text_batch_tanckle(method=...)` 或 `file_to_embedding(split_method=...)` 传入。**不改注册表、不改白名单、不改前端**。

**图片 ID 提取的三种格式**：
- sha256 命名
- `img_n.ext`
- `[图片:img_n]` 占位符

**hash 生成**：
- `doc_full_hash` —— 整篇 sha256
- `chunk_id` —— 分片 sha256
- **但都没用来去重**，只是塞进了 metadata

**清洗顺序很关键**：先压空白 → 再恢复段落结构（`clean_multi_space` 会把连续换行压成空格，必须先压空白再恢复段落）。

**空列表默认参数的坑**：用 `None + 判空转 []` 规避 Python 可变默认参数共享问题。

**短碎片过滤**：`< 10 字符` 直接扔，避免页眉页脚污染向量库。

**注意导入路径**：`from myproject.config.config_load import config_load`，其他文件是 `from config.config_load`，**可能运行时报错**。

### 3.6 `vectorydb_factory.py` —— 向量库工厂 🏭

**核心思路**：工厂模式 + 依赖注入。

**`set_chroma`**：
- **embedding 模型必须外部注入**（不是工厂自己 new）
- 相对路径 → 绝对路径（跟 config_load 一致）
- 持久化开关：`persist_save=True` 传 `persist_directory`，否则纯内存

**三个 `NotImplementedError`**：Milvus / Faiss / Redis 都留了接口，实现慢慢补。

**`init_vector_store` 兜底**：不传 model 也能跑，但**打印警告**（灰度降级 + 提示）。

**`get_chroma_client` 借道 `meta_temp`**：管理操作（列集合、删集合）用一个临时集合拿客户端。集合名不能以 `_` 开头（Chroma 校验规则），所以叫 `meta_temp`。

**小浪费**：`get_chroma_client` 里又 new 了一个 `ModelManager`，因为 Chroma API 要求必须传 `embedding_function`。

### 3.7 `text_embedding.py` —— 入库调度 🧮

**两个并列入口，不是父子关系**：
- `file_to_embedding` —— 文件进来 → 解析 → 分片 → 入库
- `simple_text_list_insert` —— 纯文本进来 → 包 Document → 入库

**共享地基**：`_normalize_collection_name` / `embedding_manager_choice` / `UPLOAD_TYPES` 等。

**`_normalize_collection_name` 的三步走**：
1. 先查 MySQL（有缓存效果）
2. 拼音转换 + 清洗（本类合法直接用，否则转拼音）
3. 写回 MySQL 映射

**为什么用 MySQL 持久化映射？** 拼音算法可能变、多音字可能有歧义，把"用户看到的名字"和"底层集合名"绑定到表里，**稳定**。

**`need_skip_base` 的逻辑**：
- `parent_child` / `preview_question` / `summary` / `enrich` → 跳过基础向量入库
- 因为它们的集合是"专属结构"，先做基础入库会污染集合，导致后续索引构建被 `_collection_exists` 拦掉

**写后验证**：入库后重新打开 Chroma，`count()` 一下确认真的写进去了。**防御性编程**，踩过坑才会加。

**已知问题**：
- `text_to_embedding_ollama` / `text_to_embedding_dashscope` 代码完全一样，可以合并
- `need_skip_base` 在两个入口里各写一遍，可以抽函数
- `chunk_id` 没用来做去重
- `_validate_collection_name` 定义了但没调用（死代码）

### 3.8 `pre_retrieval.py` —— 9 种检索插件 🧰

**这是项目最重的文件。**

**9 个方法的共性**：
1. "集合已存在就跳过入库"（避免重复污染）
2. "不支持就降级"
3. 返回类型不统一（retriever / docs / dict）
4. 多用 UUID 做 `connection_id`（比位置索引稳定）

**逐个分析**：

**`summary` / `preview_question_index`** 结构一样：
- LLM 生成内容（摘要 / 预览问题）
- 生成内容进向量库，原文进 byte_store
- 靠 `connection_id` 关联
- **依赖 RedisStore 或 InMemoryByteStore**
- ⚠️ **如果 Redis 关了，用 InMemoryByteStore，进程重启就丢**，检索失效

**`parent_child`**：
- 父块 1000 字进 docstore
- 子块 200 字进向量库
- 检索时子块匹配 → 返回父块

**`self_query`**：**最野**
- LLM 生成 `metadata_field_info = [AttributeInfo(...), ...]` 代码
- `exec(raw_code, globals(), local_vars)` 执行
- 拿 `metadata_field_info` 喂给 `SelfQueryRetriever.from_llm`
- ⚠️ `exec` 执行 LLM 代码，理论有风险（prompt 约束严格，实践可控）

**`enrich`**：**最特殊**
- 含 `input()` 控制台交互
- 自动链路跳过，只打印提示
- ⚠️ **用户勾了没反应**，是一个"半接通"的功能

**`enrich_index_build`**：给 `enrich` 配套的索引构建

**`multi_query` / `decomposition`**：结构几乎一样
- `multi_query` 生成**同义**查询
- `decomposition` 生成**子**问题
- 都返回 `List[Document]`
- 去重 key 用 `page_content`（Document 不可哈希）

**`hybrid_search`**：
- BM25 + 向量加权融合
- ⚠️ **BM25 每次调用都要从向量库重建**（内存索引不落盘）
- 大库慢，小库没问题

**`_collection_exists` 的粒度问题**（重点）：
- 只认集合名，不认内容
- ⚠️ **同名不同版本会误判**：用户传 "合同v2.docx" 到已存在的「合同库」，会被跳过，更新不生效
- 用户视角：提示"上传成功"，实际什么都没变

### 3.9 `rag_retrieval_dispatcher.py` —— 检索分发 🚦

**两个注册表**（A 类 / B 类），**参数名也配置化**（`text_param` 字段）——分发器不用写一堆 if/else。

**`dispatch_pre_retrieval` 的关键点**：
- **遍历的是注册表，不是用户选择列表** —— 保证结果顺序稳定
- 每个方法独立 try/except，失败不中断（多路容错）
- 全失败兜底到单路向量检索
- **`enrich` 显式跳过**，只打印提示

**异常用字符串约定**：
```python
raise RuntimeError("COLLECTION_NOT_EXIST_NO_INIT_DOC")
```
上层靠 `in` 判断。**简便，但隐晦**。

**单路包装成二维**：`[single_docs]`，跟多路格式一致，后置处理不用分情况。

### 3.10 `rag_decision_engine.py` —— 决策引擎 🎯

**核心优先级**：**用户选择 > LLM 决策 > 默认 `multi_query`**

**五条路径**：
1. 用户勾 `normal` → 直接返回，不调 LLM
2. 用户勾了其他 → 拼 `user_hint` 注入 prompt，让 LLM 参考
3. 用户没勾 → 纯 LLM 决策
4. LLM 结果空但有用户选择 → 用用户选择
5. LLM 异常 → 有用户选择用用户选择，无用户选择用默认

**关键设计思路**：
- 用户选择**不是直接覆盖**，而是作为"强烈建议"注入 prompt
- LLM 可能**补充**（用户选了 multi_query，LLM 觉得还需要 hybrid_search）
- 也可能**否决**（虽然 prompt 说"如无充分理由不要否决"）
- 代价是**多一次 LLM 调用**

**`_parse_and_verify_decision` 三层校验**：
- 结构校验（缺字段补默认）
- 白名单过滤（非法方法名剔除）
- 一致性修正（不用预检索就清空方法列表；说要用但没给方法就兜底）

**异常降级**：JSON 解析失败 → 返回 `{True, ["multi_query"]}`。**"宁可多查点"比"少查点"安全**。

**⚠️ 一个边界**：用户勾了 `normal + multi_query`，`normal` 优先，其他全被忽略。**用户可能不知道这个行为**。

### 3.11 `rag_post_processor.py` + `post_retrieval.py` —— 后置处理 🎛️

**壳 + 核**结构：
- `rag_post_processor` —— 薄封装，校验 + 两级降级
- `post_retrieval` —— 真正的实现，重排 + 压缩

**两级降级**（壳层）：
- 压缩空 → 用重排后文本
- 都空 → 用原始 `retrieve_groups`

**重排两个插件**：

**`_rrf_fusion_sort`**：
- 用 `dumps(doc)` 序列化做 key（Document 不可哈希）
- 公式：`score(doc) = Σ 1 / (rank + k)`

**`_ali_rerank_sort` 六层逻辑**：
1. 关或空 → 直接截断
2. 有 `rerank` 方法 → 用它打分排序
3. 有 `compress_documents` 方法 → 用它
4. 都没有 → 跳过重排
5. 重排后空 → 返回全部原始文档
6. 异常 → 返回全部原始文档

**分数写进 metadata**：`doc.metadata["relevance_score"] = float(score)` —— 不丢弃，方便追溯。

**压缩四个插件**：
- `extract` —— LLM 抽片段
- `llm_filter` —— LLM 判断留/丢
- `emb_filter` —— 向量相似度过滤
- `pipeline` —— 四连击

**⚠️ 压缩的"多文档 → 一段文本"降维**：
- `doc_compress_all` 拿到的是**拼接后的单段文本**，不是文档列表
- 文档级过滤（`llm_filter`）只有 1 篇可判断
- 文章级抽取（`extract`）对整段抽
- **语义上不完美**

**孤儿方法**：
- `run_full_pipeline` —— 主链路不走它
- `multi_query_retrieve` —— 只被 `run_full_pipeline` 调用

**死配置**：`self.enable_rerank` 读了 `config.rerank.enable`，但代码里没用。

### 3.12 `auto_rag_service.py` —— 主编排 🎯

**五步流程**：决策 → 检索 → 后置 → 生成 → 保存

**每步都有兜底**：
- 决策失败 → 决策引擎内部降级
- 检索失败 → 返回兜底答案
- 后置为空 → 返回兜底答案
- LLM 失败 → 返回资料 + 失败提示
- LLM 空内容 → 返回资料 + 提示

**`_build_generate_prompt` 的关键不一致**：
- 图片从**原始 `retrieve_groups`** 提取
- 文本是**压缩后的 `final_context`**
- 压缩丢文档时，图片没跟着丢 → **可能对不上**

**`data:image/jpeg` 前缀固定**：不管实际格式，一律标 jpeg。DashScope 会兜住，但不严谨。

**大量 print 调试**：每步都有，甚至有一段专门查 Chroma 集合列表的调试代码（每次问答都跑）。

**参数名转换**：对外 `user_question`，对内 `user_query`。

**死代码**：`self.output_parser` 定义了但没用。

### 3.13 持久化层 💾

**两条线并行**：

**线 1：Redis / SQLite**（存对话消息）
- `rag_chat_saver` —— 封装 + 开关
- `choose_conversation_save_method` —— 工厂
- `SaveChatRedis` —— 全局 POOL 连接池，`lpush` + `expire`
- `SaveChatSqlite` —— 每次新建连接，`executemany` 批量

**线 2：MySQL**（知识库映射 + 会话 + 消息）
- 三张表：`rag_kb_mapping` / `rag_conversation` / `rag_conversation_message`
- `ping(reconnect=True)` 断线重连
- `ON DUPLICATE KEY UPDATE` UPSERT（软删除可逆）
- `_migrate_support_methods_column` 简化版 schema migration
- 软删除（`is_deleted=1`）

**加载历史逻辑**：Redis 优先，MySQL 兜底，读到 MySQL 后回写 Redis（缓存模式）。

**⚠️ Redis 消息顺序问题**：
- `lpush`（头插）+ `lrange`（正序读）= **整体反序**
- MySQL 是 `ORDER BY create_time ASC` 正序
- **两边不一致**
- 待验证

**`persist_time` 默认 60 秒**：很短。可能是有意（省内存），也可能默认值偏小。

---

## 四、潜在问题与风险清单

### 🔴 严重风险

**R1. 去重缺失**
- hash 生成了（`doc_full_hash` / `chunk_id`）但没用
- 同一文件传两次 → 向量库里有重复 chunk → 检索 top_k 一半是重复内容
- 影响：召回质量下降、答案缺斤少两、RRF 排名失真

**R2. 图片与压缩后文本可能错配**
- `_build_generate_prompt` 图片从原始 `retrieve_groups` 提取，文本是压缩后的 `final_context`
- 压缩丢了某篇文档，图片还在 → 模型看到无关图片 → 可能触发"知识库暂无相关信息"兜底规则
- 影响：图文问答精度明显下降

**R3. `_collection_exists` 粒度太粗**
- 只认集合名，不认内容
- 同名不同版本会误判 → 更新不生效
- 用户不知道被跳过 → 以为上传成功

**R4. Redis 消息顺序可能反了**
- `lpush` 头插 + `lrange` 正序读 = 整体反序
- MySQL 是正序
- 两边不一致，前端渲染可能倒着
- **需实际验证**

### 🟠 中等风险

**R5. `enrich` 前端勾了没反应**
- 不在任何注册表里，分发时跳过
- 只打印一行提示，用户可能困惑
- 含 `input()` 交互，自动链路接不通

**R6. 压缩的"多文档 → 一段文本"降维**
- `doc_compress_all` 拿到的是拼接后的单段文本
- 文档级过滤（`llm_filter`）只有 1 篇可判断
- 语义上不完美

**R7. `run_full_pipeline` / `multi_query_retrieve` 是孤儿**
- 主链路不走
- 跟 `process_context` 功能重叠
- 死代码或备用入口

**R8. `self.enable_rerank` 死配置**
- 读了 `config.rerank.enable` 但没用
- 实际控制重排的是 `enable_model_rerank` 入参
- 两者容易混淆

**R9. `summary` / `preview_question` 依赖 Redis**
- Redis 关了用 `InMemoryByteStore`，进程重启就丢
- 检索失效，用户不知道

**R10. `self_query` 用 `exec` 执行 LLM 代码**
- 理论风险（prompt 约束严格，实践可控）
- 严格来说是个安全缺口

**R11. BM25 每次重建**
- `hybrid_search` 每次调用都从向量库捞文档重建 BM25
- 大库慢
- 小库没问题

### 🟡 轻度问题

**R12. 加插件要动多处**
- 加新检索方法要改 4 处：`pre_retrieval.py` + `retriever_method_map` + `valid_pre_methods` + 前端 checkbox
- 容易漏

**R13. 清洗策略未暴露到前端**
- `special_text_medical` / `special_text_device` 只能代码里手动调
- 上传时没地方选

**R14. 空列表 metadata 处理**
- `img_id_list` 为空时，`embedding_manager_choice` 把它转成空字符串
- 下游 `.get("img_id_list", [])` 拿到的是 `""` 不是 `[]`
- 靠 `isinstance` 判断兼容

**R15. `data_wash.py` 导入路径不一致**
- `from myproject.config.config_load` vs 其他文件的 `from config.config_load`
- 可能运行时报错

**R16. 重复代码**
- `text_to_embedding_ollama` / `text_to_embedding_dashscope` 完全一样
- `need_skip_base` 判断在两个入口里各写一遍
- `_collection_exists` 三处重复实现
- `_validate_collection_name` 死代码
- `output_parser` 死代码

**R17. Redis 全局连接池在 import 时初始化**
- 配置加载失败直接抛异常
- 影响模块导入

**R18. `data:image/jpeg` 前缀固定**
- png 图片也标 jpeg
- DashScope 兜住，但不严谨

**R19. `run_full_auto_rag` 每次问答都查 Chroma 集合列表**
- 调试代码在主流程里
- 浪费性能

**R20. 参数名不一致**
- 对外 `user_question`，对内 `user_query`
- 来回切，读起来别扭

**R21. 图片名跨文件覆盖风险**
- `img_n.png` 全局编号
- 多文件解析可能互相覆盖
- 个人测试问题不大

**R22. `normal` 与其他方法互斥**
- 勾了 `normal`，其他勾选全被忽略
- 用户可能不知道

**R23. `persist_time` 默认 60 秒偏短**
- 用户聊完不到一分钟，Redis 缓存就过期了

**R24. MinerU 慢**
- CPU 上跑一个 PDF 可能几分钟
- 个人电脑上体验差
- 有原生解析优先，大部分不触发

**R25. `prompt` 里说"8 种"，白名单里 9 个**
- 因为 `normal` 不算"预检索"，但为了识别用户意图放进白名单
- 描述不一致

---

## 五、后续可做清单（按 ROI 排序）

| 优先级 | 事项                                               | 投入 | 收益         |
| ------ | -------------------------------------------------- | ---- | ------------ |
| 🥇      | 补 chunk 级去重（`add_documents(ids=[chunk_id])`） | 一行 | 检索结果干净 |
| 🥇      | 修图片与压缩的错配（从压缩后文档提 img_id_list）   | 小   | 图文精度提升 |
| 🥈      | 补文件级去重（MySQL 文件 hash 表）                 | 小   | 省解析时间   |
| 🥈      | 验证 Redis 消息顺序                                | 小   | 修 bug       |
| 🥈      | 接通 `enrich`（前端单独交互入口）                  | 中   | 释放插件价值 |
| 🥉      | 清洗策略暴露到前端                                 | 中   | 扩展场景     |
| 🥉      | 统一加插件的注册点                                 | 中   | 降低扩展成本 |
| 🥉      | 清理死代码 / 重复代码                              | 中   | 可维护性     |
| 🥉      | 把 `models.py` 的 chat 也注册表化                  | 中   | 一致性       |
| 📌      | 把 `print` 换成 logging                            | 中   | 生产可用     |
| 📌      | 拆出 `run_full_auto_rag` 里的集合调试代码          | 小   | 性能         |
| 📌      | 统一 `user_question` / `user_query` 命名           | 小   | 可读性       |

---

## 六、设计哲学总结

1. **插拔式架构** 🔌 —— 每层统一接口，插件独立可换
2. **容错降级** 🛟 —— 每层都有兜底，几乎没有崩溃路径
3. **配置 + 用户双层驱动** ⚙️ —— 全局模型选型走 config，单次实验走前端
4. **中间过程可见** 📋 —— 大量 print，方便测试平台观察
5. **轻编排 + 重分发** 🧩 —— 编排层不实现业务，只把子模块串起来
6. **持久化是前提，不是优化** 💾 —— 向量化一次性落盘，图片落盘靠 ID 关联

---

梳理完毕 🎉 这份文档涵盖了聊天里讨论的所有细节、完整流程图、以及全部问题风险。有要补的随时说 🙌