import base64
import hashlib
import re
from pathlib import Path
from traceback import print_exc
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional, Callable
from myproject.config.config_load import config_load

try:
    config = config_load()
except Exception as e:
    import sys
    print(f"【RAG系统】data_wash 模块配置加载失败: {e}")
    sys.exit(1)

class TextTanckle:
    def __init__(
            self,
            chunk_size: int = config["split"]["chunk_size"],
            chunk_overlap: int = config["split"]["chunk_overlap"],
            img_dir: str = config["path"]["output_img"],
            text_source_list: Optional[List[str]] = None,
            text_list: Optional[List[str]] = None,
    ):
        # 分片参数健壮校验
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 不能小于0且不能大于等于chunk_size")

        # ========== 修复：避免Python可变默认参数共享状态的经典坑 ==========
        self.text_source_list = text_source_list if text_source_list is not None else []
        self.text_list = text_list if text_list is not None else []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.img_dir: Path = Path(img_dir).resolve()
        self.img_dir.mkdir(parents=True, exist_ok=True)

        # 全局复用分割器，优先markdown标题分隔符
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "]
        )

        # 图片匹配正则：哈希文件名 / img_数字.后缀 / 占位符格式
        self.sha256_img_pat = re.compile(r"\b[a-z0-9]{64}\.[a-zA-Z]+\b")
        self.serial_img_pat = re.compile(r"\bimg_(\d+)\.([a-zA-Z]+)\b")
        # ========== 新增：兼容data_wash输出的[图片:img_xx]占位符 ==========
        self.img_placeholder_pat = re.compile(r"\[图片:img_(\d+)\]")

        # 通用清洗正则（纯文本+md共用基础清洗）
        self.clean_ctrl_char = re.compile(r"[\x00-\x1F\x7F]")
        self.clean_multi_space = re.compile(r"\s+")
        self.clean_multi_enter = re.compile(r"\n\s*\n+")

        # md专属清洗正则
        self.clean_page_note = re.compile(r"<!-- page \d+ -->")
        self.clean_link = re.compile(r"\[(.*?)\]\(.*?\)")
        self.clean_ref_mark = re.compile(r"\[\d+\]")

        # 药品类文本正则
        self._medical_spliter_title = re.compile(r"【(.*?)】")
        self._medical_spliter_content = re.compile(r"】(.*?)【")
        self._medical_approve_line = re.compile(r"核准日期[:：]\s*(\d{4}年\d{1,2}月\d{1,2}日)")
        self._medical_modify_block = re.compile(r"(修改日期[:：][\s\S]*?(?=\n\n|\Z))")
        self._medical_data_type = re.compile(r"\d{4}年\d{1,2}月\d{1,2}日")
        self._medical_warning_block = re.compile(r"\d+、[\s\S]*?(?=\n\d+、|\n【|\Z)")

        # 医疗器械说明书专用正则
        self._device_model_pat = re.compile(r"型号[:：]\s*([A-Za-z0-9_\-]+)")
        self._device_step_pat = re.compile(r"\d+、[\s\S]*?(?=\d+、|\n\n|\Z)")
        self._device_forbid_pat = re.compile(r"【禁忌】[\s\S]*?(?=\n\n【|\Z)")
        self._device_fault_pat = re.compile(r"故障[:：][\s\S]*?解决方案[:：][\s\S]*?(?=\n\n|\Z)")
        self._device_date_pat = re.compile(r"修订日期[:：]\s*(\d{4}年\d{1,2}月\d{1,2}日)")

    def _clean_text(self, raw_text: str, is_markdown: bool) -> str:
        #内部统一清洗工具，区分纯文本/md，消除重复代码
        if not isinstance(raw_text, str):
            return ""
        txt = raw_text.strip()
        # 基础清洗：不可见字符、全角空格、连续空白换行
        txt = self.clean_ctrl_char.sub("", txt)
        txt = txt.replace("　", " ")
        txt = self.clean_multi_space.sub(" ", txt)
        txt = self.clean_multi_enter.sub("\n\n", txt)
        # md额外清洗注释、超链接、引用标记
        if is_markdown:
            txt = self.clean_page_note.sub("", txt)
            txt = self.clean_link.sub(r"\1", txt)
            txt = self.clean_ref_mark.sub("", txt)
        return txt.strip()

    def _extract_img_list(self, chunk_str: str) -> List[str]:
        #提取文本内所有图片id，消除重复代码，统一复用
        img_names = []
        sha_imgs = self.sha256_img_pat.findall(chunk_str)
        if sha_imgs:
            img_names.extend(sha_imgs)
        serial_tuples = self.serial_img_pat.findall(chunk_str)
        if serial_tuples:
            for num, ext in serial_tuples:
                img_names.append(f"img_{num}.{ext}")
        # ========== 新增：兼容占位符格式图片提取 ==========
        placeholder_imgs = self.img_placeholder_pat.findall(chunk_str)
        if placeholder_imgs:
            for num in placeholder_imgs:
                img_names.append(f"img_{num}")
        return list(set(img_names))

    def img_to_base64(self, img_id_list: Optional[List[str]]) -> List[str]:
        #批量读取图片转base64，完整健壮校验保留
        res = []
        if not img_id_list:
            return res
        unique_imgs = list(set(img_id_list))
        for img_name in unique_imgs:
            img_path = self.img_dir / img_name
            try:
                if not img_path.exists() or not img_path.is_file():
                    print(f"[警告] 图片不存在: {img_path}")
                    continue
                if img_path.stat().st_size == 0:
                    print(f"[警告] 空白损坏图片跳过: {img_path}")
                    continue
                with open(img_path, "rb") as f:
                    data = f.read()
                res.append(base64.b64encode(data).decode("utf-8"))
            except PermissionError:
                print(f"[权限不足] {img_path}")
            except Exception as e:
                print(f"图片读取失败 {img_path}: {e}")
                print_exc()
        return res

    def text_to_sha256(self, content: str) -> Optional[str]:
        # 文本生成sha256，兼容gbk编码兜底
        try:
            if not isinstance(content, str):
                content = ""
            content = content.strip()
            if not content:
                return None
            byte_data = content.encode("utf-8")
            return hashlib.sha256(byte_data).hexdigest()
        except UnicodeEncodeError:
            byte_data = content.encode("gbk")
            return hashlib.sha256(byte_data).hexdigest()
        except Exception as e:
            print(f"哈希计算异常: {e}")
            print_exc()
            return None

    def split_text_to_document(self, raw_text: str, is_markdown: bool = False,text_source: str = "") -> Optional[List[Document]]:
        # 统一对外入口，区分纯文本/md，内部完成清洗、分片、封装Document
        try:
            # 入参健壮校验
            if not isinstance(raw_text, str) or len(raw_text.strip()) == 0:
                print("[提示] 输入文本为空")
                return []
            # 执行对应清洗逻辑
            clean_str = self._clean_text(raw_text, is_markdown)
            if len(clean_str) == 0:
                return []

            # 整篇文档全局唯一hash（增量更新标识）
            full_doc_hash = self.text_to_sha256(clean_str)
            if full_doc_hash is None:
                print("[警告] 全文哈希生成失败")
                full_doc_hash = ""
            # 切分成文本片段
            chunk_str_list = self._splitter.split_text(clean_str)

            doc_result: List[Document] = []
            for chunk_str in chunk_str_list:
                chunk_str = chunk_str.strip()
                # 过滤空片段、小于10字符无效碎片
                if not chunk_str or len(chunk_str) < 10:
                    continue

                img_names = self._extract_img_list(chunk_str)
                # 单分片独立hash
                chunk_hash = self.text_to_sha256(chunk_str)
                if chunk_hash is None:
                    continue
                meta = {
                    "doc_full_hash": full_doc_hash,
                    "chunk_id": chunk_hash,
                    "img_id_list": img_names,
                    "source":text_source
                }
                # 固定标准写法生成Document
                single_doc = Document(page_content=chunk_str, metadata=meta)
                doc_result.append(single_doc)
            return doc_result
        except Exception as err:
            print(f"文本分割处理异常: {err}")
            print_exc()
            return None

    def special_text_medical(self, text: str, is_markdown: bool = False, text_source: str = "") -> Optional[
        List[Document]]:
        """
        药品说明书专用分片策略
        定向解析国药药品说明书markdown文本，独立拆分核准日期、修改日期、警告、各药品章节正文
        自动生成三层关联ID，附带章节标记、图片列表、文件来源元数据
        """
        doc_sha256 = self.text_to_sha256(text)
        try:
            clean_str = self._clean_text(text, is_markdown)
            if len(clean_str.strip()) == 0:
                return []

            # 1、提取固定独立区块：核准日期
            approve_data_chunk = "核准日期：无"
            approve_match = self._medical_approve_line.search(clean_str)
            if approve_match:
                date_str = approve_match.group(1)
                approve_data_chunk = f"核准日期：{date_str}"

            # 2、提取修改日期区块
            change_data_chunk = "修改日期：无"
            modify_match = self._medical_modify_block.search(clean_str)
            if modify_match:
                change_block_text = modify_match.group(1)
                change_data = self._medical_data_type.findall(change_block_text)
                if change_data:
                    dates = "\n".join(change_data)
                    change_data_chunk = f"修改日期：\n{dates}"

            # 3、提取警告段落
            warnning_data_chunk = "警告内容：无"
            warn_match = self._medical_warning_block.search(clean_str)
            if warn_match:
                warnning_data_chunk = warn_match.group()

            # 4、匹配所有章节标题+对应正文
            medical_title_chunks = self._medical_spliter_title.findall(clean_str)
            medical_content_chunks = self._medical_spliter_content.findall(clean_str)
            text_chunk_list = []
            for title, content in zip(medical_title_chunks, medical_content_chunks):
                full_chunk = f"【{title}】{content}".strip()
                if full_chunk:
                    text_chunk_list.append((title, full_chunk))

            document_list: List[Document] = []
            MIN_TEXT_LEN = 10

            # 封装构造基础文档方法，消除重复代码
            def build_base_doc(content: str, mark: str, father_id: str = "") -> Document:
                chunk_hash = self.text_to_sha256(content)
                return Document(
                    page_content=content,
                    metadata={
                        "chunk_id": chunk_hash,
                        "father_id": father_id,
                        "grandfather_id": doc_sha256,
                        "mark": mark,
                        "source": text_source,
                        "img_id_list": []
                    }
                )

            # 写入三类独立固定文档
            if len(approve_data_chunk) > 6:
                document_list.append(build_base_doc(approve_data_chunk, mark="核准日期"))
            if len(change_data_chunk) > 6:
                document_list.append(build_base_doc(change_data_chunk, mark="修改日期"))
            if len(warnning_data_chunk) > 4:
                document_list.append(build_base_doc(warnning_data_chunk, mark="警告"))

            # 遍历各个章节进行分片（修复原版双层重复分割问题）
            for title, parse_chunk in text_chunk_list:
                # 章节整体进行一次分片作为父块
                parent_chunk_list = self._splitter.split_text(parse_chunk)
                for parent_text in parent_chunk_list:
                    parent_text = parent_text.strip()
                    if not parent_text or len(parent_text) < MIN_TEXT_LEN:
                        continue
                    parent_hash = self.text_to_sha256(parent_text)

                    # 父块再次细分最小子块，维持父子层级结构
                    child_chunk_list = self._splitter.split_text(parent_text)
                    for child_text in child_chunk_list:
                        child_text = child_text.strip()
                        if not child_text or len(child_text) < MIN_TEXT_LEN:
                            continue
                        child_hash = self.text_to_sha256(child_text)
                        img_names = self._extract_img_list(child_text)

                        title_res = self._medical_spliter_title.search(child_text)
                        # 和你原始逻辑保持一致：无匹配标题则标记为通用章节
                        mark_name = title_res.group(1) if title_res else "通用章节"

                        document_list.append(Document(
                            page_content=child_text,
                            metadata={
                                "chunk_id": child_hash,
                                "father_id": parent_hash,
                                "grandfather_id": doc_sha256,
                                "mark": mark_name,
                                "img_id_list": img_names,
                                "source": text_source
                            }
                        ))

            return document_list
        except Exception as e:
            print(f"【药品说明书解析异常】：{e}")
            print_exc()
            return []

    def special_text_device(self, text: str, is_markdown: bool = False,text_source: str = "") -> Optional[List[Document]]:
        try:
            clean_str = self._clean_text(text, is_markdown)
            if len(clean_str) == 0:
                return []

            document_list = []
            # 提取设备型号
            model_match = self._device_model_pat.search(clean_str)
            if model_match:
                model_chunk = f"设备型号：{model_match.group(1)}"
                doc_model = Document(
                    page_content=model_chunk,
                    metadata={
                        "chunk_id": self.text_to_sha256(model_chunk),
                        "mark": "设备型号",
                        "source":text_source
                    }
                )
                document_list.append(doc_model)

            # 提取修订日期
            date_match = self._device_date_pat.search(clean_str)
            if date_match:
                date_chunk = f"文档修订日期：{date_match.group(1)}"
                doc_date = Document(
                    page_content=date_chunk,
                    metadata={
                        "chunk_id": self.text_to_sha256(date_chunk),
                        "mark": "修订日期",
                        "source":text_source
                    }
                )
                document_list.append(doc_date)

            # 提取禁忌内容
            forbid_match = self._device_forbid_pat.search(clean_str)
            if forbid_match:
                forbid_chunk = forbid_match.group().strip()
                if forbid_chunk:
                    doc_forbid = Document(
                        page_content=forbid_chunk,
                        metadata={
                            "chunk_id": self.text_to_sha256(forbid_chunk),
                            "mark": "禁忌",
                            "source":text_source
                        }
                    )
                    document_list.append(doc_forbid)

            # 逐条拆分操作步骤
            step_list = self._device_step_pat.findall(clean_str)
            for step_content in step_list:
                step_content = step_content.strip()
                if not step_content or len(step_content) < 10:
                    continue
                chunk_hash = self.text_to_sha256(step_content)
                if chunk_hash is None:
                    continue
                img_names = self._extract_img_list(step_content)
                document_list.append(Document(
                    page_content=step_content,
                    metadata={
                        "chunk_id": chunk_hash,
                        "mark": "操作步骤",
                        "img_id_list": img_names,
                        "source":text_source
                    }
                ))

            # 逐条拆分故障与解决方案
            fault_list = self._device_fault_pat.findall(clean_str)
            for fault_content in fault_list:
                fault_content = fault_content.strip()
                if not fault_content or len(fault_content) < 10:
                    continue
                chunk_hash = self.text_to_sha256(fault_content)
                if chunk_hash is None:
                    continue
                img_names = self._extract_img_list(fault_content)
                document_list.append(Document(
                    page_content=fault_content,
                    metadata={
                        "chunk_id": chunk_hash,
                        "mark": "故障解决方案",
                        "img_id_list": img_names,
                        "source":text_source
                    }
                ))

            return document_list
        except Exception as e:
            print(f"器械说明书解析异常：{e}")
            print_exc()
            return []

    def text_batch_tanckle(
            self,
            method: Optional[Callable[[str, bool, str], Optional[List[Document]]]] = None,
            text_list: Optional[List[str]] = None,
            text_source_list: Optional[List[str]] = None,
            is_markdown: bool = False
    ) -> List[Document]:
        """
        批量文本处理入口
        :param method: 类内文本解析方法，可选：split_text_to_document / special_text_medical / special_text_device
        :param text_list: 待处理原始文本数组
        :param text_source_list: 每条文本对应的来源标识，与text_list一一对应
        :param is_markdown: 是否为markdown格式文本
        :return: 全部文本合并后的Document列表
        """
        if method is None:
            method = self.split_text_to_document
        all_docs: List[Document] = []
        if text_list is None or len(text_list) == 0:
            text_list = self.text_list
        # ========== 修复：先拷贝列表，不修改实例原属性，避免副作用 ==========
        if text_source_list is not None and len(text_source_list) > 0:
            use_source_list = list(text_source_list)
        else:
            use_source_list = list(self.text_source_list)

        text_count = len(text_list)

        # 对齐来源列表长度，不足部分自动填充「未知来源」，防止zip截断数据
        if len(use_source_list) < text_count:
            fill_num = text_count - len(use_source_list)
            use_source_list += ["未知来源"] * fill_num
        # 截断多余来源，保证和文本数量严格一一对应
        use_source_list = use_source_list[:text_count]

        # 逐条绑定文本+来源处理
        for raw_text, text_source in zip(text_list, use_source_list):
            try:
                doc_list = method(raw_text, is_markdown, text_source)
                if doc_list and isinstance(doc_list, list):
                    all_docs.extend(doc_list)
            except Exception as err:
                print(f"【警告】单条文本批量处理失败，来源:{text_source}，错误信息：{err}")
                print_exc()
                continue
        return all_docs
