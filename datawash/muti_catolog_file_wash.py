import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import fitz
from myproject.datawash.mineru_wash import mineru_wash

class data_wash:
    def __init__(self, file_path_list: list|str, image_save_path: str, markdown_save_dir: str, outdir: str):
        # 入参统一转为Path对象，跨系统路径兼容
        self.file_path_list: List[str] = [str(Path(p).resolve()) for p in file_path_list] if file_path_list else []
        self.image_save_path: Path = Path(image_save_path).resolve()
        self.markdown_save_dir: Path = Path(markdown_save_dir).resolve()
        self.outdir: Path = Path(outdir).resolve()
        self.allow_file_type = {"txt", "md", "docx", "pdf"}  # 集合查询效率更高

        # 初始化全部输出目录，前置创建，避免方法内重复mkdir
        self.image_save_path.mkdir(parents=True, exist_ok=True)
        self.markdown_save_dir.mkdir(parents=True, exist_ok=True)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def text_wash(self, file_path: str) -> Optional[str]:
        """纯txt文本读取，增加编码容错、空文件过滤"""
        file_p = Path(file_path).resolve()
        if not file_p.exists() or not file_p.is_file():
            print(f"[警告] 文件不存在或不是有效文件: {file_path}")
            return None
        if file_p.stat().st_size == 0:
            print(f"[警告] 空白文件跳过: {file_path}")
            return None
        try:
            with open(file_p, 'r', encoding="utf-8") as f:
                text = f.read().strip()
            return text
        except UnicodeDecodeError:
            # 中文Windows编码兜底
            with open(file_p, 'r', encoding="gbk") as f:
                text = f.read().strip()
            return text
        except Exception as e:
            print(f"[错误] txt解析失败 {file_path}: {str(e)}")
            return None

    def md_wash(self, file_path: str) -> Optional[str]:
        """md文件读取，和txt共用容错逻辑，代码复用"""
        return self.text_wash(file_path)

    def docx_wash(self, file_path: str, image_save_path: str, markdown_save_dir: str) -> Optional[Tuple[str, List[str]]]:
        """docx原生解析，还原你原始自增编号图片逻辑，修复md图片语法缺失闭合括号bug"""
        file_p = Path(file_path).resolve()
        img_save_p = Path(image_save_path).resolve()
        md_save_p = Path(markdown_save_dir).resolve()

        if not file_p.exists() or file_p.suffix.lower().strip(".") != "docx":
            print(f"[警告] docx文件无效: {file_path}")
            return None
        if file_p.stat().st_size == 0:
            print(f"[警告] 空白docx文件跳过: {file_path}")
            return None

        img_serial_number = 1
        img_path_list: List[str] = []

        def save_image(image):
            nonlocal img_serial_number
            img_data = image.read()
            img_type = image.content_type.split("/")[-1]
            img_file_name = f"img_{img_serial_number}.{img_type}"
            img_full_path = str(img_save_p / img_file_name)
            img_path_list.append(img_full_path)
            with open(img_full_path, 'wb') as f:
                f.write(img_data)
            img_serial_number += 1
            # 修复原代码markdown图片语法缺失闭合括号bug
            return f"![img_{img_serial_number-1}]({img_full_path})"

        try:
            import mammoth
            with open(file_p, 'rb') as f:
                convert_result = mammoth.convert_to_markdown(
                    f,
                    convert_image=mammoth.images.img_element(save_image)
                )
            markdown_content = convert_result.value.strip()
            warning_messages = convert_result.messages
            if warning_messages:
                print(f"[docx解析警告 {file_path}] {warning_messages}")

            # ========== 新增：把md图片标签统一替换成 [图片:img_xx] 占位 ==========
            img_pat = re.compile(r"!\[.*?\]\(.*?img_(\d+).*?\)")
            markdown_content = img_pat.sub(r"[图片:img_\1]", markdown_content)

            # 保存md文件，强制utf-8编码
            md_file_name = f"{file_p.stem}.md"
            md_output_path = md_save_p / md_file_name
            with open(md_output_path, 'w', encoding="utf-8") as f:
                f.write(markdown_content)
            return markdown_content, img_path_list
        except Exception as e:
            print(f"[错误] docx原生解析异常 {file_path}: {str(e)}")
            return None

    def docx_wash_method(self, file_path: str, image_save_path: str, markdown_save_dir: str, outdir: str) -> Optional[Tuple[str, str, List[str]]]:
        """保留你原有降级逻辑：mammoth失败自动走MinerU，返回md文本+mineru输出路径+图片列表"""
        file_p = Path(file_path).resolve()
        out_p = Path(outdir).resolve()
        try:
            wash_result = self.docx_wash(file_path, image_save_path, markdown_save_dir)
            if wash_result is not None:
                md_content, img_list = wash_result
                # 成功返回文本，第二个占位无mineru路径
                return md_content, "", img_list
            raise RuntimeError("docx原生解析返回空结果")
        except Exception:
            try:
                # ========== 直接使用 mineru_wash 返回值，不重复查找 ==========
                md_content, mineru_markdown_path = mineru_wash(str(file_p), str(out_p))
                if md_content is None or mineru_markdown_path is None:
                    print(f"[警告] MinerU解析失败: {file_path}")
                    return None
                # 扫描 MinerU 输出目录的 images 文件夹，获取图片列表
                mineru_img_dir = Path(mineru_markdown_path).parent / "images"
                img_list: List[str] = []
                if mineru_img_dir.exists():
                    img_list = sorted([str(p) for p in mineru_img_dir.iterdir() if p.is_file()])
                return md_content, mineru_markdown_path, img_list
            except Exception as e:
                print(f"[错误] docx降级MinerU解析全部失败 {file_path}: {str(e)}")
                return None

    def pdf_wash(self, file_path: str, image_save_path: str, markdown_save_dir: str) -> Optional[Tuple[str, List[str]]]:
        file_p = Path(file_path).resolve()
        img_save_p = Path(image_save_path).resolve()
        md_save_p = Path(markdown_save_dir).resolve()
        # 自动创建目录
        img_save_p.mkdir(parents=True, exist_ok=True)
        md_save_p.mkdir(parents=True, exist_ok=True)

        # 文件校验
        if not file_p.exists() or file_p.suffix.lower().strip(".") != "pdf":
            print(f"[警告] pdf文件无效: {file_path}")
            return None
        if file_p.stat().st_size == 0:
            print(f"[警告] 空白pdf文件跳过: {file_path}")
            return None

        full_text = ""
        image_save_path_list: List[str] = []
        img_id = 1
        doc = None
        try:
            doc = fitz.open(file_p)
            for page in doc:
                page_content_parts = []
                blocks = page.get_text("blocks")
                # 区块从上到下、从左到右排序
                blocks.sort(key=lambda b: (b[1], b[0]))

                # 提取当前页所有表格
                table_info = []
                table_list = page.find_tables()
                for tbl in table_list.tables:
                    tbl_data = tbl.extract()
                    if not tbl_data or all(len("".join(str(c).strip() for c in row)) == 0 for row in tbl_data):
                        # 过滤全空白表格
                        continue
                    y_min = tbl.bbox[1]
                    # 构建md表格
                    header_row = tbl_data[0]
                    md_table_lines = ["| " + " | ".join([str(cell).strip() for cell in header_row]) + " |"]
                    md_table_lines.append("| " + " | ".join(["----"] * len(header_row)) + " |")
                    for row in tbl_data[1:]:
                        md_table_lines.append("| " + " | ".join([str(cell).strip() for cell in row]) + " |")
                    full_md_table = "\n" + "\n".join(md_table_lines) + "\n"
                    table_info.append({"y_min": y_min, "y_max": tbl.bbox[3], "table_text": full_md_table, "bbox": tbl.bbox})

                # ========== 修复：提取图片并记录位置信息，后续按y坐标插入对应位置 ==========
                page_images = []
                all_img_xref = page.get_images(full=True)
                for xref, _, _, _, _, _, _, _ in all_img_xref:
                    pix = fitz.Pixmap(doc, xref)
                    # 过滤纯白空白图
                    is_blank = True
                    if pix.n >= 1:
                        sample = pix.pixel(10, 10)
                        if not all(v == 255 for v in sample[:pix.n]):
                            is_blank = False
                    if not is_blank:
                        img_file_name = f"img_{img_id}.png"
                        img_full_path = str(img_save_p / img_file_name)
                        pix.save(img_full_path)
                        image_save_path_list.append(img_full_path)
                        # 获取图片在页面上的位置信息
                        img_bbox = doc.get_image_bbox(xref)
                        page_images.append({
                            "y_min": img_bbox[1],
                            "y_max": img_bbox[3],
                            "content": f"\n[图片:img_{img_id}]\n"
                        })
                        img_id += 1
                    pix.destroy()
                    pix = None
                # 按y坐标排序图片
                page_images.sort(key=lambda img: img["y_min"])

                # 遍历所有图文区块
                for block in blocks:
                    x0, y0, x1, y1, blk_text, blk_type = block

                    # 插入当前区块上方所有未放置的图片
                    insert_imgs = []
                    remain_imgs = []
                    for img in page_images:
                        if img["y_max"] < y0:
                            insert_imgs.append(img)
                        else:
                            remain_imgs.append(img)
                    # 从上至下插入图片
                    insert_imgs.sort(key=lambda i: i["y_min"])
                    for img in insert_imgs:
                        page_content_parts.append(img["content"])
                    page_images = remain_imgs

                    # 插入当前区块上方所有未放置的表格
                    insert_tbls = []
                    remain_tbls = []
                    for tbl in table_info:
                        if tbl["y_max"] < y0:
                            insert_tbls.append(tbl)
                        else:
                            remain_tbls.append(tbl)
                    # 从上至下插入表格
                    insert_tbls.sort(key=lambda t: t["y_min"])
                    for t in insert_tbls:
                        page_content_parts.append(t["table_text"])
                    table_info = remain_tbls

                    # 文字保留原有换行，不粗暴strip合并
                    if blk_type == 0:
                        page_content_parts.append(blk_text)

                # 页面末尾插入剩余图片
                for last_img in page_images:
                    page_content_parts.append(last_img["content"])

                # 页面末尾插入剩余表格
                for last_tbl in table_info:
                    page_content_parts.append(last_tbl["table_text"])

                page_full = "".join(page_content_parts)
                full_text += page_full + "\n\n"

            # 落地MD文件
            md_file_name = f"{file_p.stem}.md"
            md_output_path = md_save_p / md_file_name
            with open(md_output_path, 'w', encoding="utf-8") as f:
                f.write(full_text.strip())
            return full_text.strip(), image_save_path_list

        except Exception as e:
            print(f"[错误] pdf原生解析失败 {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if doc is not None:
                doc.close()

    def pdf_wash_method(self, file_path: str, image_save_path: str, markdown_save_dir: str, outdir: str) -> Optional[Tuple[str, str, List[str]]]:
        """PDF降级MinerU逻辑，完全保留你原有业务流程，补充异常打印，返回md文本+mineru输出路径+图片列表"""
        file_p = Path(file_path).resolve()
        out_p = Path(outdir).resolve()
        try:
            wash_result = self.pdf_wash(file_path, image_save_path, markdown_save_dir)
            if wash_result is not None:
                md_content, img_list = wash_result
                return md_content, "", img_list
            raise RuntimeError("pdf原生解析返回空结果")
        except Exception:
            try:
                # ========== 直接使用 mineru_wash 返回值，不重复查找 ==========
                md_content, mineru_markdown_path = mineru_wash(str(file_p), str(out_p))
                if md_content is None or mineru_markdown_path is None:
                    print(f"[警告] MinerU解析失败: {file_path}")
                    return None
                # 扫描 MinerU 输出目录的 images 文件夹，获取图片列表
                mineru_img_dir = Path(mineru_markdown_path).parent / "images"
                img_list: List[str] = []
                if mineru_img_dir.exists():
                    img_list = sorted([str(p) for p in mineru_img_dir.iterdir() if p.is_file()])
                return md_content, mineru_markdown_path, img_list
            except Exception as e:
                print(f"[错误] pdf降级MinerU解析全部失败 {file_path}: {str(e)}")
                return None

    def multi_data_wash(self) -> Dict[str, Dict[str, Any]]:
        """批量文件统一清洗入口，字典存储【文本+图片列表】不丢图片信息"""
        content_list: Dict[str, Dict[str, Any]] = {}
        if not self.file_path_list:
            print("[提示] 传入文件列表为空，无文档待清洗")
            return content_list

        for file_path in self.file_path_list:
            file_p = Path(file_path).resolve()
            # 修复原split(".")[-1]识别带点文件名bug
            file_suffix = file_p.suffix.lower().strip(".")
            file_key = file_p.stem  # 字典key和你原有逻辑保持一致

            if file_suffix not in self.allow_file_type:
                print(f"[跳过] 不支持的文件格式 {file_suffix}: {file_path}")
                continue

            wash_result = None
            img_list: List[str] = []
            mineru_md_path = ""

            if file_suffix == "txt":
                text = self.text_wash(file_path)
                wash_result = text
            elif file_suffix == "md":
                text = self.md_wash(file_path)
                wash_result = text
            elif file_suffix == "docx":
                # 先走降级兜底方法（内部已包含原生解析），返回文本+mineru路径+图片列表
                fallback_res = self.docx_wash_method(file_path, str(self.image_save_path), str(self.markdown_save_dir), str(self.outdir))
                if fallback_res is not None:
                    wash_result, mineru_md_path, img_list = fallback_res
            elif file_suffix == "pdf":
                # 先走降级兜底方法（内部已包含原生解析），返回文本+mineru路径+图片列表
                fallback_res = self.pdf_wash_method(file_path, str(self.image_save_path), str(self.markdown_save_dir), str(self.outdir))
                if fallback_res is not None:
                    wash_result, mineru_md_path, img_list = fallback_res

            # 过滤空文档，不写入结果字典
            if wash_result is None or len(str(wash_result).strip()) == 0:
                print(f"[跳过] 文件解析无有效内容: {file_path}")
                continue

            # 存储完整数据，上层可直接取用
            content_list[file_key] = {
                "markdown_text": wash_result,
                "image_paths": img_list,
                "mineru_md_file": mineru_md_path,
                "original_file_path": str(file_p)
            }
        return content_list


# 兼容旧调用别名（需先实例化再调用）
# 用法示例：wash = data_wash(...); result = wash.multi_data_wash()
muti_data_wash = data_wash.multi_data_wash
