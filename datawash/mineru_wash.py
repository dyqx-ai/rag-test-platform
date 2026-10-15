import subprocess
import os
from pathlib import Path
from typing import Optional, Tuple

# 优先从 config.yaml 读取 MinerU 路径，读取失败回退硬编码兜底
try:
    from config.config_load import config_load
    _cfg = config_load()
    MINERU_EXE = _cfg.get("path", {}).get("mineru_exe", r"C:\pyProject\minerU\pdf_env\Scripts\mineru.exe")
except Exception:
    MINERU_EXE = r"C:\pyProject\minerU\pdf_env\Scripts\mineru.exe"

# 注意：移除 os.chdir(BASE_DIR)，避免改变全局工作目录影响后续相对路径操作
# MinerU 命令行调用使用绝对路径，不需要改变工作目录


def mineru_wash(file_path: str, out_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    调用MinerU解析PDF / DOCX，返回 (md完整文本, md真实文件路径)
    :param file_path: 原始文件路径（支持pdf、docx）
    :param out_dir: MinerU顶层输出目录
    :return: (markdown文本 / None, md完整路径 / None)
    """
    # ========== 前置校验1：校验mineru程序是否存在 ==========
    mineru_p = Path(MINERU_EXE)
    if not mineru_p.exists() or not mineru_p.is_file():
        print(f"【致命错误】MinerU程序不存在，检查路径：{MINERU_EXE}")
        return None, None

    # ========== 前置校验2：校验输入文件，兼容pdf、docx ==========
    file_p = Path(file_path).resolve()
    if not file_p.exists() or not file_p.is_file():
        print(f"【错误】待解析文件不存在：{file_path}")
        return None, None
    # 支持两种后缀
    allow_suffix = {".pdf", ".docx"}
    file_suffix = file_p.suffix.lower()
    if file_suffix not in allow_suffix:
        print(f"【错误】MinerU仅支持pdf/docx，当前文件格式：{file_suffix}，文件：{file_path}")
        return None, None

    # ========== 修复文件名分割bug：兼容文件名带小数点 ==========
    file_stem = file_p.stem  # 自动剥离后缀，v1.2说明书.pdf → v1.2说明书
    out_p = Path(out_dir).resolve()
    # 提前创建顶层输出文件夹
    out_p.mkdir(parents=True, exist_ok=True)

    cmd = [
        MINERU_EXE,
        "-p", str(file_p),
        "-o", str(out_p),
        "-b", "pipeline",
        "--device", "cpu"
    ]

    print(f"【MinerU】开始执行：文件={file_p.name}，输出目录={out_p}")
    print(f"【MinerU】命令：{' '.join(cmd)}")
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600
        )
        # ========== 修复：编码兜底，utf-8失败自动回退gbk ==========
        try:
            stdout = res.stdout.decode("utf-8")
            stderr = res.stderr.decode("utf-8")
        except UnicodeDecodeError:
            stdout = res.stdout.decode("gbk", errors="ignore")
            stderr = res.stderr.decode("gbk", errors="ignore")

        # ========== 打印完整日志 ==========
        print(f"【MinerU】返回码：{res.returncode}")
        if stdout:
            print(f"【MinerU】标准输出：\n{stdout}")
        if stderr:
            print(f"【MinerU】错误输出：\n{stderr}")

        if res.returncode == 0:
            print("【MinerU】解析执行成功")
            # 拼接子文件夹路径 out_dir/文件名/auto
            sub_output_path = out_p / file_stem / "auto"
            # 校验子文件夹是否生成
            if not sub_output_path.exists() or not sub_output_path.is_dir():
                print(f"【警告】MinerU未生成auto子目录，完整日志：\n{stdout}")
                return None, None

            # 筛选md文件，排序后优先取主内容文件
            all_files = os.listdir(sub_output_path)
            md_file_list = [file for file in all_files if file.endswith(".md")]
            if not md_file_list:
                print("【警告】子目录无Markdown输出，完整运行日志：\n", stdout)
                return None, None

            # 排序 + 优先取带content的主文件
            md_file_list.sort()
            main_md = [f for f in md_file_list if "_content" in f]
            target_md = main_md[0] if main_md else md_file_list[0]
            real_md_path = str(sub_output_path / target_md)

            # 读取md内容
            with open(real_md_path, "r", encoding="utf-8") as f:
                full_md_text = f.read()
            print("文档前800字符预览：\n", full_md_text[:800])
            # 返回文本+文件路径，给上层data_wash调用
            return full_md_text, real_md_path
        else:
            print("MinerU解析失败，错误日志：", stderr)
            return None, None

    except FileNotFoundError as e:
        print("【异常】找不到mineru.exe程序，路径异常：", e)
        return None, None
    except subprocess.TimeoutExpired:
        print("【异常】文件解析超时（10分钟限制），文件过大或损坏")
        return None, None
    except PermissionError:
        print("【异常】文件权限不足，无法读取源文件或写入输出目录")
        return None, None
    except Exception as e:
        print(f"【未知异常】MinerU执行失败：{str(e)}")
        return None, None
