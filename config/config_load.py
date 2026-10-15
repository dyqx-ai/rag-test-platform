import os
from pathlib import Path
from typing import Optional
import yaml

# ====================== 项目根目录（基于 __file__ 向上递归计算绝对路径）======================
_CONFIG_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))  # config/
_PROJECT_ROOT = os.path.dirname(_CONFIG_MODULE_DIR)              # myproject/
_DEFAULT_CONFIG_PATH = os.path.join(_CONFIG_MODULE_DIR, "config.yaml")


def get_project_root() -> str:
    """获取项目根目录绝对路径"""
    return _PROJECT_ROOT


def config_load(yaml_path: Optional[str] = None) -> dict:
    """
    加载配置文件
    - 优先使用传入的绝对路径
    - 未传参时自动基于 __file__ 计算 config/config.yaml 的绝对路径
    - 兼容 Docker 容器部署：支持环境变量 CONFIG_PATH 覆盖
    """
    # 环境变量覆盖（Docker 部署场景）
    if not yaml_path:
        yaml_path = os.getenv("RAG_CONFIG_PATH", _DEFAULT_CONFIG_PATH)

    yaml_p = Path(yaml_path)
    if not yaml_p.is_absolute():
        # 相对路径 → 基于项目根目录拼接
        yaml_p = Path(_PROJECT_ROOT) / yaml_path

    yaml_p = yaml_p.resolve()

    if not yaml_p.exists():
        raise FileNotFoundError(f"【RAG系统】配置文件不存在: {yaml_p}，请检查路径")

    try:
        with open(yaml_p, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None:
            raise ValueError(f"【RAG系统】配置文件为空: {yaml_p}")
        print(f"[RAG启动] 配置文件加载成功: {yaml_p}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"【RAG系统】配置文件YAML解析失败: {yaml_p}，错误: {e}")