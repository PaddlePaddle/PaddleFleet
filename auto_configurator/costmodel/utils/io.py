#!/usr/bin/env python3
"""
通用 I/O 工具 - 文件加载与解析

提供 yaml/json 文件加载的统一接口，供 costmodel 和 config 模块复用。
"""

import json
from pathlib import Path
from typing import Any, Dict


def load_dict_from_file(path: str) -> Dict[str, Any]:
    """
    从文件加载字典，支持 .json / .yaml / .yml

    对于无后缀或未知后缀的文件，依次尝试 JSON 和 YAML 解析。

    Args:
        path: 文件路径

    Returns:
        解析后的字典

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式或解析失败
    """
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path}")

    suffix = file_path.suffix.lower()
    raw_text = file_path.read_text(encoding="utf-8")

    if suffix == ".json":
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败 ({file_path}): {e}")

    if suffix in (".yaml", ".yml"):
        return _load_yaml(raw_text, file_path)

    # 无后缀或未知后缀：尝试 JSON，失败则尝试 YAML
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    return _load_yaml(raw_text, file_path)


def _load_yaml(raw_text: str, file_path: Path) -> Dict[str, Any]:
    """解析 YAML 文本，缺少 PyYAML 时给出清晰错误提示。"""
    try:
        import yaml
    except ImportError:
        raise ValueError("解析 YAML 需要 PyYAML，请安装: pip install pyyaml")
    try:
        data = yaml.safe_load(raw_text)
    except Exception as e:
        raise ValueError(f"YAML 解析失败 ({file_path}): {e}")
    if not isinstance(data, dict):
        raise ValueError(f"YAML 文件顶层必须是字典 ({file_path})")
    return data
