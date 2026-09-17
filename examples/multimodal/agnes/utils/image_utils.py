"""
Agnes 图片处理工具函数
======================

提供本地图片转 Data URI、MIME 推断、公网图片下载与路径校验等辅助能力。
仅依赖 Python 标准库，不引入 Pillow 等第三方库。

作者: yaosh
日期: 2026-09-16
"""

import base64
import os
import urllib.request

from examples.multimodal.agnes.utils.errors import AgnesImageError


def infer_mime(image_path: str) -> str:
    """
    根据文件扩展名推断 MIME 子类型，未知后缀默认按 png 处理

    Args:
        image_path: 图片文件路径

    Returns:
        MIME 子类型，如 "jpeg" / "webp" / "png"
    """
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "jpeg"
    if ext == "webp":
        return "webp"
    return "png"


def local_image_to_data_uri(image_path: str) -> str:
    """
    将本地图片文件转换为 Base64 Data URI

    Args:
        image_path: 本地图片文件路径

    Returns:
        符合 API 要求的 data:image/<格式>;base64,<...> 字符串

    Raises:
        AgnesImageError: 文件不存在
    """
    if not os.path.exists(image_path):
        raise AgnesImageError(f"本地图片文件不存在：{image_path}")
    with open(image_path, "rb") as f:
        data = f.read()
    mime = infer_mime(image_path)
    return f"data:image/{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def download_image(url: str, save_path: str) -> str:
    """
    下载公网图片到本地

    Args:
        url: 图片公网地址
        save_path: 本地保存路径

    Returns:
        本地保存路径

    Raises:
        AgnesImageError: 下载失败
    """
    try:
        urllib.request.urlretrieve(url, save_path)
    except Exception as e:
        raise AgnesImageError(f"下载图片失败：{url}，原因：{e}") from e
    return save_path


def validate_image_path(image_path: str) -> str:
    """
    校验本地图片路径存在

    Args:
        image_path: 本地图片文件路径

    Returns:
        原样返回路径

    Raises:
        AgnesImageError: 文件不存在
    """
    if not os.path.exists(image_path):
        raise AgnesImageError(f"本地图片文件不存在：{image_path}")
    return image_path
