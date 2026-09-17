"""
Agnes 核心共享基础
==================

提供统一的客户端创建、结果保存与参考图归一化逻辑，
供 generate / style_edit / text_edit 三个核心模块复用。

作者: Sol
日期: 2026-09-16
"""

import base64
import os
import urllib.parse
import urllib.request
import uuid

from examples.multimodal.agnes.config import get_model_config
from examples.multimodal.agnes.utils.errors import AgnesConfigError
from examples.multimodal.agnes.utils.image_utils import local_image_to_data_uri

# 生成结果默认保存目录
DEFAULT_SAVE_DIR = "output"


def create_client(api_key: str | None = None, base_url: str | None = None):
    """
    创建 Agnes 的 OpenAI 兼容客户端

    Args:
        api_key: Agnes API Key；为 None 时从环境变量 AGNES_API_KEY 读取
        base_url: API 地址；为 None 时优先取环境变量 AGNES_BASE_URL，再缺省用默认值

    Returns:
        openai.OpenAI 客户端实例

    Raises:
        AgnesConfigError: 未配置 API Key 时
    """
    from openai import OpenAI

    config = get_model_config()
    key = api_key or os.getenv(config.api_key_env)
    if not key:
        raise AgnesConfigError(
            f"未找到 {config.api_key_env} 环境变量，"
            "请将根目录 .env.example 复制为 .env，填入 Agnes API Key 后重试"
        )
    return OpenAI(api_key=key, base_url=base_url or config.base_url)


def save_generated_images(response, save_dir: str = DEFAULT_SAVE_DIR) -> list[str]:
    """
    将生成结果保存到本地目录

    支持两种返回形式：
    - item.url      ：通过 urllib 下载到本地
    - item.b64_json ：Base64 解码后写盘

    Args:
        response: client.images.generate() 的返回对象
        save_dir: 保存目录，默认 "output"

    Returns:
        保存成功的文件路径列表
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    for idx, item in enumerate(response.data, start=1):
        # 场景一：返回公网 URL，直接下载到本地
        if getattr(item, "url", None):
            # 从 URL 路径提取扩展名，取不到时默认 .png
            ext = os.path.splitext(urllib.parse.urlparse(item.url).path)[1] or ".png"
            file_path = os.path.join(save_dir, f"{uuid.uuid4().hex[:8]}_{idx}{ext}")
            urllib.request.urlretrieve(item.url, file_path)
            saved_paths.append(file_path)
        # 场景二：返回 Base64 编码，解码后写盘
        elif getattr(item, "b64_json", None):
            b64 = item.b64_json
            # 若带 data:image/...;base64, 前缀则先剥离
            if b64.startswith("data:image"):
                b64 = b64.split(",", 1)[1]
            data = base64.b64decode(b64)
            file_path = os.path.join(save_dir, f"{uuid.uuid4().hex[:8]}_{idx}.png")
            with open(file_path, "wb") as f:
                f.write(data)
            saved_paths.append(file_path)
    return saved_paths


def _normalize_images(image):
    """
    将参考图输入归一化为 API 可用的字符串或字符串列表

    - 单图字符串 / 列表均可，多图参考时传列表
    - 非 http(s) 开头的字符串视为本地文件路径，自动转 Data URI
    - 公网 URL 原样透传

    Args:
        image: 公网 URL 字符串、本地文件路径字符串，或二者组成的列表

    Returns:
        单图返回字符串，多图返回列表
    """
    images = image if isinstance(image, list) else [image]
    normalized = []
    for img in images:
        # 非 http/https 开头的字符串视为本地文件路径，转 Base64
        if isinstance(img, str) and not img.startswith(("http://", "https://")):
            normalized.append(local_image_to_data_uri(img))
        else:
            normalized.append(img)
    return normalized if len(normalized) > 1 else normalized[0]
