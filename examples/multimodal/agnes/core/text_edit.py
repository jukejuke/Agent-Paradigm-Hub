"""
Agnes 文字编辑模块
==================

支持两种文字操作：
1. 文生图带文字：生成时就渲染指定文字（在图片中添加文字）
2. 图生图改字 / 删字：参考一张含文字的图，按指令修改或删除图中文字

作者: yaosh
日期: 2026-09-16
"""

from examples.multimodal.agnes.config import MODEL_CONFIG, validate_ratio, validate_size
from examples.multimodal.agnes.core.base import DEFAULT_SAVE_DIR, _normalize_images, save_generated_images
from examples.multimodal.agnes.utils.errors import with_retry


@with_retry()
def generate_image_with_text(
    client,
    prompt: str,
    size: str = MODEL_CONFIG.default_size,
    ratio: str = MODEL_CONFIG.default_ratio,
    n: int = 1,
    save_dir: str = DEFAULT_SAVE_DIR,
) -> list[str]:
    """
    文生图生成带指定文字的图片（在图片中添加文字）

    说明：
    - 提示词需把要渲染的文字内容显式写出（可用「」标注），并强调文字一致性
      （可用 prompt_utils.build_text_add_prompt 拼装）
    - 支持中英文文字渲染（agnes-image-2.5-flash 特性），适合海报标题、店招等场景

    Args:
        client: Agnes OpenAI 兼容客户端
        prompt: 图像描述提示词（含要渲染的文字内容）
        size: 分辨率档位，默认 "2K"
        ratio: 宽高比，默认 "1:1"
        n: 生成图片数量，默认 1
        save_dir: 保存目录，默认 "output"

    Returns:
        保存到本地的图片文件路径列表

    Raises:
        AgnesConfigError: size / ratio 不合法
        AgnesAPIError: API 调用失败（429/5xx 已自动指数退避重试）
    """
    validate_size(size)
    validate_ratio(ratio)

    response = client.images.generate(
        model=MODEL_CONFIG.model_id,
        prompt=prompt,
        size=size,
        n=n,
        extra_body={"ratio": ratio},
    )
    return save_generated_images(response, save_dir=save_dir)


@with_retry()
def edit_image_text(
    client,
    image,
    instruction: str,
    size: str = MODEL_CONFIG.default_size,
    ratio: str = MODEL_CONFIG.default_ratio,
    save_dir: str = DEFAULT_SAVE_DIR,
) -> list[str]:
    """
    图生图修改 / 删除图中文字：参考一张含文字的图，按指令替换或删除文字

    说明：
    - 参考图支持公网 URL 字符串、本地文件路径字符串，或二者组成的列表
    - 提示词需同时指定「原文字 -> 新文字 / 删除文字」与「保持主体 / 版式 / 画风」
      （可用 prompt_utils.build_text_replace_prompt / build_text_remove_prompt 拼装）
    - 若服务端图生图走 /v1/images/edits 端点，可将下方 generate 调用
      替换为 client.images.edit(...)，请求体形态保持一致

    Args:
        client: Agnes OpenAI 兼容客户端
        image: 参考图（建议为含文字的图），公网 URL 字符串、本地文件路径字符串，或二者组成的列表
        instruction: 文字修改指令提示词
        size: 分辨率档位，默认 "2K"
        ratio: 宽高比，默认 "1:1"
        save_dir: 保存目录，默认 "output"

    Returns:
        保存到本地的图片文件路径列表

    Raises:
        AgnesConfigError: size / ratio 不合法
        AgnesImageError: 本地参考图文件不存在
        AgnesAPIError: API 调用失败（429/5xx 已自动指数退避重试）
    """
    validate_size(size)
    validate_ratio(ratio)

    # 参考图归一化：本地路径转 Data URI，URL 原样透传
    image_arg = _normalize_images(image)

    response = client.images.generate(
        model=MODEL_CONFIG.model_id,
        prompt=instruction,
        size=size,
        n=1,
        extra_body={"image": image_arg, "ratio": ratio},
    )
    return save_generated_images(response, save_dir=save_dir)
