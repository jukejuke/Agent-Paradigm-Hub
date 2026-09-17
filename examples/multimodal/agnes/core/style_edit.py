"""
Agnes 图片样式修改模块（图生图风格迁移）
========================================

根据用户提供的文本提示词，对现有图片进行样式调整（色彩 / 风格 / 构图等），
同时保持主体与构图的一致性。

作者: yaosh
日期: 2026-09-16
"""

from examples.multimodal.agnes.config import MODEL_CONFIG, validate_ratio, validate_size
from examples.multimodal.agnes.core.base import DEFAULT_SAVE_DIR, _normalize_images, save_generated_images
from examples.multimodal.agnes.utils.errors import with_retry


@with_retry()
def style_transfer(
    client,
    image,
    style_prompt: str,
    size: str = MODEL_CONFIG.default_size,
    ratio: str = MODEL_CONFIG.default_ratio,
    save_dir: str = DEFAULT_SAVE_DIR,
) -> list[str]:
    """
    图片样式修改：以参考图为输入，按风格提示词生成新风格图片

    说明：
    - 参考图支持公网 URL 字符串、本地文件路径字符串，或二者组成的列表
    - 本地文件路径自动转换为 Base64 编码（data:image/<格式>;base64,<...>）
    - 提示词写法：目标风格 + 保持主体 / 构图 + 画质词
      （可用 prompt_utils.build_style_prompt 统一拼装）
    - 若服务端图生图走 /v1/images/edits 端点，可将下方 generate 调用
      替换为 client.images.edit(...)，请求体形态保持一致

    Args:
        client: Agnes OpenAI 兼容客户端
        image: 参考图，公网 URL 字符串、本地文件路径字符串，或二者组成的列表
        style_prompt: 风格迁移提示词（中文优先）
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
        prompt=style_prompt,
        size=size,
        n=1,
        # openai SDK 3.x 不识别 image / ratio 参数，需经 extra_body 透传到请求体
        extra_body={"image": image_arg, "ratio": ratio},
    )
    return save_generated_images(response, save_dir=save_dir)
