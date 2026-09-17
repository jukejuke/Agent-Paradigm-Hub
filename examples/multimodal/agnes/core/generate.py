"""
Agnes 图片生成模块（文生图）
============================

根据用户提供的文本提示词，调用 agnes-image-2.5-flash 生成符合要求的图片。

作者: yaosh
日期: 2026-09-16
"""

from examples.multimodal.agnes.config import MODEL_CONFIG, validate_ratio, validate_size
from examples.multimodal.agnes.core.base import DEFAULT_SAVE_DIR, save_generated_images
from examples.multimodal.agnes.utils.errors import with_retry


@with_retry()
def generate_image(
    client,
    prompt: str,
    size: str = MODEL_CONFIG.default_size,
    ratio: str = MODEL_CONFIG.default_ratio,
    n: int = 1,
    save_dir: str = DEFAULT_SAVE_DIR,
) -> list[str]:
    """
    文生图：根据文本提示词生成图片

    说明：
    - 分辨率档位 size 支持 1K / 2K / 3K / 4K，档位越高细节越好、耗时越长，
      生产环境按需求选档：快速预览用 1K，正式出图用 2K / 3K
    - 宽高比 ratio 支持 1:1 / 3:4 / 4:3 / 16:9 / 9:16 / 2:3 / 3:2 / 21:9
    - 批量出图可传 n>1，建议不超过 4，避免单次请求超时

    Args:
        client: Agnes OpenAI 兼容客户端（复用同一实例，避免重复建连）
        prompt: 图像描述提示词（中英文均可）
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
    # 前置参数校验，尽早暴露非法配置
    validate_size(size)
    validate_ratio(ratio)

    response = client.images.generate(
        model=MODEL_CONFIG.model_id,
        prompt=prompt,
        size=size,
        n=n,
        # openai SDK 3.x 不再识别 ratio 参数，需经 extra_body 透传到请求体
        extra_body={"ratio": ratio},
    )
    return save_generated_images(response, save_dir=save_dir)
