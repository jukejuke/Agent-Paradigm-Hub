"""
Agnes 工具函数目录：错误处理 / 图片处理 / 提示词构建
"""
from examples.multimodal.agnes.utils.errors import (
    AgnesAPIError,
    AgnesConfigError,
    AgnesError,
    AgnesImageError,
    with_retry,
)
from examples.multimodal.agnes.utils.image_utils import (
    download_image,
    infer_mime,
    local_image_to_data_uri,
    validate_image_path,
)
from examples.multimodal.agnes.utils.prompt_utils import (
    build_style_prompt,
    build_text_add_prompt,
    build_text_remove_prompt,
    build_text_replace_prompt,
)

__all__ = [
    # 异常体系
    "AgnesError",
    "AgnesConfigError",
    "AgnesImageError",
    "AgnesAPIError",
    "with_retry",
    # 图片处理
    "infer_mime",
    "local_image_to_data_uri",
    "download_image",
    "validate_image_path",
    # 提示词构建
    "build_style_prompt",
    "build_text_add_prompt",
    "build_text_replace_prompt",
    "build_text_remove_prompt",
]
