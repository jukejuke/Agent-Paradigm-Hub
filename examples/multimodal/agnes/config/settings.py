"""
Agnes 模型配置目录
==================

集中管理 agnes-image-2.5-flash 的模型 ID、API 地址、分辨率档位、
宽高比与版本信息，并提供参数校验函数。

作者: yaosh
日期: 2026-09-16
"""

import os
from dataclasses import dataclass

from examples.multimodal.agnes.utils.errors import AgnesConfigError


@dataclass(frozen=True)
class ModelConfig:
    """Agnes 图像模型的不可变配置数据类"""

    # 模型 ID
    model_id: str = "agnes-image-2.5-flash"
    # model_id: str = "agnes-image-2.0-flash"
    # OpenAI 兼容 API 地址（国际站节点；国内节点可配 https://apihub.agnes-ai.cn/v1）
    base_url: str = "https://apihub.agnes-ai.com/v1"
    # API Key 对应的环境变量名
    api_key_env: str = "AGNES_API_KEY"
    # 默认分辨率档位与宽高比
    default_size: str = "2K"
    default_ratio: str = "1:1"
    # 支持的分辨率档位：档位越高细节越好、耗时越长
    supported_sizes: tuple = ("1K", "2K", "3K", "4K")
    # 支持的宽高比
    supported_ratios: tuple = (
        "1:1", "3:4", "4:3", "16:9", "9:16", "2:3", "3:2", "21:9",
    )
    # 模型版本信息
    version: str = "2.5"
    release_date: str = "2026-07"


# 模块级默认配置实例（供全局复用）
MODEL_CONFIG = ModelConfig()


def get_model_config() -> ModelConfig:
    """
    获取模型配置：读取环境变量 AGNES_BASE_URL 覆盖默认 API 地址

    Returns:
        覆盖后的 ModelConfig 实例
    """
    base_url = os.getenv("AGNES_BASE_URL", MODEL_CONFIG.base_url)
    return ModelConfig(base_url=base_url)


def validate_size(size: str) -> str:
    """
    校验分辨率档位，非法时抛出中文配置错误

    Args:
        size: 分辨率档位，如 "1K" / "2K" / "3K" / "4K"

    Returns:
        原样返回合法的 size

    Raises:
        AgnesConfigError: size 不在 supported_sizes 中
    """
    if size not in MODEL_CONFIG.supported_sizes:
        raise AgnesConfigError(
            f"不支持的分辨率档位 {size!r}，可选值：{', '.join(MODEL_CONFIG.supported_sizes)}"
        )
    return size


def validate_ratio(ratio: str) -> str:
    """
    校验宽高比，非法时抛出中文配置错误

    Args:
        ratio: 宽高比，如 "1:1" / "16:9" / "9:16"

    Returns:
        原样返回合法的 ratio

    Raises:
        AgnesConfigError: ratio 不在 supported_ratios 中
    """
    if ratio not in MODEL_CONFIG.supported_ratios:
        raise AgnesConfigError(
            f"不支持的宽高比 {ratio!r}，可选值：{', '.join(MODEL_CONFIG.supported_ratios)}"
        )
    return ratio


