"""
Agnes 视频模型配置
==================

集中管理 agnes-video-2.5-flash 的模型 ID、API 地址、默认视频参数
（时长 / 分辨率档位 / 宽高比 / 轮询间隔）、支持的档位与版本信息，
并提供参数校验函数。

依据官方文档（wiki.agnes-ai.com/docs/agnes-video-25-flash）：
- mode 必填：text（文生视频）/ keyframe（首尾帧控制）/ reference（参考图生成）
- size 固定 "720P"（字符串），其他值返回 HTTP 400
- seconds 为字符串 "4" ~ "12"，默认 "5"
- aspect_ratio 支持 21:9 / 16:9 / 4:3 / 1:1 / 3:4 / 9:16，默认 16:9
- 结果查询推荐 GET /agnesapi?video_id=<ID>&model_name=<模型ID>

作者: yaosh
日期: 2026-09-17
"""

import os
from dataclasses import dataclass, field

from utils.errors import AgnesConfigError


@dataclass(frozen=True)
class ModelConfig:
    """Agnes 视频模型的不可变配置数据类"""

    # 模型 ID（默认 agnes-video-2.5-flash；如账号暂不可用，可回退 agnes-video-v2.0）
    model_id: str = "agnes-video-2.5-flash"
    # OpenAI 兼容 API 地址（国际站节点；国内节点可配 https://api.agnes-ai.cn/v1）
    base_url: str = "https://apihub.agnes-ai.com/v1"
    # API Key 对应的环境变量名
    api_key_env: str = "AGNES_API_KEY"
    # 视频专用 API 地址环境变量名（优先于通用 AGNES_BASE_URL）
    base_url_env: str = "AGNES_VIDEO_BASE_URL"
    # 模型 ID 可覆盖的环境变量名
    model_env: str = "AGNES_VIDEO_MODEL"
    # 默认视频参数
    seconds: str = "5"           # 视频时长（秒），字符串 "4"~"12"，默认 "5"
    size: str = "720P"           # 分辨率档位，flash 模型固定 "720P"
    aspect_ratio: str = "16:9"   # 宽高比，默认 16:9
    poll_interval: int = 3       # 任务轮询间隔（秒），官方建议 1-2 秒
    timeout: int = 1800          # 生成超时上限（秒）
    # 支持的生成模式
    modes: tuple = ("text", "keyframe", "reference")
    # 支持的分辨率档位（flash 仅 720P；回退模型 agnes-video-v2.0 另有档位）
    sizes: tuple = ("720P",)
    # 支持的宽高比
    aspect_ratios: tuple = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
    # 模型版本信息
    version: str = "2.5"
    release_date: str = "2026-09"


# 模块级默认配置实例（供全局复用）
MODEL_CONFIG = ModelConfig()


def _normalize_base_url(base_url: str) -> str:
    """
    将环境变量中的 API 地址归一化为 OpenAI 兼容的根地址（形如 https://host/v1）

    兼容图片模块遗留的完整端点地址（如 https://api.agnes-ai.cn/v1/images/generations）：
    自动截取到 /v1 为止；不含 /v1 时自动补全。

    Args:
        base_url: 原始 API 地址

    Returns:
        归一化后的根地址
    """
    base_url = base_url.strip().rstrip("/")
    marker = "/v1"
    idx = base_url.find(marker)
    if idx != -1:
        return base_url[:idx + len(marker)]
    return base_url + marker


def get_model_config() -> ModelConfig:
    """
    获取模型配置：读取环境变量覆盖默认值

    优先级：AGNES_VIDEO_BASE_URL > AGNES_BASE_URL（自动归一化到 /v1 根地址）> 默认值；
    模型 ID 由 AGNES_VIDEO_MODEL 覆盖。

    Returns:
        覆盖后的 ModelConfig 实例
    """
    base_url = (
        os.getenv(MODEL_CONFIG.base_url_env)
        or os.getenv("AGNES_BASE_URL")
        or MODEL_CONFIG.base_url
    )
    base_url = _normalize_base_url(base_url)
    model_id = os.getenv(MODEL_CONFIG.model_env, MODEL_CONFIG.model_id)
    return ModelConfig(base_url=base_url, model_id=model_id)


def validate_mode(mode: str) -> str:
    """
    校验生成模式，非法时抛出中文配置错误

    Args:
        mode: 生成模式：text / keyframe / reference

    Returns:
        原样返回合法的 mode

    Raises:
        AgnesConfigError: mode 不在支持的范围内
    """
    if mode not in MODEL_CONFIG.modes:
        raise AgnesConfigError(
            f"不支持的生成模式 {mode!r}，可选值：{', '.join(MODEL_CONFIG.modes)}"
        )
    return mode


def validate_seconds(seconds: str) -> str:
    """
    校验视频时长（字符串 "4"~"12"），非法时抛出中文配置错误

    Args:
        seconds: 视频时长字符串，如 "5" / "8"

    Returns:
        原样返回合法的 seconds

    Raises:
        AgnesConfigError: 时长不在 4~12 秒范围内
    """
    try:
        value = int(seconds)
    except (TypeError, ValueError):
        raise AgnesConfigError(
            f"非法时长 {seconds!r}：需为 '4'~'12' 之间的字符串，如 '5'"
        ) from None
    if value < 4 or value > 12:
        raise AgnesConfigError(
            f"非法时长 {seconds!r}：支持范围为 4~12 秒，默认 '5'"
        )
    return seconds


def validate_size(size: str) -> str:
    """
    校验分辨率档位，非法时抛出中文配置错误

    Args:
        size: 分辨率档位，flash 模型仅支持 "720P"

    Returns:
        原样返回合法的 size

    Raises:
        AgnesConfigError: size 不在支持的档位中
    """
    if size not in MODEL_CONFIG.sizes:
        raise AgnesConfigError(
            f"不支持的分辨率档位 {size!r}，flash 模型仅支持 {', '.join(MODEL_CONFIG.sizes)}"
        )
    return size


def validate_aspect_ratio(ratio: str) -> str:
    """
    校验宽高比，非法时抛出中文配置错误

    Args:
        ratio: 宽高比，如 "16:9" / "9:16" / "1:1"

    Returns:
        原样返回合法的 ratio

    Raises:
        AgnesConfigError: ratio 不在支持的宽高比中
    """
    if ratio not in MODEL_CONFIG.aspect_ratios:
        raise AgnesConfigError(
            f"不支持的宽高比 {ratio!r}，可选值：{', '.join(MODEL_CONFIG.aspect_ratios)}"
        )
    return ratio
