"""
Agnes 视频生成模块（核心）
==========================

提供三个场景的视频生成函数：
- generate_video_from_text    ：文生视频（mode=text）
- generate_video_from_image   ：图生视频（mode=reference，images 参考图）
- generate_video_with_style   ：风格迁移视频（提示词控制风格，可选风格参考图）

所有函数遵循统一流程：组装参数 → 创建任务 → 轮询进度 → 下载保存。

依据官方文档（wiki.agnes-ai.com/docs/agnes-video-25-flash）：
- mode 必填：text / keyframe / reference
- size 固定 "720P"；seconds 为字符串 "4"~"12"；aspect_ratio 默认 16:9
- reference 模式下 images 最多 5 张，且必须是公网可访问的图片 URL

作者: yaosh
日期: 2026-09-17
"""

import base64
import logging
import os

import requests

from config.settings import (
    MODEL_CONFIG,
    validate_aspect_ratio,
    validate_mode,
    validate_seconds,
    validate_size,
)
from src.client import create_video_task, wait_for_video
from utils.downloader import parse_video_url, save_video
from utils.errors import AgnesVideoError

logger = logging.getLogger("agnes-video")

# 默认结果保存目录
DEFAULT_SAVE_DIR = "outputs"

# 默认生成模式：text = 文生视频
DEFAULT_MODE = "text"


def _build_payload(
    prompt: str,
    mode: str = DEFAULT_MODE,
    seconds: str = MODEL_CONFIG.seconds,
    size: str = MODEL_CONFIG.size,
    aspect_ratio: str = MODEL_CONFIG.aspect_ratio,
    seed: int | None = None,
    n: int = 1,
    first_frame: str | None = None,
    last_frame: str | None = None,
    images: list[str] | None = None,
) -> dict:
    """
    组装视频生成任务请求体

    Args:
        prompt: 视频内容描述提示词（reference 模式下用 <Picture N> 指代参考图）
        mode: 生成模式：text / keyframe / reference，默认 "text"
        seconds: 视频时长字符串 "4"~"12"，默认 "5"
        size: 分辨率档位，flash 模型固定 "720P"
        aspect_ratio: 宽高比，默认 "16:9"
        seed: 随机种子，固定值可复现结果；None 时不发送
        n: 输出数量，仅支持 1，默认 1
        first_frame: 首帧图片 URL（keyframe 模式）
        last_frame: 尾帧图片 URL（keyframe 模式）
        images: 参考图 URL 列表（reference 模式，最多 5 张）

    Returns:
        可提交给 POST /videos 的请求体字典
    """
    payload = {
        "model": MODEL_CONFIG.model_id,
        "prompt": prompt,
        "mode": mode,
        "seconds": seconds,
        "size": size,
        "aspect_ratio": aspect_ratio,
    }
    if seed is not None:
        payload["seed"] = seed
    if n != 1:
        payload["n"] = n
    if first_frame:
        payload["first_frame"] = first_frame
    if last_frame:
        payload["last_frame"] = last_frame
    if images:
        payload["images"] = images
    return payload


def _normalize_image(image: str) -> str:
    """
    将图生视频的输入图归一化为 API 可用形式

    - 公网 URL（http/https 开头）原样透传
    - 本地文件路径自动转换为 Base64 Data URI（官方要求公网 URL，
      本地文件建议先上传到公网后传入 URL）

    Args:
        image: 公网图片 URL 或本地图片文件路径

    Returns:
        归一化后的图片参数字符串

    Raises:
        AgnesVideoError: 本地图片文件不存在
    """
    if image.startswith(("http://", "https://")):
        return image
    if not os.path.exists(image):
        raise AgnesVideoError(f"本地图片文件不存在：{image}")
    with open(image, "rb") as f:
        data = f.read()
    ext = os.path.splitext(image)[1].lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        mime = "jpeg"
    elif ext == "webp":
        mime = "webp"
    else:
        mime = "png"
    logger.warning("本地图片已转 Base64 传入；官方要求公网可访问 URL，若生成失败请改用图片 URL")
    return f"data:image/{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def _generate_and_save(
    client: requests.Session,
    payload: dict,
    save_dir: str,
    filename: str | None,
    on_progress=None,
    poll_interval: int = MODEL_CONFIG.poll_interval,
    timeout: int = MODEL_CONFIG.timeout,
) -> str:
    """
    通用生成流程：创建任务 → 轮询 → 下载保存

    Args:
        client: create_client() 返回的会话
        payload: 视频生成任务请求体
        save_dir: 视频保存目录
        filename: 自定义文件名；None 时自动生成
        on_progress: 进度回调（入参 0-100）
        poll_interval: 轮询间隔（秒）
        timeout: 生成超时上限（秒）

    Returns:
        保存到本地的视频文件路径

    Raises:
        AgnesConfigError / AgnesAPIError / AgnesVideoError: 见各环节说明
    """
    # 1. 创建任务
    data = create_video_task(client, payload)
    task_id = data.get("task_id") or data.get("id")
    video_id = data.get("video_id")

    # 2. 轮询等待完成（on_progress 驱动进度条）
    result = wait_for_video(
        client,
        task_id,
        video_id=video_id,
        poll_interval=poll_interval,
        timeout=timeout,
        on_progress=on_progress,
    )

    # 3. 解析下载地址（完成态位于 metadata.url）并保存
    video_url = parse_video_url(result)
    logger.info("开始下载视频：%s", video_url)
    return save_video(video_url, save_dir=save_dir, filename=filename)


def _validate_common(seconds: str, size: str, aspect_ratio: str, mode: str) -> None:
    """对公共参数做前置校验，尽早暴露非法配置"""
    validate_mode(mode)
    validate_seconds(seconds)
    validate_size(size)
    validate_aspect_ratio(aspect_ratio)


def generate_video_from_text(
    client: requests.Session,
    prompt: str,
    mode: str = DEFAULT_MODE,
    seconds: str = MODEL_CONFIG.seconds,
    size: str = MODEL_CONFIG.size,
    aspect_ratio: str = MODEL_CONFIG.aspect_ratio,
    seed: int | None = None,
    save_dir: str = DEFAULT_SAVE_DIR,
    filename: str | None = None,
    on_progress=None,
    poll_interval: int = MODEL_CONFIG.poll_interval,
    timeout: int = MODEL_CONFIG.timeout,
) -> str:
    """
    文生视频：根据文本提示词直接生成视频（mode=text）

    说明：
    - mode 必须为 "text"，不允许携带 first_frame / last_frame / images 等媒体字段
    - 提示词建议包含：主体、动作、镜头运动、场景氛围（见 README 提示词指南）
    - 时长通过 seconds（"4"~"12"）控制，分辨率固定 720P

    Args:
        client: create_client() 返回的会话
        prompt: 视频内容描述提示词（必填，中英文均可）
        mode: 生成模式，默认 "text"
        seconds: 视频时长字符串 "4"~"12"，默认 "5"
        size: 分辨率档位，默认 "720P"
        aspect_ratio: 宽高比，默认 "16:9"
        seed: 随机种子，固定值可复现结果
        save_dir: 保存目录，默认 "outputs"
        filename: 自定义文件名；None 时自动生成
        on_progress: 进度回调（入参 0-100）
        poll_interval: 轮询间隔（秒），默认 3
        timeout: 生成超时上限（秒），默认 1800

    Returns:
        保存到本地的视频文件路径

    Raises:
        AgnesConfigError: seconds / size / aspect_ratio / mode 不合法
        AgnesAPIError: API 调用失败（429/5xx 已自动指数退避重试）
        AgnesVideoError: 任务失败、超时或下载失败
    """
    _validate_common(seconds, size, aspect_ratio, mode)

    payload = _build_payload(
        prompt=prompt,
        mode=mode,
        seconds=seconds,
        size=size,
        aspect_ratio=aspect_ratio,
        seed=seed,
    )
    logger.info("开始文生视频：%s", prompt)
    return _generate_and_save(
        client, payload, save_dir, filename, on_progress, poll_interval, timeout,
    )


def generate_video_from_image(
    client: requests.Session,
    image: str,
    prompt: str = "",
    mode: str = "reference",
    seconds: str = MODEL_CONFIG.seconds,
    size: str = MODEL_CONFIG.size,
    aspect_ratio: str = MODEL_CONFIG.aspect_ratio,
    seed: int | None = None,
    save_dir: str = DEFAULT_SAVE_DIR,
    filename: str | None = None,
    on_progress=None,
    poll_interval: int = MODEL_CONFIG.poll_interval,
    timeout: int = MODEL_CONFIG.timeout,
) -> str:
    """
    图生视频：以参考图为依据生成动态视频（mode=reference）

    说明：
    - image 支持公网 URL（推荐，官方要求公网可访问）或本地路径（自动转 Data URI）
    - prompt 内使用 <Picture 1> 指代参考图；留空时自动生成引用描述
    - reference 模式 images 最多 5 张（本函数单图调用）

    Args:
        client: create_client() 返回的会话
        image: 参考图，公网 URL 或本地图片文件路径（必填）
        prompt: 视频内容描述（可用 <Picture 1> 指代参考图），可留空
        mode: 生成模式，默认 "reference"
        seconds: 视频时长字符串 "4"~"12"，默认 "5"
        size: 分辨率档位，默认 "720P"
        aspect_ratio: 宽高比，默认 "16:9"
        seed: 随机种子
        save_dir: 保存目录，默认 "outputs"
        filename: 自定义文件名；None 时自动生成
        on_progress: 进度回调（入参 0-100）
        poll_interval: 轮询间隔（秒），默认 3
        timeout: 生成超时上限（秒），默认 1800

    Returns:
        保存到本地的视频文件路径

    Raises:
        AgnesConfigError: 参数不合法
        AgnesVideoError: 本地图片不存在、任务失败或下载失败
        AgnesAPIError: API 调用失败
    """
    _validate_common(seconds, size, aspect_ratio, mode)

    image_arg = _normalize_image(image)
    # 无提示词时生成引用式描述，使参考图参与生成
    prompt_arg = prompt or "以 <Picture 1> 为主体，保持主体与风格一致，画面自然动起来"

    payload = _build_payload(
        prompt=prompt_arg,
        mode=mode,
        seconds=seconds,
        size=size,
        aspect_ratio=aspect_ratio,
        seed=seed,
        images=[image_arg],
    )
    logger.info("开始图生视频：image=%s mode=%s", image, mode)
    return _generate_and_save(
        client, payload, save_dir, filename, on_progress, poll_interval, timeout,
    )


def generate_video_with_style(
    client: requests.Session,
    style: str,
    prompt: str = "",
    image: str | None = None,
    mode: str | None = None,
    seconds: str = MODEL_CONFIG.seconds,
    size: str = MODEL_CONFIG.size,
    aspect_ratio: str = MODEL_CONFIG.aspect_ratio,
    seed: int | None = None,
    save_dir: str = DEFAULT_SAVE_DIR,
    filename: str | None = None,
    on_progress=None,
    poll_interval: int = MODEL_CONFIG.poll_interval,
    timeout: int = MODEL_CONFIG.timeout,
) -> str:
    """
    风格迁移视频：将目标艺术风格应用到视频内容

    说明：
    - style 描述目标风格（如 水墨画 / 赛博朋克 / 吉卜力动画 等），
      内部会拼装为「以<风格>的风格呈现，<动态描述>」的提示词
    - 无参考图时使用 mode=text 纯提示词控制风格（兼容性最好）
    - 提供 image 时自动切换 mode=reference，以 <Picture 1> 作为风格参考

    Args:
        client: create_client() 返回的会话
        style: 目标艺术风格描述（必填）
        prompt: 视频内容动态描述（如 镜头缓缓推进），可留空
        image: 可选风格参考图（公网 URL 或本地路径）
        mode: 生成模式；None 时按是否有参考图自动选择 text / reference
        seconds: 视频时长字符串 "4"~"12"，默认 "5"
        size: 分辨率档位，默认 "720P"
        aspect_ratio: 宽高比，默认 "16:9"
        seed: 随机种子
        save_dir: 保存目录，默认 "outputs"
        filename: 自定义文件名；None 时自动生成
        on_progress: 进度回调（入参 0-100）
        poll_interval: 轮询间隔（秒），默认 3
        timeout: 生成超时上限（秒），默认 1800

    Returns:
        保存到本地的视频文件路径

    Raises:
        AgnesConfigError: 参数不合法
        AgnesVideoError: 本地图片不存在、任务失败或下载失败
        AgnesAPIError: API 调用失败
    """
    # 自动选择模式：有参考图 → reference，否则 text
    effective_mode = mode or ("reference" if image else DEFAULT_MODE)
    _validate_common(seconds, size, aspect_ratio, effective_mode)

    # 拼装风格提示词：目标风格 + 动态描述
    if image:
        style_prompt = (
            f"以 <Picture 1> 的风格呈现：{style}。"
            f"{prompt or '画面流畅自然、细节丰富，保持参考图风格一致'}"
        )
    else:
        style_prompt = f"以{style}的风格呈现，{prompt or '画面流畅自然、细节丰富'}"

    image_arg = _normalize_image(image) if image else None

    payload = _build_payload(
        prompt=style_prompt,
        mode=effective_mode,
        seconds=seconds,
        size=size,
        aspect_ratio=aspect_ratio,
        seed=seed,
        images=[image_arg] if image_arg else None,
    )
    logger.info("开始风格迁移视频：style=%s mode=%s", style, effective_mode)
    return _generate_and_save(
        client, payload, save_dir, filename, on_progress, poll_interval, timeout,
    )
