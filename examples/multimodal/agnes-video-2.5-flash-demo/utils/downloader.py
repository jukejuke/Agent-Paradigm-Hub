"""
Agnes 视频下载与保存工具
========================

提供从任务结果中解析视频 URL、以及将视频流式下载到本地 outputs 目录的能力。

作者: Sol
日期: 2026-09-17
"""

import os
import time
import uuid

import requests

from utils.errors import AgnesVideoError

# 结果中可能包含视频 URL 的字段名（按优先级尝试）
_VIDEO_URL_KEYS = ("video_url", "output_url", "download_url", "file_url", "url", "output", "video")


def parse_video_url(result_json: dict) -> str:
    """
    从任务结果 JSON 中提取视频下载地址

    兼容字段：video_url / output_url / download_url / file_url / url /
    output / video，以及嵌套在 data / metadata 下的同名字段。
    agnes-video-2.5-flash 完成态的视频地址位于 metadata.url。

    Args:
        result_json: GET 查询任务返回的 JSON 字典

    Returns:
        视频公网下载地址

    Raises:
        AgnesVideoError: 未找到可用的视频 URL 字段
    """
    # 扁平化搜索：优先顶层字段，其次 data / metadata 嵌套
    candidates = []
    candidates.extend(result_json.get(k) for k in _VIDEO_URL_KEYS if result_json.get(k))
    for container_key in ("data", "metadata"):
        container = result_json.get(container_key)
        if isinstance(container, dict):
            candidates.extend(container.get(k) for k in _VIDEO_URL_KEYS if container.get(k))
        elif isinstance(container, list) and container:
            first = container[0]
            if isinstance(first, dict):
                candidates.extend(first.get(k) for k in _VIDEO_URL_KEYS if first.get(k))

    for url in candidates:
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return url
    raise AgnesVideoError(
        f"任务结果中未找到视频下载地址，返回字段：{list(result_json.keys())}"
    )


def save_video(video_url: str, save_dir: str = "outputs", filename: str | None = None) -> str:
    """
    流式下载视频到本地保存目录

    Args:
        video_url: 视频公网下载地址
        save_dir: 保存目录，默认 "outputs"（自动创建）
        filename: 自定义文件名；为 None 时按 <uuid8>_<时间戳>.mp4 自动命名

    Returns:
        保存到本地的视频文件绝对路径

    Raises:
        AgnesVideoError: 下载失败或文件为空
    """
    os.makedirs(save_dir, exist_ok=True)
    if not filename:
        filename = f"{uuid.uuid4().hex[:8]}_{int(time.time())}.mp4"
    file_path = os.path.join(save_dir, filename)

    try:
        # 流式下载，避免大文件一次性占用内存
        with requests.get(video_url, stream=True, timeout=(10, 120)) as resp:
            resp.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
    except requests.RequestException as e:
        raise AgnesVideoError(f"下载视频失败：{video_url}，原因：{e}") from e

    # 校验文件非空，避免保存 0 字节的无效文件
    if os.path.getsize(file_path) == 0:
        os.remove(file_path)
        raise AgnesVideoError(f"下载的视频文件为空，已删除：{file_path}")
    return os.path.abspath(file_path)
