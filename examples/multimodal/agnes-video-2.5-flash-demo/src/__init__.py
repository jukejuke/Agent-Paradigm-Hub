"""
Agnes Video 2.5 Flash 示例项目 - 源代码目录
=============================================

集中导出客户端、生成函数与 CLI 入口，供 examples / cli 复用。

作者: Sol
日期: 2026-09-17
"""

from src.client import create_client, create_video_task, get_video_result, wait_for_video
from src.generate import (
    generate_video_from_image,
    generate_video_from_text,
    generate_video_with_style,
)

__all__ = [
    "create_client",
    "create_video_task",
    "get_video_result",
    "wait_for_video",
    "generate_video_from_text",
    "generate_video_from_image",
    "generate_video_with_style",
]
