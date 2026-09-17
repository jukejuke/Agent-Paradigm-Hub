"""
Agnes 视频预览工具
==================

提供生成视频的预览能力：默认调用系统默认播放器（Windows 使用
os.startfile，其他平台使用浏览器打开 file:// 地址），也可生成
含 <video> 标签的本地 HTML 预览页。

作者: yaosh
日期: 2026-09-17
"""

import logging
import os
import sys
import webbrowser

# 复用全局日志器（未初始化时退化为标准输出）
logger = logging.getLogger("agnes-video")


def preview_video(path: str, open_player: bool = True, html: bool = False) -> None:
    """
    预览生成的视频文件

    Args:
        path: 本地视频文件路径
        open_player: 是否调用系统默认播放器打开，默认 True
        html: 是否同时生成 outputs/preview.html 预览页，默认 False
    """
    if not os.path.exists(path):
        logger.warning("预览失败：文件不存在 %s", path)
        return

    # 可选：生成 HTML 预览页（含 controls 播放器）
    if html:
        html_path = _write_preview_html(path)
        logger.info("已生成 HTML 预览页：%s", html_path)
        try:
            webbrowser.open("file://" + os.path.abspath(html_path).replace("\\", "/"))
            return
        except Exception as e:  # 打开失败不中断，继续尝试默认播放器
            logger.warning("打开 HTML 预览页失败：%s", e)

    # 默认：调用系统默认播放器
    if open_player:
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # noqa: S606  Windows 专用，打开默认播放器
            else:
                webbrowser.open("file://" + os.path.abspath(path))
            logger.info("已调用系统默认播放器打开：%s", path)
        except Exception as e:
            logger.warning("调用默认播放器失败：%s，可直接用播放器打开：%s", e, path)


def _write_preview_html(video_path: str) -> str:
    """
    在视频所在目录生成 preview.html 预览页

    Args:
        video_path: 本地视频文件路径

    Returns:
        HTML 文件路径
    """
    save_dir = os.path.dirname(os.path.abspath(video_path))
    html_path = os.path.join(save_dir, "preview.html")
    # 相对路径供 <video src> 引用，保证换目录后仍可打开
    rel = os.path.basename(video_path)
    content = (
        "<!DOCTYPE html>\n"
        '<html lang="zh-CN">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>Agnes Video 预览</title>\n"
        "<style>body{font-family:sans-serif;background:#1e1e2e;color:#eee;"
        "display:flex;justify-content:center;align-items:center;height:100vh;margin:0}"
        "video{max-width:90vw;max-height:85vh;border-radius:8px;"
        "box-shadow:0 8px 30px rgba(0,0,0,.5)}</style>\n"
        "</head>\n"
        "<body>\n"
        f'<video controls autoplay muted src="{rel}"></video>\n'
        "</body>\n"
        "</html>\n"
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)
    return html_path
