"""
Agnes Video 2.5 Flash 风格迁移视频示例
======================================

演示调用 agnes-video-2.5-flash 将目标艺术风格应用到视频内容，
支持可选风格参考图（公网 URL 或本地路径）。

运行方式（在演示项目根目录执行）：
    python examples/example_style_transfer.py

环境变量：
    AGNES_API_KEY     # Agnes API Key（https://agnes-ai.com 控制台获取）
    AGNES_VIDEO_IMAGE_URL  # 可选风格参考图 URL，不配置时仅靠提示词控制风格

作者: yaosh
日期: 2026-09-17
"""

import os
import sys

# 将演示项目根目录加入 sys.path，保证可在任意工作目录运行
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from dotenv import load_dotenv

from config.settings import MODEL_CONFIG
from src.client import create_client
from src.generate import generate_video_with_style
from src.logger import setup_logger
from utils.errors import AgnesError
from utils.progress import ProgressBar

# 加载项目 .env 中的环境变量
load_dotenv()

logger = setup_logger()


def main():
    """演示 Agnes Video 2.5 Flash 风格迁移视频能力"""
    # 1. 校验 API Key
    if not os.getenv(MODEL_CONFIG.api_key_env):
        logger.error("未找到 %s 环境变量，请将 .env.example 复制为 .env 并填入 API Key",
                     MODEL_CONFIG.api_key_env)
        return

    # 复用同一客户端实例
    client = create_client()

    # 可选：风格参考图（环境变量 AGNES_VIDEO_IMAGE_URL 或 None）
    style_image = os.getenv("AGNES_VIDEO_IMAGE_URL") or None

    # 2. 风格迁移视频：水墨国风
    try:
        logger.info("=== 风格迁移视频示例（中国水墨画） ===")
        style = "中国传统水墨画，宣纸质感，浓淡墨色晕染，留白意境"
        prompt = "一只仙鹤在云雾缭绕的山水间展翅飞过，镜头缓缓平移，意境悠远"
        logger.info("目标风格：%s", style)
        if style_image:
            logger.info("风格参考图：%s", style_image)

        bar = ProgressBar()
        path = generate_video_with_style(
            client,
            style=style,
            prompt=prompt,
            image=style_image,
            seconds="5",
            size="720P",
            aspect_ratio="16:9",
            on_progress=bar.update,
        )
        bar.finish()
        logger.info("视频生成完成并保存：%s", path)
    except AgnesError as e:
        logger.error("风格迁移视频失败：%s", e)

    logger.info("演示结束，所有结果已保存到 outputs/ 目录。")


if __name__ == "__main__":
    main()
