"""
Agnes Video 2.5 Flash 图生视频示例
==================================

演示调用 agnes-video-2.5-flash 以静态图片为起点生成动态视频。
参考图支持公网 URL（推荐）或本地文件路径（自动转 Base64 Data URI）。

运行方式（在演示项目根目录执行）：
    python examples/example_image_to_video.py

参考图配置（二选一）：
    1. 环境变量 AGNES_VIDEO_IMAGE_URL=<公网图片URL>，或
    2. 修改下方 IMAGE_URL 常量

环境变量：
    AGNES_API_KEY     # Agnes API Key（https://agnes-ai.com 控制台获取）

作者: Sol
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
from src.generate import generate_video_from_image
from src.logger import setup_logger
from utils.errors import AgnesError
from utils.progress import ProgressBar

# 加载项目 .env 中的环境变量
load_dotenv()

logger = setup_logger()

# 默认参考图（可替换为你的图片公网 URL 或本地路径）
IMAGE_URL = "https://example.com/your-reference-image.png"


def main():
    """演示 Agnes Video 2.5 Flash 图生视频能力"""
    # 1. 校验 API Key 与参考图
    if not os.getenv(MODEL_CONFIG.api_key_env):
        logger.error("未找到 %s 环境变量，请将 .env.example 复制为 .env 并填入 API Key",
                     MODEL_CONFIG.api_key_env)
        return

    # 参考图来源：环境变量优先，其次使用脚本内常量
    image = os.getenv("AGNES_VIDEO_IMAGE_URL", IMAGE_URL)
    if not image or image.startswith("https://example.com/"):
        logger.error("未配置参考图。请通过环境变量 AGNES_VIDEO_IMAGE_URL "
                     "提供公网图片 URL，或修改本脚本 IMAGE_URL 常量后重试。")
        logger.info("用法示例：")
        logger.info("  set AGNES_VIDEO_IMAGE_URL=https://xxx.com/cat.png")
        logger.info("  python examples/example_image_to_video.py")
        return

    # 复用同一客户端实例
    client = create_client()

    # 2. 图生视频：静态图 → 动态视频
    try:
        logger.info("=== 图生视频示例 ===")
        logger.info("参考图：%s", image)
        prompt = "让画面中的猫咪缓缓抬头并眨眼，背景云彩轻轻飘动，镜头稳定"

        bar = ProgressBar()
        path = generate_video_from_image(
            client,
            image=image,
            prompt=prompt,
            seconds="5",
            size="720P",
            aspect_ratio="16:9",
            on_progress=bar.update,
        )
        bar.finish()
        logger.info("视频生成完成并保存：%s", path)
    except AgnesError as e:
        logger.error("图生视频失败：%s", e)

    logger.info("演示结束，所有结果已保存到 outputs/ 目录。")


if __name__ == "__main__":
    main()
