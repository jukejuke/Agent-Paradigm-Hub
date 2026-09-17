"""
Agnes Video 2.5 Flash 文生视频示例
==================================

演示通过 Agnes 视频异步任务接口调用 agnes-video-2.5-flash 完成文生视频，
包含输入提示词、参数说明与预期输出（本地保存路径）。

运行方式（在演示项目根目录执行）：
    python examples/example_text_to_video.py

环境变量：
    AGNES_API_KEY     # Agnes API Key（https://agnes-ai.com 控制台获取）
    AGNES_BASE_URL    # 可选，默认 https://apihub.agnes-ai.com/v1

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
from src.generate import generate_video_from_text
from src.logger import setup_logger
from utils.errors import AgnesError
from utils.progress import ProgressBar

# 加载项目 .env 中的环境变量
load_dotenv()

logger = setup_logger()


def main():
    """演示 Agnes Video 2.5 Flash 文生视频能力"""
    # 1. 校验 API Key 是否已配置
    if not os.getenv(MODEL_CONFIG.api_key_env):
        logger.error("未找到 %s 环境变量，请将 .env.example 复制为 .env 并填入 API Key",
                     MODEL_CONFIG.api_key_env)
        return

    # 复用同一客户端实例，避免重复建连
    client = create_client()

    # 2. 文生视频：赛博风格动态场景
    try:
        logger.info("=== 文生视频示例 ===")
        prompt = "一只赛博小狗在赛博坦星球上激烈战斗，激光与火花四溅，电影级镜头，镜头缓缓推进"
        logger.info("输入提示词：%s", prompt)

        bar = ProgressBar()
        path = generate_video_from_text(
            client,
            prompt=prompt,
            seconds="5",
            size="720P",
            aspect_ratio="16:9",
            on_progress=bar.update,
        )
        bar.finish()
        logger.info("视频生成完成并保存：%s", path)
    except AgnesError as e:
        logger.error("文生视频失败：%s", e)

    logger.info("演示结束，所有结果已保存到 outputs/ 目录。")


if __name__ == "__main__":
    main()
