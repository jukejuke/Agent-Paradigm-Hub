"""
Agnes Video 2.5 Flash 命令行接口
================================

通过命令行参数指定输入内容与输出路径，支持三种视频生成场景：

    python src/cli.py --mode text  --prompt "一只赛博小狗在赛博坦星球战斗"
    python src/cli.py --mode image --image https://xxx.com/ref.png
    python src/cli.py --mode style --style "中国水墨画" --prompt "镜头缓缓推进"

运行方式（在演示项目根目录执行）：
    python src/cli.py --help

环境变量：
    AGNES_API_KEY     # Agnes API Key（必填，https://agnes-ai.com 控制台获取）
    AGNES_BASE_URL    # 可选，默认 https://apihub.agnes-ai.com/v1
    AGNES_VIDEO_MODEL # 可选，默认 agnes-video-2.5-flash

作者: yaosh
日期: 2026-09-17
"""

import argparse
import os
import sys

# 将演示项目根目录加入 sys.path，保证可在任意工作目录运行
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from dotenv import load_dotenv

from config.settings import (
    MODEL_CONFIG,
    validate_aspect_ratio,
    validate_seconds,
    validate_size,
)
from src.client import create_client
from src.generate import (
    DEFAULT_SAVE_DIR,
    generate_video_from_image,
    generate_video_from_text,
    generate_video_with_style,
)
from src.logger import setup_logger
from utils.errors import AgnesError
from utils.preview import preview_video
from utils.progress import ProgressBar

# 加载项目 .env 中的环境变量
load_dotenv()

logger = setup_logger()


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器（中文帮助）"""
    parser = argparse.ArgumentParser(
        description="Agnes Video 2.5 Flash 视频生成示例（文生视频 / 图生视频 / 风格迁移）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["text", "image", "style"],
        help="生成场景：text=文生视频，image=图生视频，style=风格迁移",
    )
    parser.add_argument(
        "--api-mode",
        default=None,
        choices=["text", "keyframe", "reference"],
        help="API 生成模式（高级），默认按场景自动选择：text→text，image→reference，"
             "style 无参考图→text / 有参考图→reference",
    )
    parser.add_argument("--prompt", default="", help="视频内容描述提示词（text 必填，其余可选）")
    parser.add_argument("--image", default=None, help="参考图：公网 URL 或本地路径（image 模式必填）")
    parser.add_argument("--style", default="", help="目标艺术风格描述（style 模式必填，如：中国水墨画）")
    parser.add_argument("--output", default=DEFAULT_SAVE_DIR, help="视频保存目录")
    parser.add_argument("--seconds", default=MODEL_CONFIG.seconds,
                        help="视频时长（字符串 4~12），默认 5")
    parser.add_argument("--size", default=MODEL_CONFIG.size, choices=list(MODEL_CONFIG.sizes),
                        help="分辨率档位，flash 模型固定 720P")
    parser.add_argument("--aspect-ratio", default=MODEL_CONFIG.aspect_ratio,
                        choices=list(MODEL_CONFIG.aspect_ratios),
                        help="宽高比，默认 16:9")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，固定值可复现结果")
    parser.add_argument("--poll-interval", type=int, default=MODEL_CONFIG.poll_interval,
                        help="任务轮询间隔（秒）")
    parser.add_argument("--timeout", type=int, default=MODEL_CONFIG.timeout,
                        help="生成超时上限（秒）")
    parser.add_argument("--no-preview", action="store_true", help="生成后不自动预览")
    parser.add_argument("--html-preview", action="store_true", help="以 HTML 预览页方式预览")
    return parser


def _resolve_api_mode(scene: str, api_mode: str | None, image: str | None) -> str:
    """
    根据场景与显式参数确定 API 生成模式

    Args:
        scene: 场景 text / image / style
        api_mode: 用户显式指定的 API 模式（可为 None）
        image: 是否携带参考图

    Returns:
        text / keyframe / reference 之一
    """
    if api_mode:
        return api_mode
    if scene == "image":
        return "reference"
    if scene == "style" and image:
        return "reference"
    return "text"


def main(argv: list[str] | None = None) -> int:
    """
    CLI 主流程：校验参数 → 生成视频 → 展示进度 → 保存 → 预览

    Args:
        argv: 命令行参数列表；None 时使用 sys.argv[1:]

    Returns:
        进程退出码（0 成功，1 失败）
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # 前置校验：API Key 与必填参数
    if not os.getenv(MODEL_CONFIG.api_key_env):
        logger.error("未找到 %s 环境变量，请将 .env.example 复制为 .env 并填入 API Key",
                     MODEL_CONFIG.api_key_env)
        return 1
    if args.mode == "text" and not args.prompt:
        parser.error("text 模式必须提供 --prompt")
    if args.mode == "image" and not args.image:
        parser.error("image 模式必须提供 --image")
    if args.mode == "style" and not args.style:
        parser.error("style 模式必须提供 --style")

    # 参数合法性校验（尽早暴露非法配置）
    try:
        validate_seconds(args.seconds)
        validate_size(args.size)
        validate_aspect_ratio(args.aspect_ratio)
    except AgnesError as e:
        logger.error("参数校验失败：%s", e)
        return 1

    # 复用同一客户端实例
    try:
        client = create_client()
    except AgnesError as e:
        logger.error("%s", e)
        return 1

    # 确定 API 生成模式
    api_mode = _resolve_api_mode(args.mode, args.api_mode, args.image)

    # 进度条：轮询时实时刷新
    bar = ProgressBar()

    try:
        if args.mode == "text":
            path = generate_video_from_text(
                client,
                prompt=args.prompt,
                mode=api_mode,
                seconds=args.seconds,
                size=args.size,
                aspect_ratio=args.aspect_ratio,
                seed=args.seed,
                save_dir=args.output,
                on_progress=bar.update,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
            )
        elif args.mode == "image":
            path = generate_video_from_image(
                client,
                image=args.image,
                prompt=args.prompt,
                mode=api_mode,
                seconds=args.seconds,
                size=args.size,
                aspect_ratio=args.aspect_ratio,
                seed=args.seed,
                save_dir=args.output,
                on_progress=bar.update,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
            )
        else:  # style
            path = generate_video_with_style(
                client,
                style=args.style,
                prompt=args.prompt,
                image=args.image,
                mode=api_mode,
                seconds=args.seconds,
                size=args.size,
                aspect_ratio=args.aspect_ratio,
                seed=args.seed,
                save_dir=args.output,
                on_progress=bar.update,
                poll_interval=args.poll_interval,
                timeout=args.timeout,
            )
    except AgnesError as e:
        bar.finish()
        logger.error("视频生成失败：%s", e)
        return 1

    bar.finish()
    logger.info("视频已保存：%s", path)

    # 预览功能：默认打开系统播放器，--html-preview 生成 HTML 预览页
    if not args.no_preview:
        preview_video(path, open_player=True, html=args.html_preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
