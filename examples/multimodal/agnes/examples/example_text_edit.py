"""
Agnes Image 2.5 Flash 文字编辑示例
==================================

演示 agnes-image-2.5-flash 的文字渲染 / 修改能力，覆盖三个场景：
1. 文生图带文字（在图片中添加文字）：海报标题，支持中英文
2. 图生图改字：把图中「中秋快乐」替换为「新年快乐」，保持主体、版式与画风
3. 图生图删字：删除图中指定文字，用自然内容填补原文字区域

运行方式（在仓库根目录执行）：
    python -m examples.multimodal.agnes.examples.example_text_edit

环境变量：
    AGNES_API_KEY   # Agnes API Key

作者: yaosh
日期: 2026-09-16
"""

import os
import sys

from dotenv import load_dotenv

# 将项目根目录加入 sys.path，使本脚本可在任意工作目录运行
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
)

from examples.multimodal.agnes.config import MODEL_CONFIG
from examples.multimodal.agnes.core import (
    create_client,
    edit_image_text,
    generate_image_with_text,
)
from examples.multimodal.agnes.utils import (
    AgnesError,
    build_text_add_prompt,
    build_text_remove_prompt,
    build_text_replace_prompt,
)

# 加载根目录 .env 中的环境变量
load_dotenv()

# 场景一：文生图带文字（海报标题）
POSTER_PROMPT = build_text_add_prompt(
    "中秋快乐",
    scene_hint="中秋节促销海报，居中大标题，背景为圆月与玉兔，国潮插画风格，暖色调，画面干净",
)

# 场景二：图生图改字（先把海报生成出来，再替换其中的标题文字）
REPLACE_PROMPT = build_text_replace_prompt("中秋快乐", "新年快乐")

# 场景三：图生图删字
REMOVE_PROMPT = build_text_remove_prompt("中秋快乐")


def main():
    """演示基于提示词的图片文字添加 / 修改 / 删除"""
    # 1. 校验 API Key 是否已配置
    if not os.getenv(MODEL_CONFIG.api_key_env):
        print(f"提示：未找到 {MODEL_CONFIG.api_key_env} 环境变量。")
        print("请将根目录 .env.example 复制为 .env，填入 Agnes API Key 后重试。")
        return

    client = create_client()

    # 2. 场景一：文生图生成带文字的图片（海报标题）
    poster_paths = []
    try:
        print("\n=== 场景一：文生图带文字（海报标题） ===")
        print(f"输入提示词：{POSTER_PROMPT}")
        poster_paths = generate_image_with_text(client, POSTER_PROMPT)
        print(f"生成并保存 {len(poster_paths)} 张图片：")
        for p in poster_paths:
            print(f"  - {p}")
    except AgnesError as e:
        # 单场景失败不中断演示，打印错误后继续
        print(f"[文生图带文字失败] {e}")

    # 3. 场景二：图生图修改图中文字（参考图优先使用刚生成的海报）
    try:
        print("\n=== 场景二：图生图改字（中秋快乐 -> 新年快乐） ===")
        if not poster_paths:
            raise AgnesError("场景一未生成海报，跳过改字（可替换为任意含文字的本地图片路径）")
        ref_image = poster_paths[0]
        print(f"参考图：{ref_image}")
        print(f"输入修改指令：{REPLACE_PROMPT}")
        paths = edit_image_text(client, ref_image, REPLACE_PROMPT)
        print(f"生成并保存 {len(paths)} 张图片：")
        for p in paths:
            print(f"  - {p}")
    except AgnesError as e:
        print(f"[图生图改字失败] {e}")

    # 4. 场景三：图生图删除图中文字
    try:
        print("\n=== 场景三：图生图删字（删除海报标题） ===")
        if not poster_paths:
            raise AgnesError("场景一未生成海报，跳过删字（可替换为任意含文字的本地图片路径）")
        ref_image = poster_paths[0]
        print(f"参考图：{ref_image}")
        print(f"输入删除指令：{REMOVE_PROMPT}")
        paths = edit_image_text(client, ref_image, REMOVE_PROMPT)
        print(f"生成并保存 {len(paths)} 张图片：")
        for p in paths:
            print(f"  - {p}")
    except AgnesError as e:
        print(f"[图生图删字失败] {e}")

    print("\n演示结束，所有结果已保存到 output/ 目录。")


if __name__ == "__main__":
    main()
