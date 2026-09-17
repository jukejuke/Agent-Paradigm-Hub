"""
Agnes Image 2.5 Flash 图片样式修改示例
======================================

演示基于提示词的图片风格迁移（图生图）：
1. 先用文生图生成一张基础参考图（本地保存，不依赖外部公网图片）
2. 通过不同的风格提示词，将参考图依次转换为多种风格
   （水墨画 / 赛博朋克 / 油画）
3. 结果自动保存到本地 output 目录

运行方式（在仓库根目录执行）：
    python -m examples.multimodal.agnes.examples.example_style_edit

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
from examples.multimodal.agnes.core import create_client, generate_image, style_transfer
from examples.multimodal.agnes.utils import AgnesError, build_style_prompt

# 加载根目录 .env 中的环境变量
load_dotenv()

# 基础参考图提示词：先用文生图生成参考图再风格迁移，
# 避免直接引用外部公网 URL（公网图可能不可达或超时）
BASE_IMAGE_PROMPT = "一只戴着墨镜的橘猫，坐在海边，日落，超写实"

# 多组风格迁移：(风格名, 风格描述)
# 提示词由 build_style_prompt 统一拼装：目标风格 + 保持主体/构图 + 画质词
STYLE_LIST: list[tuple[str, str]] = [
    ("水墨画", "中国传统水墨画，宣纸质感，浓淡墨色晕染，留白意境"),
    ("赛博朋克", "赛博朋克，霓虹灯光，赛博城市夜景，高饱和色调，电影感"),
    ("油画", "莫奈印象派油画，柔和笔触，光影斑驳，色彩明亮"),
]


def main():
    """演示基于提示词的图片风格迁移（图生图）"""
    # 1. 校验 API Key 是否已配置
    if not os.getenv(MODEL_CONFIG.api_key_env):
        print(f"提示：未找到 {MODEL_CONFIG.api_key_env} 环境变量。")
        print("请将根目录 .env.example 复制为 .env，填入 Agnes API Key 后重试。")
        return

    client = create_client()

    # 2. 先文生图生成基础参考图（本地保存，避免外部公网 URL 不可达）
    try:
        print("\n=== 生成基础参考图（文生图） ===")
        print(f"输入提示词：{BASE_IMAGE_PROMPT}")
        ref_paths = generate_image(client, prompt=BASE_IMAGE_PROMPT)
        print(f"生成并保存 {len(ref_paths)} 张图片：")
        for p in ref_paths:
            print(f"  - {p}")
    except AgnesError as e:
        print(f"[生成基础参考图失败] {e}")
        return

    ref_image = ref_paths[0]
    
    # ref_image = "output/f049ea07_1.png"

    # 3. 对同一张参考图依次执行多组风格迁移
    print(f"\n参考图（本地生成）：{ref_image}")
    for style_name, style_desc in STYLE_LIST:
        try:
            print(f"\n=== 风格迁移：{style_name} ===")
            style_prompt = build_style_prompt(style_desc)
            print(f"输入风格提示词：{style_prompt}")
            paths = style_transfer(client, ref_image, style_prompt)
            print(f"生成并保存 {len(paths)} 张图片：")
            for p in paths:
                print(f"  - {p}")
        except AgnesError as e:
            # 单组失败不中断演示，打印错误后继续下一组
            print(f"[{style_name} 风格迁移失败] {e}")

    print("\n演示结束，所有结果已保存到 output/ 目录。")


if __name__ == "__main__":
    main()
