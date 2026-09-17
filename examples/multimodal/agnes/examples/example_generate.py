"""
Agnes Image 2.5 Flash 文生图示例
================================

演示通过 Agnes 的 OpenAI 兼容接口调用 agnes-image-2.5-flash 完成文生图，
包含输入提示词、参数说明与预期输出（本地保存路径）。

运行方式（在仓库根目录执行）：
    python -m examples.multimodal.agnes.examples.example_generate

环境变量：
    AGNES_API_KEY   # Agnes API Key（https://agnes-ai.com 控制台获取）
    AGNES_BASE_URL  # 可选，默认 https://apihub.agnes-ai.com/v1

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
from examples.multimodal.agnes.core import create_client, generate_image
from examples.multimodal.agnes.utils import AgnesError

# 加载根目录 .env 中的环境变量
load_dotenv()


def main():
    """演示 Agnes Image 2.5 Flash 文生图能力"""
    # 1. 校验 API Key 是否已配置
    if not os.getenv(MODEL_CONFIG.api_key_env):
        print(f"提示：未找到 {MODEL_CONFIG.api_key_env} 环境变量。")
        print("请将根目录 .env.example 复制为 .env，填入 Agnes API Key 后重试。")
        return

    # 复用同一客户端实例，避免重复建连（性能优化）
    client = create_client()

    # 2. 文生图：写实风（1:1 方形构图）
    try:
        print("\n=== 1. 文生图（写实风，1:1） ===")
        prompt = "一只戴着墨镜的橘猫，坐在海边，日落，超写实"
        print(f"输入提示词：{prompt}")
        paths = generate_image(client, prompt=prompt, size="2K", ratio="1:1")
        print(f"生成并保存 {len(paths)} 张图片：")
        for p in paths:
            print(f"  - {p}")
    except AgnesError as e:
        print(f"[文生图失败] {e}")

    # 3. 文生图：国潮海报（16:9 横版构图）
    try:
        print("\n=== 2. 文生图（国潮海报，16:9） ===")
        prompt = "中秋节国潮插画海报，圆月与玉兔，暖色调，高饱和，画面干净"
        print(f"输入提示词：{prompt}")
        paths = generate_image(client, prompt=prompt, size="2K", ratio="16:9")
        print(f"生成并保存 {len(paths)} 张图片：")
        for p in paths:
            print(f"  - {p}")
    except AgnesError as e:
        print(f"[文生图失败] {e}")

    print("\n演示结束，所有结果已保存到 output/ 目录。")


if __name__ == "__main__":
    main()
