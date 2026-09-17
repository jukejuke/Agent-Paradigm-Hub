"""
Doubao Seedream 5.0-lite 图片风格修改示例（火山方舟 Agent Plan API）
====================================================================

本示例演示如何通过火山方舟「Agent Plan」的 OpenAI 兼容接口，
使用豆包图像生成模型 Doubao-Seedream-5.0-lite 完成「提示词驱动的图片风格迁移」：

1. 先用文生图生成一张基础参考图（本地保存，不依赖外部公网图片）
2. 通过不同的风格提示词，将参考图依次转换为多种风格
   （水墨画 / 赛博朋克 / 莫奈油画 / 3D 卡通渲染 等）
3. 结果自动保存到本地 output 目录

风格提示词写法要点（详见 STYLE_PROMPTS）：
- 同时约束「目标风格」+「保持主体与构图」+「画质词」，效果更稳定

模型 ID 对照：
- doubao-seedream-5-0-lite-260128  ：Seedream 5.0-lite（支持联网搜索）

运行方式（在仓库根目录执行）：
    python -m examples.multimodal.seedream.prompt_edit.style_edit

环境变量：
    AGENT_PLAN_API_KEY   # Agent Plan 专属 API Key（非方舟普通 API Key）

作者: Sol
日期: 2026-09-14
"""

import os
import sys

# 将项目根目录加入 sys.path，使本脚本可在任意工作目录运行
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
)

from dotenv import load_dotenv

# 复用基础示例（example.py）中的共享工具函数与常量，避免重复实现
from examples.multimodal.seedream.example import (
    DEFAULT_MODEL,
    DEFAULT_SAVE_DIR,
    _local_image_to_data_uri,
    create_client,
    save_generated_images,
    text_to_image,
)

# 加载根目录 .env 中的环境变量
load_dotenv()

# 基础参考图提示词：先用文生图生成参考图再风格迁移，
# 避免直接引用外部公网 URL（火山方舟服务端下载公网图可能超时）
BASE_IMAGE_PROMPT = "一只戴着墨镜的橘猫，坐在海边，日落，超写实"

# 多组风格迁移提示词：(风格名, 提示词)
# 提示词写法：目标风格 + 保持主体/构图 + 画质词
STYLE_PROMPTS: list[tuple[str, str]] = [
    (
        "水墨画",
        "将参考图转换为中国传统水墨画风格，保留主体与构图，宣纸质感，浓淡墨色晕染，留白意境，超高清",
    ),
    (
        "赛博朋克",
        "将参考图转换为赛博朋克风格，保留主体与构图，霓虹灯光，赛博城市夜景，高饱和色调，电影感，超高清",
    ),
    (
        "莫奈油画",
        "将参考图转换为莫奈印象派油画风格，保留主体与构图，柔和的笔触，光影斑驳，色彩明亮，超高清",
    ),
    (
        "3D 卡通",
        "将参考图转换为 3D 卡通渲染风格，保留主体与构图，皮克斯质感，圆润可爱，柔和光照，超高清",
    ),
]


def style_transfer(
    client,
    image,
    style_prompt: str,
    size: str = "2K",
    model: str = DEFAULT_MODEL,
    save_dir: str = DEFAULT_SAVE_DIR,
    watermark: bool = True,
) -> list[str]:
    """
    图片风格迁移：以参考图为输入，按风格提示词生成新风格图片

    说明：
    - 参考图支持公网 URL 字符串、本地文件路径字符串，或二者组成的列表
    - 本地文件路径会自动转换为 Base64 编码（data:image/<格式>;base64,<...>）
    - 单图传字符串，多图传列表
    - watermark 为 False 时不添加「AI 生成」水印

    Args:
        client: Agent Plan OpenAI 兼容客户端
        image: 参考图，公网 URL 字符串、本地文件路径字符串，或二者组成的列表
        style_prompt: 风格迁移提示词（中文优先）
        size: 分辨率档位，默认 "2K"
        model: 模型 ID，默认 doubao-seedream-5-0-lite-260128
        save_dir: 保存目录，默认 "output"
        watermark: 是否在图片右下角添加「AI 生成」水印，默认 True

    Returns:
        保存到本地的图片文件路径列表
    """
    # 归一化为列表，便于统一支持单图与多图两种输入
    images = image if isinstance(image, list) else [image]
    normalized = []
    for img in images:
        # 非 http/https 开头的字符串视为本地文件路径，转 Base64
        if isinstance(img, str) and not img.startswith(("http://", "https://")):
            normalized.append(_local_image_to_data_uri(img))
        else:
            normalized.append(img)
    # 单图传字符串，多图传列表
    image_arg = normalized if len(normalized) > 1 else normalized[0]

    response = client.images.generate(
        model=model,
        prompt=style_prompt,
        size=size,
        n=1,
        # openai SDK 3.x 不再提供 image 参数，图生图参考图需经 extra_body 透传到请求体
        extra_body={"image": image_arg, "watermark": watermark},
    )
    return save_generated_images(response, save_dir=save_dir)


def main():
    """演示基于提示词的图片风格迁移（图生图）"""
    # 1. 校验 Agent Plan 专属 API Key 是否已配置
    if not os.getenv("AGENT_PLAN_API_KEY"):
        print("提示：未找到 AGENT_PLAN_API_KEY 环境变量。")
        print("请将根目录 .env.example 复制为 .env，填入 Agent Plan 专属 API Key 后重试。")
        return

    client = create_client()

    # 2. 先文生图生成一张基础参考图（本地保存，避免外部公网 URL 不可达）
    try:
        print("\n=== 生成基础参考图（文生图） ===")
        ref_paths = text_to_image(client, prompt=BASE_IMAGE_PROMPT, watermark=False)
        print(f"生成并保存 {len(ref_paths)} 张图片：")
        for p in ref_paths:
            print(f"  - {p}")
    except Exception as e:
        print(f"[生成基础参考图失败] {e}")
        return

    ref_image = ref_paths[0]

    # 3. 对同一张参考图依次执行多组风格迁移
    print(f"\n参考图（本地生成）：{ref_image}")
    for style_name, style_prompt in STYLE_PROMPTS:
        try:
            print(f"\n=== 风格迁移：{style_name} ===")
            paths = style_transfer(client, ref_image, style_prompt, watermark=False)
            print(f"生成并保存 {len(paths)} 张图片：")
            for p in paths:
                print(f"  - {p}")
        except Exception as e:
            # 单组失败不中断演示，打印错误后继续下一组
            print(f"[{style_name} 风格迁移失败] {e}")

    print("\n演示结束，所有结果已保存到 output/ 目录。")


if __name__ == "__main__":
    main()
