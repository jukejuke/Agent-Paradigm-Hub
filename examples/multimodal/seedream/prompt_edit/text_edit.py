"""
Doubao Seedream 5.0-lite 图片文字修改示例（火山方舟 Agent Plan API）
===================================================================

本示例演示如何通过火山方舟「Agent Plan」的 OpenAI 兼容接口，
使用豆包图像生成模型 Doubao-Seedream-5.0-lite 完成「提示词驱动的文字渲染/修改」：

场景一 文生图带文字：直接在提示词中指定要渲染的文字内容，生成含文字的图片
        （海报标题、店招等，支持中英文）
场景二 图生图改字：参考一张含文字的图（海报/招牌），用提示词把原文字替换为新文字，
        保持主体、版式与画风一致

文字提示词写法要点：
- 把要渲染的文字内容显式写进提示词并用「」标注
- 强调「文字必须完全一致、无错别字」，可显著提升文字渲染准确率

模型 ID 对照：
- doubao-seedream-5-0-lite-260128  ：Seedream 5.0-lite（支持联网搜索）

运行方式（在仓库根目录执行）：
    python -m examples.multimodal.seedream.prompt_edit.text_edit

环境变量：
    AGENT_PLAN_API_KEY   # Agent Plan 专属 API Key（非方舟普通 API Key）

作者: yaosh
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
)

# 加载根目录 .env 中的环境变量
load_dotenv()

# 场景一：文生图带文字（海报标题，指定渲染文字并用「」标注）
POSTER_PROMPT = (
    "中秋节促销海报，居中大标题文字「中秋快乐」，副标题文字「千里共婵娟」，"
    "背景为圆月与玉兔，国潮插画风格，暖色调，画面干净，"
    "文字必须完全一致、无错别字"
)

# 场景一：文生图带文字（店招）
SHOP_SIGN_PROMPT = (
    "一家奶茶店的门头招牌，招牌上文字「奶茶研究所」，发光灯箱效果，"
    "深夜街景，电影感，文字必须完全一致、无错别字"
)

# 场景二：图生图改字提示词（先把海报生成出来，再替换其中的标题文字）
EDIT_TEXT_PROMPT = (
    "将海报中的大标题文字「中秋快乐」替换为「新年快乐」，"
    "副标题保持「千里共婵娟」不变，保持海报主体、版式与画风完全一致，"
    "新文字必须完全一致、无错别字"
)

# 回退参考图：公网 URL（海报生成失败时兜底使用），建议替换为含文字的本地图片路径
REFERENCE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/"
    "Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg"
)


def text_to_image_with_text(
    client,
    prompt: str,
    size: str = "2K",
    model: str = DEFAULT_MODEL,
    save_dir: str = DEFAULT_SAVE_DIR,
    watermark: bool = True,
) -> list[str]:
    """
    文生图生成带指定文字的图片

    说明：
    - 提示词需把要渲染的文字内容显式写出（可用「」标注），并强调文字一致性
    - 支持中英文文字渲染（Seedream 5.0-lite 特性）
    - watermark 为 False 时不添加「AI 生成」水印

    Args:
        client: Agent Plan OpenAI 兼容客户端
        prompt: 图像描述提示词（含要渲染的文字内容）
        size: 分辨率档位，默认 "2K"
        model: 模型 ID，默认 doubao-seedream-5-0-lite-260128
        save_dir: 保存目录，默认 "output"
        watermark: 是否在图片右下角添加「AI 生成」水印，默认 True

    Returns:
        保存到本地的图片文件路径列表
    """
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        n=1,
        extra_body={"watermark": watermark},
    )
    return save_generated_images(response, save_dir=save_dir)


def edit_image_text(
    client,
    image,
    prompt: str,
    size: str = "2K",
    model: str = DEFAULT_MODEL,
    save_dir: str = DEFAULT_SAVE_DIR,
    watermark: bool = True,
) -> list[str]:
    """
    图生图修改图中文字：参考一张含文字的图，用提示词把原文字替换为新文字

    说明：
    - 参考图支持公网 URL 字符串、本地文件路径字符串，或二者组成的列表
    - 本地文件路径会自动转换为 Base64 编码（data:image/<格式>;base64,<...>）
    - 提示词需同时指定「原文字 -> 新文字」与「保持主体/版式/画风」
    - watermark 为 False 时不添加「AI 生成」水印

    Args:
        client: Agent Plan OpenAI 兼容客户端
        image: 参考图（建议为含文字的图），公网 URL 字符串、本地文件路径字符串，或二者组成的列表
        prompt: 文字替换提示词
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
        prompt=prompt,
        size=size,
        n=1,
        # openai SDK 3.x 不再提供 image 参数，图生图参考图需经 extra_body 透传到请求体
        extra_body={"image": image_arg, "watermark": watermark},
    )
    return save_generated_images(response, save_dir=save_dir)


def main():
    """演示基于提示词的图片文字渲染 / 修改"""
    # 1. 校验 Agent Plan 专属 API Key 是否已配置
    if not os.getenv("AGENT_PLAN_API_KEY"):
        print("提示：未找到 AGENT_PLAN_API_KEY 环境变量。")
        print("请将根目录 .env.example 复制为 .env，填入 Agent Plan 专属 API Key 后重试。")
        return

    client = create_client()

    # 2. 场景一：文生图生成带指定文字的图片（海报 + 店招）
    poster_paths = []
    try:
        print("\n=== 场景一：文生图带文字（海报标题） ===")
        poster_paths = text_to_image_with_text(client, POSTER_PROMPT, watermark=False)
        print(f"生成并保存 {len(poster_paths)} 张图片：")
        for p in poster_paths:
            print(f"  - {p}")
    except Exception as e:
        # 单场景失败不中断演示，打印错误后继续
        print(f"[文生图带文字失败] {e}")

    try:
        print("\n=== 场景一：文生图带文字（店招） ===")
        sign_paths = text_to_image_with_text(client, SHOP_SIGN_PROMPT, watermark=False)
        print(f"生成并保存 {len(sign_paths)} 张图片：")
        for p in sign_paths:
            print(f"  - {p}")
    except Exception as e:
        print(f"[文生图带文字失败] {e}")

    # 3. 场景二：图生图修改图中文字
    # 参考图优先使用本脚本刚生成的海报（本地路径，无需外部资源）；
    # 若海报生成失败，则回退到公网参考图，也可替换为任意含文字的本地图片路径
    try:
        print("\n=== 场景二：图生图修改图中文字 ===")
        if poster_paths:
            ref_image = poster_paths[0]
            print(f"参考图（本脚本生成的海报）：{ref_image}")
        else:
            ref_image = REFERENCE_IMAGE_URL
            print(f"参考图（公网 URL，建议替换为含文字的本地图片）：{ref_image}")
        paths = edit_image_text(client, ref_image, EDIT_TEXT_PROMPT, watermark=False)
        print(f"生成并保存 {len(paths)} 张图片：")
        for p in paths:
            print(f"  - {p}")
    except Exception as e:
        print(f"[图生图改字失败] {e}")

    print("\n演示结束，所有结果已保存到 output/ 目录。")


if __name__ == "__main__":
    main()
