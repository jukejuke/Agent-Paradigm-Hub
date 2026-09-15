"""
Doubao Seedream 5.0-lite 图像生成示例（火山方舟 Agent Plan API）
================================================================

本示例演示如何通过火山方舟「Agent Plan」的 OpenAI 兼容接口，
使用豆包图像生成模型 Doubao-Seedream-5.0-lite 完成：

1. 文生图（text-to-image）：根据提示词直接生成图片
2. 图生图 / 多图参考（image-to-image）：结合参考图生成新图
3. 联网搜索生图（web_search，5.0-lite 独有）：融合实时网络信息生成图片
4. 结果保存：自动将生成结果下载 / 解码保存到本地 output 目录

模型 ID 对照：
- doubao-seedream-5-0-lite-260128  ：Seedream 5.0-lite（支持联网搜索）
- doubao-seedream-5-0-260128       ：Seedream 5.0（标准版）

运行方式（在仓库根目录执行）：
    python -m examples.multimodal.seedream.example

环境变量：
    AGENT_PLAN_API_KEY   # Agent Plan 专属 API Key（非方舟普通 API Key）
"""

import base64
import os
import sys
import urllib.parse
import urllib.request
import uuid

# 将项目根目录加入 sys.path，使本脚本可在任意工作目录运行
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dotenv import load_dotenv

# 加载根目录 .env 中的环境变量
load_dotenv()

# Agent Plan 专属 API 地址（OpenAI 兼容协议）
AGENT_PLAN_BASE_URL = "https://ark.cn-beijing.volces.com/api/plan/v3"
# 默认模型 ID（Seedream 5.0-lite，支持联网搜索）
DEFAULT_MODEL = "doubao-seedream-5-0-lite-260128"
# 生成结果默认保存目录
DEFAULT_SAVE_DIR = "output"


def create_client(api_key: str | None = None):
    """
    创建火山方舟 Agent Plan 的 OpenAI 兼容客户端

    Args:
        api_key: Agent Plan 专属 API Key；为 None 时从环境变量 AGENT_PLAN_API_KEY 读取

    Returns:
        openai.OpenAI 客户端实例
    """
    from openai import OpenAI

    key = api_key or os.getenv("AGENT_PLAN_API_KEY")
    if not key:
        raise ValueError(
            "未找到 AGENT_PLAN_API_KEY 环境变量，请在 .env 中配置 Agent Plan 专属 API Key"
        )
    return OpenAI(api_key=key, base_url=AGENT_PLAN_BASE_URL)


def _local_image_to_data_uri(image_path: str) -> str:
    """
    将本地图片文件转换为 Base64 Data URI

    Args:
        image_path: 本地图片文件路径

    Returns:
        符合 Agent Plan API 要求的 data:image/<格式>;base64,<...> 字符串
    """
    with open(image_path, "rb") as f:
        data = f.read()
    # 根据文件后缀推断 MIME 类型，未知后缀默认按 png 处理
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        mime = "jpeg"
    elif ext == "webp":
        mime = "webp"
    else:
        mime = "png"
    return f"data:image/{mime};base64,{base64.b64encode(data).decode('utf-8')}"


def save_generated_images(response, save_dir: str = DEFAULT_SAVE_DIR) -> list[str]:
    """
    将生成结果保存到本地目录

    支持两种返回形式：
    - item.url      ：通过 urllib 下载到本地
    - item.b64_json ：Base64 解码后写盘

    Args:
        response: client.images.generate() 的返回对象
        save_dir: 保存目录，默认 "output"

    Returns:
        保存成功的文件路径列表
    """
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    for idx, item in enumerate(response.data, start=1):
        # 场景一：返回公网 URL，直接下载到本地
        if getattr(item, "url", None):
            # 从 URL 路径提取扩展名，取不到时默认 .png
            ext = os.path.splitext(urllib.parse.urlparse(item.url).path)[1] or ".png"
            file_path = os.path.join(save_dir, f"{uuid.uuid4().hex[:8]}_{idx}{ext}")
            urllib.request.urlretrieve(item.url, file_path)
            saved_paths.append(file_path)
        # 场景二：返回 Base64 编码，解码后写盘
        elif getattr(item, "b64_json", None):
            b64 = item.b64_json
            # 若带 data:image/...;base64, 前缀则先剥离
            if b64.startswith("data:image"):
                b64 = b64.split(",", 1)[1]
            data = base64.b64decode(b64)
            file_path = os.path.join(save_dir, f"{uuid.uuid4().hex[:8]}_{idx}.png")
            with open(file_path, "wb") as f:
                f.write(data)
            saved_paths.append(file_path)
    return saved_paths


def text_to_image(
    client,
    prompt: str,
    size: str = "2K",
    n: int = 1,
    model: str = DEFAULT_MODEL,
    save_dir: str = DEFAULT_SAVE_DIR,
    watermark: bool = True,
) -> list[str]:
    """
    文生图：根据提示词直接生成图片

    说明：
    - size 支持 2K / 3K 两个分辨率档位（Seedream 5.0-lite）
    - 如需生成一组内容关联的组图，可追加参数 sequential_image_generation="auto"
      （参考图数量 + 最终生成图片数量 ≤ 15 张）
    - watermark 为 False 时不添加「AI 生成」水印

    Args:
        client: Agent Plan OpenAI 兼容客户端
        prompt: 图像描述提示词（中英文均可）
        size: 分辨率档位，默认 "2K"
        n: 生成图片数量，默认 1
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
        n=n,
        extra_body={"watermark": watermark},
    )
    return save_generated_images(response, save_dir=save_dir)


def image_to_image(
    client,
    prompt: str,
    image,
    size: str = "2K",
    model: str = DEFAULT_MODEL,
    save_dir: str = DEFAULT_SAVE_DIR,
    watermark: bool = True,
) -> list[str]:
    """
    图生图 / 多图参考：结合参考图与提示词生成新图

    说明：
    - image 支持公网 URL、本地文件路径，或由二者组成的列表
    - 本地文件会自动转换为 Base64 编码（data:image/<格式>;base64,<...>）
    - 传入多张参考图可实现多图融合（多图生图 / 多图生组图）
    - watermark 为 False 时不添加「AI 生成」水印

    Args:
        client: Agent Plan OpenAI 兼容客户端
        prompt: 图像描述提示词
        image: 参考图，公网 URL 字符串、本地文件路径字符串，或二者组成的列表
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
        image=image_arg,
        size=size,
        n=1,
        extra_body={"watermark": watermark},
    )
    return save_generated_images(response, save_dir=save_dir)


def generate_with_web_search(
    client,
    prompt: str,
    size: str = "2K",
    model: str = DEFAULT_MODEL,
    save_dir: str = DEFAULT_SAVE_DIR,
    watermark: bool = True,
) -> list[str]:
    """
    联网搜索生图：融合实时网络信息生成图片（5.0-lite 独有能力）

    说明：
    - 通过 tools=[{"type": "web_search"}] 开启联网搜索
    - 模型会根据提示词自主判断是否搜索互联网，适合时效性强的主题
    - 标准版 doubao-seedream-5-0-260128 不支持该能力
    - watermark 为 False 时不添加「AI 生成」水印

    Args:
        client: Agent Plan OpenAI 兼容客户端
        prompt: 图像描述提示词
        size: 分辨率档位，默认 "2K"
        model: 模型 ID，需使用 5.0-lite 才支持联网搜索
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
        tools=[{"type": "web_search"}],
        extra_body={"watermark": watermark},
    )
    return save_generated_images(response, save_dir=save_dir)


def main():
    """演示 Doubao Seedream 5.0-lite 图像生成能力"""
    # 1. 校验 Agent Plan 专属 API Key 是否已配置
    if not os.getenv("AGENT_PLAN_API_KEY"):
        print("提示：未找到 AGENT_PLAN_API_KEY 环境变量。")
        print("请将根目录 .env.example 复制为 .env，填入 Agent Plan 专属 API Key 后重试。")
        return

    client = create_client()

    # 2. 文生图：根据提示词直接生成图片
    try:
        print("\n=== 1. 文生图 ===")
        """
        paths = text_to_image(
            client,
            prompt="一只戴着墨镜的橘猫，坐在海边，日落，超写实",
        )
        """
        """
        paths = text_to_image(
            client,
            prompt="一只小熊，做在山峰，日落，动画形式",
            watermark=False,
        )
        """
        paths = text_to_image(
            client,
            prompt="一只戴着墨镜的橘猫，坐在海边，日落，超写实",
            watermark=False,
        )
        print(f"生成并保存 {len(paths)} 张图片：")
        for p in paths:
            print(f"  - {p}")
    except Exception as e:
        print(f"[文生图失败] {e}")

    # 3. 图生图：结合参考图生成新图（示例使用公网图片 URL，可替换为自己的图片）
    try:
        print("\n=== 2. 图生图 / 多图参考 ===")
        paths = image_to_image(
            client,
            prompt="将参考图中的主体置于雪景中，保持主体一致，电影感",
            image="http://qiuniu.xingrui-cn.com/1391154446972487606.jpg",
            watermark=False,
        )
        print(f"生成并保存 {len(paths)} 张图片：")
        for p in paths:
            print(f"  - {p}")
    except Exception as e:
        print(f"[图生图失败] {e}")

    # 4. 联网搜索生图：融合实时网络信息生成图片（5.0-lite 独有）
    try:
        print("\n=== 3. 联网搜索生图（5.0-lite 独有） ===")
        paths = generate_with_web_search(
            client,
            prompt="2026年中秋节主题海报，包含月亮与玉兔，国潮风格",
            watermark=False,
        )
        print(f"生成并保存 {len(paths)} 张图片：")
        for p in paths:
            print(f"  - {p}")
    except Exception as e:
        print(f"[联网搜索生图失败] {e}")

    print("\n演示结束，所有结果已保存到 output/ 目录。")


if __name__ == "__main__":
    main()
