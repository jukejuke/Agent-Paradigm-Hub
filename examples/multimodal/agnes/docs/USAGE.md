# Agnes Image 2.5 Flash 使用指南

> 在仓库 `examples/multimodal/agnes/` 下，基于 **Agnes AI 的 OpenAI 兼容接口**实现 `agnes-image-2.5-flash` 的图片生成与编辑。

## 1. 目录结构

```
examples/multimodal/agnes/
├── config/                     # ① 模型配置：模型 ID / Base URL / 档位 / 比例 / 版本
├── core/                       # ② 核心功能：文生图 / 风格迁移 / 文字编辑
├── utils/                      # ③ 工具函数：错误处理 / 图片处理 / 提示词构建
├── examples/                   # ④ 示例代码：3 个可直接运行的完整示例
└── docs/                       # ⑤ 文档：API 文档（本指南）
```

## 2. 环境准备

1. 在 [Agnes AI 控制台](https://agnes-ai.com) 创建 API Key。
2. 复制根目录 `.env.example` 为 `.env`，配置：

```dotenv
# --- Agnes Image 2.5 Flash（OpenAI 兼容接口）---
AGNES_API_KEY=your-agnes-api-key-here
# AGNES_BASE_URL=https://apihub.agnes-ai.com/v1   # 可选，默认国际站；国内可换 https://apihub.agnes-ai.cn/v1
```

3. 依赖已包含在根目录 `requirements.txt`（`openai>=1.0.0` + `python-dotenv>=1.0.0`），**无需额外安装**。

## 3. 运行示例

均在**仓库根目录**执行：

```bash
# 图片生成（文生图）
python -m examples.multimodal.agnes.examples.example_generate

# 图片样式修改（风格迁移）
python -m examples.multimodal.agnes.examples.example_style_edit

# 文字编辑（加字 / 改字 / 删字）
python -m examples.multimodal.agnes.examples.example_text_edit
```

未配置 `AGNES_API_KEY` 时会打印中文友好提示并退出；配置后生成结果保存到 `output/` 目录。

## 4. 作为模块调用

```python
from examples.multimodal.agnes.core import create_client, generate_image, style_transfer
from examples.multimodal.agnes.utils import build_style_prompt

client = create_client()

# 1. 文生图
paths = generate_image(client, prompt="海边日落，超写实", size="2K", ratio="16:9")

# 2. 风格迁移（参考图：本地路径或公网 URL）
style_prompt = build_style_prompt("赛博朋克，霓虹灯光，电影感")
paths = style_transfer(client, "output/xxx_1.png", style_prompt)
```

## 5. 参数速查

| 参数 | 可选值 | 默认 |
| --- | --- | --- |
| `size` | `1K / 2K / 3K / 4K` | `2K` |
| `ratio` | `1:1 / 3:4 / 4:3 / 16:9 / 9:16 / 2:3 / 3:2 / 21:9` | `1:1` |
| `n` | 正整数（建议 ≤ 4） | `1` |
| `image` | 公网 URL / 本地路径 / 二者列表 | 必填（图生图） |

## 6. 性能优化建议

1. **复用客户端**：`create_client()` 只调一次，所有任务共享同一 `client`（示例已示范）。
2. **档位选择**：快速预览 `1K`，正式出图 `2K / 3K`，极致细节 `4K`。
3. **自动重试**：核心函数内置 429 / 5xx 指数退避重试，无需重复造轮子。
4. **并发生成**：独立任务用线程池并行（示例 4 的 parallelization 范式思路同样适用）。

## 7. FAQ

### Q1：提示「未找到 AGNES_API_KEY 环境变量」怎么办？
A：确认已把根目录 `.env.example` 复制为 `.env` 并填入真实 Key；`load_dotenv()` 从**当前工作目录**读取 `.env`，请从仓库根目录运行。

### Q2：图生图（风格迁移 / 改字）返回错误？
A：确认参考图存在且格式受支持（本地路径会自动转 Base64；公网 URL 需可公开访问）。若服务端将图生图收敛到 `/v1/images/edits` 端点，可将 `core/style_edit.py` 与 `core/text_edit.py` 中的 `client.images.generate(...)` 替换为 `client.images.edit(...)`，请求体形态保持一致。

### Q3：文字渲染不准确 / 有错别字怎么办？
A：提示词中把文字用「」标注并显式要求「文字必须完全一致、无错别字」（`prompt_utils` 模板已内置）；文字较长时建议拆分为标题 + 副标题分次渲染。

### Q4：如何切换国内节点？
A：在 `.env` 中设置 `AGNES_BASE_URL=https://apihub.agnes-ai.cn/v1`，代码无需改动。

## 8. 模型信息

| 项 | 值 |
| --- | --- |
| 模型 ID | `agnes-image-2.5-flash` |
| 版本 | 2.5 |
| 发布 | 2026-07 |
| 能力 | 文生图 / 图生图（重绘、换风格）/ 多图参考 / 图内多语言文字渲染 |
