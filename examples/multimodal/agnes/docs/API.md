# Agnes Image 2.5 Flash 模块 API 文档

> 模块路径：`examples/multimodal/agnes/`
> 模型 ID：`agnes-image-2.5-flash`
> Base URL：`https://apihub.agnes-ai.com/v1`（可用环境变量 `AGNES_BASE_URL` 覆盖）
> 认证：`Authorization: Bearer <AGNES_API_KEY>`

---

## 1. 核心函数总览

| 函数 | 所在模块 | 能力 |
| --- | --- | --- |
| `generate_image()` | `core/generate.py` | 文生图：根据文本提示词生成图片 |
| `style_transfer()` | `core/style_edit.py` | 图生图风格迁移：修改图片样式 |
| `generate_image_with_text()` | `core/text_edit.py` | 文生图带文字：在图片中添加文字 |
| `edit_image_text()` | `core/text_edit.py` | 图生图改字 / 删字：修改或删除图中文字 |
| `create_client()` | `core/base.py` | 创建 OpenAI 兼容客户端 |
| `save_generated_images()` | `core/base.py` | 将生成结果保存到本地 |
| `with_retry()` | `utils/errors.py` | 429 / 5xx 指数退避重试装饰器 |
| `build_style_prompt()` 等 | `utils/prompt_utils.py` | 风格 / 文字提示词构建 |

---

## 2. 公共参数说明

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `prompt` | `str` | 必填 | 图像描述提示词，中英文均可 |
| `image` | `str \| list[str]` | 必填（图生图） | 参考图：公网 URL / 本地路径 / 二者列表 |
| `size` | `str` | `"2K"` | 分辨率档位：`1K / 2K / 3K / 4K` |
| `ratio` | `str` | `"1:1"` | 宽高比：`1:1 / 3:4 / 4:3 / 16:9 / 9:16 / 2:3 / 3:2 / 21:9` |
| `n` | `int` | `1` | 生成图片数量，建议 ≤ 4 |
| `save_dir` | `str` | `"output"` | 结果保存目录 |

---

## 3. 函数详解

### 3.1 `generate_image(client, prompt, size="2K", ratio="1:1", n=1, save_dir="output") -> list[str]`

**能力**：文生图，根据文本提示词生成图片。

**调用示例**：

```python
from examples.multimodal.agnes.core import create_client, generate_image

client = create_client()
paths = generate_image(
    client,
    prompt="一只戴着墨镜的橘猫，坐在海边，日落，超写实",
    size="2K",
    ratio="1:1",
)
print(paths)  # 预期输出：["output/xxxx1234_1.png", ...]
```

**异常**：`AgnesConfigError`（size / ratio 非法）、`AgnesAPIError`（API 失败）。

**性能建议**：快速预览用 `1K`，正式出图用 `2K / 3K`；复用同一 `client`。

---

### 3.2 `style_transfer(client, image, style_prompt, size="2K", ratio="1:1", save_dir="output") -> list[str]`

**能力**：图生图风格迁移，对现有图片进行色彩 / 风格 / 构图等样式调整，保持主体与构图。

**调用示例**：

```python
from examples.multimodal.agnes.core import create_client, style_transfer
from examples.multimodal.agnes.utils import build_style_prompt

client = create_client()
style_prompt = build_style_prompt("中国传统水墨画，宣纸质感，浓淡墨色晕染，留白意境")
paths = style_transfer(client, "output/ref.png", style_prompt)
print(paths)  # 预期输出：["output/xxxx5678_1.png", ...]
```

**异常**：`AgnesConfigError`、`AgnesImageError`（本地参考图不存在）、`AgnesAPIError`。

---

### 3.3 `generate_image_with_text(client, prompt, size="2K", ratio="1:1", n=1, save_dir="output") -> list[str]`

**能力**：文生图生成带指定文字的图片（在图片中添加文字），支持中英文。

**调用示例**：

```python
from examples.multimodal.agnes.core import create_client, generate_image_with_text
from examples.multimodal.agnes.utils import build_text_add_prompt

client = create_client()
prompt = build_text_add_prompt(
    "中秋快乐",
    scene_hint="中秋节促销海报，居中大标题，背景为圆月与玉兔，国潮插画风格",
)
paths = generate_image_with_text(client, prompt)
```

**提示词要点**：文字用「」标注并强调「文字必须完全一致、无错别字」。

---

### 3.4 `edit_image_text(client, image, instruction, size="2K", ratio="1:1", save_dir="output") -> list[str]`

**能力**：图生图修改 / 删除图中文字，保持主体、版式与画风一致。

**调用示例（改字）**：

```python
from examples.multimodal.agnes.core import create_client, edit_image_text
from examples.multimodal.agnes.utils import build_text_replace_prompt

client = create_client()
instruction = build_text_replace_prompt("中秋快乐", "新年快乐")
paths = edit_image_text(client, "output/poster.png", instruction)
```

**调用示例（删字）**：

```python
from examples.multimodal.agnes.utils import build_text_remove_prompt

instruction = build_text_remove_prompt("中秋快乐")
paths = edit_image_text(client, "output/poster.png", instruction)
```

---

### 3.5 `create_client(api_key=None, base_url=None)`

**能力**：创建 `openai.OpenAI` 客户端；Key 缺失时抛 `AgnesConfigError` 并给出中文提示。

```python
client = create_client()  # 从环境变量读取 AGNES_API_KEY / AGNES_BASE_URL
```

---

### 3.6 `with_retry(max_retries=3, base_delay=1.0)`

**能力**：装饰器，对 `429` 限流与 `5xx` 服务端错误按 `base_delay * 2 ** attempt` 指数退避重试；网络层错误直接包装为 `AgnesAPIError` 抛出。核心函数默认已启用。

---

## 4. 异常体系

| 异常 | 触发场景 |
| --- | --- |
| `AgnesError` | 统一基类，上层只需捕获它即可覆盖所有业务异常 |
| `AgnesConfigError` | API Key 缺失、分辨率 / 宽高比非法 |
| `AgnesImageError` | 本地图片文件不存在、下载失败 |
| `AgnesAPIError` | API 调用失败（HTTP 错误 / 网络错误） |

## 5. 错误处理规范

- 核心函数内部已内置参数校验与自动重试，业务侧只需：
  ```python
  from examples.multimodal.agnes.utils import AgnesError

  try:
      paths = generate_image(client, prompt=prompt)
  except AgnesError as e:
      print(f"[生成失败] {e}")
  ```
- 单场景失败不中断整批任务（参考示例 `main()` 的 `try/except` 写法）。

## 6. 性能优化建议

1. **复用客户端**：`create_client()` 只调一次，多个任务共享同一 `client`。
2. **档位选择**：按用途选 `1K / 2K / 3K / 4K`，避免无谓的高分辨率开销。
3. **批量出图**：文生图用 `n` 参数一次生成多张，减少请求次数。
4. **自动重试**：核心函数已内置 429 / 5xx 指数退避重试，无需业务侧重试。
5. **并行生成**：多个独立任务可用 `concurrent.futures.ThreadPoolExecutor` 并发调用（API 为 HTTP 请求，线程池即可获得并发收益）。
