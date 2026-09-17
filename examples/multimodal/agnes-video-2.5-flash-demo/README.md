# Agnes Video 2.5 Flash 视频生成示例

基于 Agnes Video 2.5 Flash 视频模型的 Python 示例项目，演示三种视频生成场景：

| 场景 | 说明 | 示例脚本 |
| --- | --- | --- |
| 文生视频 | 根据文本提示词直接生成视频 | `examples/example_text_to_video.py` |
| 图生视频 | 以参考图为依据生成动态视频（reference 模式） | `examples/example_image_to_video.py` |
| 风格迁移 | 将目标艺术风格应用到视频内容（提示词 / 风格参考图） | `examples/example_style_transfer.py` |

视频生成采用**异步任务式 API**：创建任务 → 轮询进度 → 下载 MP4，流程完整封装在 [src/generate.py](src/generate.py) 中。

> 作者: Sol ｜ 日期: 2026-09-17 ｜ 依赖: Python 3.8+ ｜ 接口依据官方 [Agnes Video 2.5 Flash 文档](https://wiki.agnes-ai.com/zh-Hans/docs/agnes-video-25-flash)

---

## 一、环境配置

### 1. 申请 API Key

访问 [Agnes AI 控制台](https://agnes-ai.com) 注册账号并创建 API Key（`sk-` 开头）。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

将项目根目录的 `.env.example` 复制为 `.env`，填入 API Key：

```bash
copy .env.example .env   # Windows
```

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `AGNES_API_KEY` | 是 | Agnes API Key |
| `AGNES_VIDEO_BASE_URL` | 否 | 视频专用 API 地址（优先级最高），如 `https://api.agnes-ai.cn/v1` |
| `AGNES_BASE_URL` | 否 | 通用 API 地址。视频未单独配置时使用；若其带有 `/images/generations` 等端点路径，会自动归一化到 `/v1` 根地址 |
| `AGNES_VIDEO_MODEL` | 否 | 模型 ID，默认 `agnes-video-2.5-flash`；如当前账号不可用可回退 `agnes-video-v2.0` |
| `AGNES_VIDEO_IMAGE_URL` | 否 | 图生视频示例的默认参考图（公网 URL） |

---

## 二、使用方法

### 方式一：命令行接口（推荐）

```bash
# 文生视频
python src/cli.py --mode text --prompt "一只赛博小狗在赛博坦星球激烈战斗，电影级镜头"

# 图生视频（image 为公网图片 URL，或本地路径）
python src/cli.py --mode image --image https://example.com/ref.png

# 风格迁移（纯提示词控制风格）
python src/cli.py --mode style --style "中国水墨画" --prompt "仙鹤展翅飞过山水间"

# 查看全部参数说明
python src/cli.py --help
```

常用参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--mode {text,image,style}` | 生成场景（必填） | — |
| `--prompt` | 视频内容描述提示词 | text 必填，其余可选 |
| `--image` | 参考图（URL 或本地路径） | image 模式必填 |
| `--style` | 目标艺术风格 | style 模式必填 |
| `--api-mode {text,keyframe,reference}` | API 生成模式（高级） | 按场景自动选择 |
| `--seconds` | 视频时长（字符串 4~12 秒） | `5` |
| `--size` | 分辨率档位，flash 固定 `720P` | `720P` |
| `--aspect-ratio` | 宽高比 | `16:9` |
| `--seed` | 随机种子（固定可复现） | 随机 |
| `--output` | 视频保存目录 | `outputs` |
| `--no-preview` | 生成后不自动预览 | 自动预览 |
| `--html-preview` | 以 HTML 预览页方式预览 | 默认播放器 |

### 方式二：示例脚本

```bash
python examples/example_text_to_video.py      # 文生视频
python examples/example_image_to_video.py     # 图生视频（需先配置参考图）
python examples/example_style_transfer.py     # 风格迁移视频
```

图生视频示例的参考图来源：环境变量 `AGNES_VIDEO_IMAGE_URL` 优先，其次使用脚本内 `IMAGE_URL` 常量。

---

## 三、参数配置指南

### 生成模式（mode）

| 模式 | 用途 | 必备字段 | 说明 |
| --- | --- | --- | --- |
| `text` | 文生视频 | `prompt` | 纯文本生成，不允许携带媒体字段 |
| `reference` | 参考图生成（图生视频） | `prompt` + `images[]` | 最多 5 张图，提示词用 `<Picture 1>` 指代 |
| `keyframe` | 首尾帧控制 | `first_frame` / `last_frame` | 至少提供一个帧 URL |

### 时长（seconds）

- 字符串类型，范围 `"4"`~`"12"`，默认 `"5"` 秒
- 时长越长生成耗时与费用越高

### 分辨率与宽高比

- flash 模型分辨率**固定 720P**（`size="720P"`），传入其他值返回 HTTP 400
- 宽高比支持 `21:9 / 16:9 / 4:3 / 1:1 / 3:4 / 9:16`，默认 `16:9`（输出 1280x704）

### 提示词建议

| 要素 | 说明 | 示例 |
| --- | --- | --- |
| 主体 | 明确角色 / 物体及特征 | 一只赛博小狗 |
| 动作 | 具体动作与幅度 | 激烈战斗、展翅飞过 |
| 镜头 | 镜头运动方式 | 镜头缓缓推进、平移 |
| 氛围 | 环境、光线、情绪 | 日落、云雾缭绕、电影级 |

### 风格迁移提示词写法

内部按「以`<style>`的风格呈现，`<prompt>`」自动拼装，推荐：

- `style`：目标风格 + 质感 + 氛围，如「中国传统水墨画，宣纸质感，浓淡墨色晕染，留白意境」
- `prompt`：主体动作 + 镜头，如「仙鹤在云雾缭绕的山水间展翅飞过，镜头缓缓平移」
- 可选 `--image` 风格参考图：提供后自动切换 `reference` 模式，以 `<Picture 1>` 作为风格基准

---

## 四、功能说明

### 输出目录

生成视频自动保存到 **`outputs/`**（目录不存在时自动创建），文件名格式：
`<随机8位>_<时间戳>.mp4`，可用 `--output` 指定其他目录。

### 进度显示

轮询任务时实时刷新文本进度条（基于 API 的 `progress` 字段），支持 Ctrl+C 安全中断。

### 视频预览

- 默认：调用系统默认播放器打开（Windows 使用 `os.startfile`）
- `--html-preview`：生成 `outputs/preview.html` 预览页（含 `<video>` 播放器）
- `--no-preview`：仅保存不预览

### 错误处理与日志

- 统一异常体系：`AgnesError` → `AgnesConfigError`（配置）/ `AgnesAPIError`（API 调用）/ `AgnesVideoError`（任务失败、下载失败）
- 429 限流与 5xx 服务端错误自动**指数退避重试**（最多 3 次）
- 日志输出到控制台，格式：`时间 | 级别 | 消息`

---

## 五、常见问题

| 问题 | 处理方式 |
| --- | --- |
| 未找到 AGNES_API_KEY 环境变量 | 复制 `.env.example` 为 `.env` 并填入 Key |
| 模型 ID 不可用（404 / 模型不存在） | 环境变量 `AGNES_VIDEO_MODEL=agnes-video-v2.0` 回退官方公开版本 |
| 返回 401 Invalid token | 当前 Key 与网关不匹配，设置 `AGNES_VIDEO_BASE_URL` 为 Key 所属区域网关（如 `https://api.agnes-ai.cn/v1`） |
| 返回 404 Invalid URL / 路径异常 | 视频地址被通用 `AGNES_BASE_URL` 带偏（如带 `/images/generations`），已自动归一化到 `/v1`；仍异常时显式设置 `AGNES_VIDEO_BASE_URL` |
| `mode 无效` / `size must be 720P` | mode 仅支持 text / keyframe / reference；size 仅支持 720P |
| 参考图生成失败 | images / first_frame 必须为公网可访问的图片 URL，且生成期间保持有效 |
| 生成任务长时间排队 | 免费服务高峰期会排队，适当调大 `--timeout` |
