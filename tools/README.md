# 编程输入提示词优化 Agent（Reflection - LangGraph 实现）

将你粗糙的开发需求（一句话想法、不完整的提示词）交给本 Agent，它通过 **Reflection（生成 → 自我批评 → 改进）** 循环迭代，打磨出一份高质量、可直接交给 **编程智能体（Trae / Claude Code / OpenCode / Cursor 等）** 执行的任务提示词。

```
原始需求 → 生成初稿 → 严厉批评 → 按批评改进 → …(循环)… → 最终优化提示词
                       (LangGraph StateGraph 显式建模该循环)
```

---

## 目录结构

```
tools/
├── __init__.py
├── prompt_optimizer/
│   ├── __init__.py        # 导出 PromptOptimizerAgent
│   ├── config.py          # 模型 / 温度 / 最大迭代次数配置（读环境变量）
│   ├── criteria.py        # 编程提示词质量评审维度清单
│   ├── prompts.py         # generate / reflect / refine 系统提示词 + 结束关键词
│   ├── agent.py           # 核心：Reflection - LangGraph Agent
│   └── __main__.py        # CLI 入口
├── requirements.txt       # 依赖
├── .env.example           # 环境变量模板
└── README.md
```

## 安装与配置

```bash
# 1. 安装依赖（需 Python 3.10+）
pip install -r tools/requirements.txt

# 2. 配置环境变量（复制仓库根目录 .env 或 tools/.env 均可）
Copy-Item tools\.env.example .env   # Windows PowerShell
# 编辑 .env 填入 OPENAI_API_KEY；兼容 OpenAI 协议的中转/本地模型可加 OPENAI_BASE_URL
```

> 说明：Agent 会自动向上找到仓库根目录的 `.env` 并加载，因此复用根目录 `.env` 最省事。

## 使用

### 方式一：命令行直接传参

```bash
python -m tools.prompt_optimizer "请帮我写一个 Python 脚本，从 CSV 读取数据并生成统计图表。"
```

### 方式二：交互式输入

```bash
python -m tools.prompt_optimizer
```

### 方式三：作为模块调用

```python
from tools.prompt_optimizer import PromptOptimizerAgent

agent = PromptOptimizerAgent()
optimized = agent.optimize("给电商后台加一个导出订单 Excel 的功能")
print(optimized)  # 复制给 Trae / Claude Code / OpenCode 执行
```

## 把优化后的提示词交给编程智能体

运行结束后终端会打印形如下方的**最终优化提示词**，整段复制到你的编程智能体对话/任务输入框即可：

```
============== 最终优化后的提示词 ==============
你是一名资深 Python 开发工程师……
任务目标：……
上下文：……
执行步骤：……
约束：……
验收标准：……
输出格式：……
=================================================
```

## 可调参数（环境变量）

| 变量                              | 说明                           | 默认值       |
| --------------------------------- | ------------------------------ | ------------ |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | 模型凭据与模型名               | gpt-4o-mini  |
| `OPENAI_BASE_URL`                 | 兼容 OpenAI 协议的中转/本地地址 | 无           |
| `PROMPT_OPTIMIZER_TEMPERATURE`    | 采样温度                       | 0.7          |
| `PROMPT_OPTIMIZER_MAX_ITERATIONS` | 反思-改进最大迭代次数          | 3            |

## 工作原理（Reflection 循环）

1. **generate**：LLM 扮演「提示词工程师」，把原始需求写成提示词初稿；
2. **reflect**：LLM 扮演「严厉批评者」，按 7 个维度（目标明确性 / 上下文完整 / 步骤可执行 / 约束禁忌 / 验收标准 / 输出格式 / 规模控制）逐条批评；
3. **refine**：LLM 扮演「改进专家」，按批评意见改进提示词；
4. 重复 2-3，直到批评意见判定「已足够好」或达到最大迭代次数。

图结构：`generate → reflect → (refine → reflect)* → END`
