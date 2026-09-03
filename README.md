# Agent-Paradigm-Hub · 智能体范式示例库

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Anthropic-green)
![LangChain](https://img.shields.io/badge/Built%20with-LangChain-orange)
![License](https://img.shields.io/badge/License-TBD-lightgrey)

> **一站式学习 10 种主流 AI Agent 架构范式**：每种范式同时提供 **原生实现**（零框架、懂原理）和 **LangChain 实现**（生产级、可落地）两套等价代码，配套可直接运行的 Demo 与详尽中文注释。

***

## ✨ 项目亮点

| 亮点                      | 说明                                                                                                       |
| ----------------------- | -------------------------------------------------------------------------------------------------------- |
| 🎓 **10 种主流范式，全覆盖**     | 从经典的 ReAct / Plan-and-Execute，到前沿的 Reflection / Evaluator-Optimizer，3 大分类（单智能体 / 多智能体 / 工作流）共 10 种范式一次学透 |
| ⚖️ **原生 vs 框架，三实现对照**   | 每个示例既有原生 LLM API 手写的「从零实现版」（懂原理）、LangChain 的「生产版」（Runnable、AgentExecutor），也有 LangGraph 的「图编排版」（StateGraph 显式建模循环 / 并发 / 路由） |
| 🚀 **开箱即用，零门槛体验**       | 只需填入 API Key，`python -m examples.xxx` 即可运行完整 Demo；每个示例自带 `main()` 函数与演示问题，无需自己拼凑                         |
| 🧱 **模块化结构，易扩展**        | 统一的 `utils.llm_client.LLMClient` 屏蔽不同 LLM 厂商差异，新增 Provider、新增工具、新增范式都有清晰的扩展套路                            |
| 📝 **中文注释 + 中文 Prompt** | 所有代码头注释、函数级注释、系统 Prompt 全部为中文，便于国内开发者快速理解 Agent 内部工作机制                                                   |

***

## 🚀 快速开始

### 1. 环境要求

- Python **3.10 及以上**（使用了类型注解 `list[dict]` / `tuple[str, str]` 等 3.9+ 语法，以及部分 3.10 风格）

- 一个有效的 **OpenAI API Key**（默认使用 `gpt-4o-mini`）或 **Anthropic API Key**

### 2. 安装依赖

本仓库支持两种模式的依赖安装，按需要选择：

#### 方案 A：原生实现（推荐入门，仅需 4 个包）

```bash
pip install -r requirements.txt
```

内容：`openai` + `anthropic` + `python-dotenv`

#### 方案 B：原生 + LangChain（native + langchain 共 20 个示例）

```bash
pip install -r requirements.txt
pip install -r requirements-langchain.txt
```

内容：在方案 A 基础上追加 `langchain-core` / `langchain-community` / `langchain-openai` / `langchain-anthropic`。

#### 方案 C：原生 + LangGraph（30 个示例全部能跑）

```bash
pip install -r requirements.txt
pip install -r requirements-langgraph.txt
```

内容：在方案 A 基础上追加 `langgraph` / `langchain-openai`，用于运行每个范式的 `langgraph/` 实现。

### 3. 配置环境变量

```bash
# 1. 复制模板文件
cp .env.example .env        # Linux / macOS
# 或 Windows (PowerShell)
Copy-Item .env.example .env
```

编辑 `.env` 填入真实值，至少配置一个 API Key：

```dotenv
# --- 必填：OpenAI ---
OPENAI_API_KEY=sk-your-real-key-here
OPENAI_MODEL=gpt-4o-mini

# --- 可选：Anthropic（若想切换 Claude 才需要填）---
# ANTHROPIC_API_KEY=sk-ant-your-real-key-here
# ANTHROPIC_MODEL=claude-3-sonnet-20240229

# --- 全局默认 ---
DEFAULT_PROVIDER=openai       # openai 或 anthropic
DEFAULT_TEMPERATURE=0.7       # 0.0-2.0，越高越随机
DEFAULT_MAX_TOKENS=4096       # 单次调用最大输出 token
```

> 💡 如果你使用 **兼容 OpenAI 协议的中转服务 / 本地模型**（如 vLLM、Ollama、LM Studio 等），额外加一行：
>
> ```dotenv
> OPENAI_BASE_URL=https://your-compatible-endpoint/v1
> ```

### 4. 运行第一个 Demo

以经典的 **ReAct（推理+行动）范式 · 原生实现** 为例，它会演示 Agent 如何先思考、再调用计算器、再调用搜索，最后给出答案：

```bash
# 方式一（推荐）：以模块方式运行，包内 import 正确
python -m examples.single_agent.react.native.agent

# 方式二：cd 到目录后直接运行脚本
cd examples/single_agent/react/native
python agent.py
```

预期会看到类似以下的输出（节选）：

```
--- Step 1 ---
Agent: Thought: 用户想计算15*27，我应该先使用计算器。
Action: calculator
Action Input: 15 * 27
Tool (calculator): [计算结果] 15 * 27 = 405

--- Step 2 ---
Agent: Thought: 计算得到 405，接下来搜索这个数字的含义...
==================================================
最终答案: 15 * 27 = 405，而 405 在...（略）
```

🎉 至此环境已经跑通，你可以继续尝试下面任意一种范式。

***

## 📁 项目目录结构

```
Agent-Paradigm-Hub/
├── utils/
│   ├── __init__.py
│   └── llm_client.py                 # 🧠 统一 LLM 客户端（OpenAI / Anthropic 同接口）
│
├── examples/
│   ├── __init__.py
│   │
│   ├── single_agent/                 # 【一】单智能体范式（3 种）
│   │   ├── react/                      # ① ReAct：Reason + Act 交替循环
│   │   │   ├── native/agent.py           #   原生实现
│   │   │   ├── langchain/agent.py        #   LangChain 实现
│   │   │   └── langgraph/agent.py        #   LangGraph 实现
│   │   ├── plan_execute/               # ② Plan-and-Execute：先规划后执行
│   │   │   ├── native/agent.py
│   │   │   ├── langchain/agent.py
│   │   │   └── langgraph/agent.py
│   │   └── reflection/                 # ③ Reflection：自我反思改进
│   │       ├── native/agent.py
│   │       ├── langchain/agent.py
│   │       └── langgraph/agent.py
│   │
│   ├── multi_agent/                  # 【二】多智能体协作范式（3 种）
│   │   ├── debate/                     # ④ Debate：对抗式辩论 + 裁判
│   │   │   ├── native/agent.py
│   │   │   ├── langchain/agent.py
│   │   │   └── langgraph/agent.py
│   │   ├── multi_brain/                # ⑤ Multi-Brain：多视角协作 + 汇总
│   │   │   ├── native/agent.py
│   │   │   ├── langchain/agent.py
│   │   │   └── langgraph/agent.py
│   │   └── orchestrator_worker/        # ⑥ Orchestrator-Worker：编排者 + 执行者
│   │       ├── native/agent.py
│   │       ├── langchain/agent.py
│   │       └── langgraph/agent.py
│   │
│   └── workflow/                     # 【三】工作流模式（4 种）
│       ├── prompt_chaining/            # ⑦ Prompt Chaining：多步骤顺序链式
│       │   ├── native/workflow.py
│       │   ├── langchain/workflow.py
│       │   └── langgraph/workflow.py
│       ├── routing/                    # ⑧ Routing：意图识别路由分发
│       │   ├── native/workflow.py
│       │   ├── langchain/workflow.py
│       │   └── langgraph/workflow.py
│       ├── parallelization/            # ⑨ Parallelization：无依赖任务并发
│       │   ├── native/workflow.py
│       │   ├── langchain/workflow.py
│       │   └── langgraph/workflow.py
│       └── evaluator_optimizer/        # ⑩ Evaluator-Optimizer：生成-评估-改进循环
│           ├── native/workflow.py
│           ├── langchain/workflow.py
│           └── langgraph/workflow.py
│
├── requirements.txt                   # 原生实现依赖
├── requirements-langchain.txt         # LangChain 额外依赖
├── requirements-langgraph.txt         # LangGraph 额外依赖
├── .env.example                       # 环境变量模板
├── .gitignore
└── README.md                          # 👈 你正在阅读的文档
```

***

## 📚 范式大全

### 一、单智能体范式（Single-Agent）

#### ① ReAct（Reason + Act · 推理与行动交替）

**核心思想**：Agent 在每一步先通过 LLM「思考」接下来做什么（Thought），然后决定调用哪个工具（Action + Input），拿到工具返回（Observation）后继续思考，直到得出最终答案。边想边做，对需要外部信息的问题非常自然。

**典型场景**：需要实时搜索、调用 API、做计算的开放式问答；研究助理。

**核心流程**：

```
用户问题
  ↓
┌─→ LLM 输出 Thought / Action / Action Input
│      ↓
│   执行工具（web_search / calculator 等）
│      ↓
│   得到 Observation，拼入对话历史
└── 若输出 "Final Answer" 则结束，否则继续
```

**源码路径**：

- 原生实现：[examples/single\_agent/react/native/agent.py](examples/single_agent/react/native/agent.py)

- LangChain 实现：[examples/single\_agent/react/langchain/agent.py](examples/single_agent/react/langchain/agent.py)

- LangGraph 实现：[examples/single\_agent/react/langgraph/agent.py](examples/single_agent/react/langgraph/agent.py)

**运行命令**：

```bash
python -m examples.single_agent.react.native.agent
python -m examples.single_agent.react.langchain.agent
python -m examples.single_agent.react.langgraph.agent
```

***

#### ② Plan-and-Execute（先规划后执行）

**核心思想**：将任务切为两个阶段——先让 LLM 输出完成任务所需的**完整步骤列表**（JSON 格式），再逐个执行每个步骤，最后让 LLM 汇总执行结果。适合任务边界清晰、流程稳定的场景。

**与 ReAct 的区别**：ReAct 是**边想边做**（每一步即时决定下一步），Plan-and-Execute 是**先想好了再做**（开始前先把所有步骤都列出来），对需要多环节配合的业务任务更可控。

**典型场景**：报告撰写（查数据→分析→写报告→发邮件）、自动化工作流、运营 SOP。

**核心流程**：

```
用户任务
  ↓
阶段一：LLM 规划 → 输出 JSON 步骤列表 plan
  ↓
阶段二：for step in plan → 调用对应工具 → 收集 results
  ↓
阶段三：LLM 汇总 results → 最终总结
```

**源码路径**：

- 原生实现：[examples/single\_agent/plan\_execute/native/agent.py](examples/single_agent/plan_execute/native/agent.py)

- LangChain 实现：[examples/single\_agent/plan\_execute/langchain/agent.py](examples/single_agent/plan_execute/langchain/agent.py)

- LangGraph 实现：[examples/single\_agent/plan\_execute/langgraph/agent.py](examples/single_agent/plan_execute/langgraph/agent.py)

**运行命令**：

```bash
python -m examples.single_agent.plan_execute.native.agent
python -m examples.single_agent.plan_execute.langchain.agent
python -m examples.single_agent.plan_execute.langgraph.agent
```

***

#### ③ Reflection（生成 → 反思 → 改进 · 自我迭代）

**核心思想**：模仿人类「写完初稿 → 自己审视缺点 → 根据缺点修改」的写作过程。LLM 先产出一个草稿，再以「严厉批评者」的身份挑出逻辑错误/遗漏点，最后按批评意见逐条改进答案，循环 N 轮或自我满意为止。

**与 ReAct 的区别**：ReAct 通过外部工具获取新信息（**向外求**），Reflection 让 LLM 自我审视已有输出、提升质量（**向内求**）。二者可以互补使用。

**典型场景**：内容撰写（博客/文案）、方案设计、代码审查、论证优化等需要多次打磨质量的任务。

**核心流程**：

```
用户问题
  ↓
LLM 生成初稿（Generator）
  ↓
for i in 1..max_iterations:
    LLM 批评当前答案（Reflector）→ 输出 critique
    if critique 含 "已经很好" → break
    LLM 按 critique 改进（Refiner）→ current_answer
  ↓
输出最终答案
```

**源码路径**：

- 原生实现：[examples/single\_agent/reflection/native/agent.py](examples/single_agent/reflection/native/agent.py)

- LangChain 实现：[examples/single\_agent/reflection/langchain/agent.py](examples/single_agent/reflection/langchain/agent.py)

- LangGraph 实现：[examples/single\_agent/reflection/langgraph/agent.py](examples/single_agent/reflection/langgraph/agent.py)

**运行命令**：

```bash
python -m examples.single_agent.reflection.native.agent
python -m examples.single_agent.reflection.langchain.agent
python -m examples.single_agent.reflection.langgraph.agent
```

***

### 二、多智能体协作范式（Multi-Agent）

#### ④ Multi-Agent Debate（对抗式辩论 + 中立裁判裁决）

**核心思想**：两个（或多个）Agent 持相反立场针对同一主题多轮辩论，最后由独立的裁判 LLM 总结双方论点强弱并给出综合判断。在有争议的话题上可以避免单 Agent 的片面立场。

**典型场景**：产品方向评审（做 vs 不做）、技术方案对比、投资决策讨论、社会议题深度思考。

**核心流程**：

```
辩论主题 + 正反方立场设定
  ↓
for round in 1..rounds:
    正方发言 → 追加到 history
    反方发言 → 追加到 history
  ↓
Judge LLM 读完整 history → 给出中立总结 + 综合判断
```

**源码路径**：

- 原生实现：[examples/multi\_agent/debate/native/agent.py](examples/multi_agent/debate/native/agent.py)

- LangChain 实现：[examples/multi\_agent/debate/langchain/agent.py](examples/multi_agent/debate/langchain/agent.py)

- LangGraph 实现：[examples/multi\_agent/debate/langgraph/agent.py](examples/multi_agent/debate/langgraph/agent.py)

**运行命令**：

```bash
python -m examples.multi_agent.debate.native.agent
python -m examples.multi_agent.debate.langchain.agent
python -m examples.multi_agent.debate.langgraph.agent
```

***

#### ⑤ Multi-Brain（多脑协作 · 群体智慧）

**核心思想**：让一组持不同视角/角色的 Agent（如技术专家、产品经理、财务顾问、风险评估员）并行给出各自的专业判断，最后由一个综合者（Synthesizer）将多方观点融汇、找到联系和共识，形成最终答案。类似企业中的**专家会诊**。

**与 Debate 的区别**：Debate 是**对抗式**（正方反驳反方），Multi-Brain 是**协作式**（各方从不同维度思考，然后互补）。Debate 适合有明确正反方的问题，Multi-Brain 适合需要多维考虑的复杂决策。

**典型场景**：重大投资决策（技术+产品+财务+风险四维）、项目立项评审、策略研讨。

**核心流程**：

```
用户问题（如「是否投入 100 万做 AI Agent 产品？」）
  ↓
Brain 1 [技术专家] think() → p1
Brain 2 [产品经理] think() → p2
Brain 3 [财务顾问] think() → p3
Brain 4 [风险评估] think() → p4
  ↓
Synthesizer LLM 整合 p1~p4 → 最终决策建议
```

**源码路径**：

- 原生实现：[examples/multi\_agent/multi\_brain/native/agent.py](examples/multi_agent/multi_brain/native/agent.py)

- LangChain 实现：[examples/multi\_agent/multi\_brain/langchain/agent.py](examples/multi_agent/multi_brain/langchain/agent.py)

- LangGraph 实现：[examples/multi\_agent/multi\_brain/langgraph/agent.py](examples/multi_agent/multi_brain/langgraph/agent.py)

**运行命令**：

```bash
python -m examples.multi_agent.multi_brain.native.agent
python -m examples.multi_agent.multi_brain.langchain.agent
python -m examples.multi_agent.multi_brain.langgraph.agent
```

***

#### ⑥ Orchestrator-Worker（编排者 + 执行者分工协作）

**核心思想**：一个**总控 Orchestrator** 负责理解用户的大任务、拆成有序的子步骤列表（可带依赖），然后把每个子任务分配给专门的 **Worker**（搜索研究员、数据分析师、报告撰写者、质量审核员等）去执行，最后 Orchestrator 汇总所有 Worker 的产出形成最终结果。**典型的「一个老板 + 多个专家下属」结构**。

**典型场景**：大型研究报告撰写（搜索→分析→写作→审核）、跨职能项目执行、自动化内容流水线。

**核心流程**：

```
用户任务（如「写一份 AI Agent 趋势报告」）
  ↓
Orchestrator.decompose_task() → JSON steps（每个 step 带 worker + task + depends_on）
  ↓
for step in steps（按 depends_on 拓扑排序，无依赖的可并行）:
    Worker.execute(task, previous_context) → 结果加入 context
  ↓
Orchestrator 汇总 context → 最终整合报告
```

**源码路径**：

- 原生实现：[examples/multi\_agent/orchestrator\_worker/native/agent.py](examples/multi_agent/orchestrator_worker/native/agent.py)

- LangChain 实现：[examples/multi\_agent/orchestrator\_worker/langchain/agent.py](examples/multi_agent/orchestrator_worker/langchain/agent.py)

- LangGraph 实现：[examples/multi\_agent/orchestrator\_worker/langgraph/agent.py](examples/multi_agent/orchestrator_worker/langgraph/agent.py)

**运行命令**：

```bash
python -m examples.multi_agent.orchestrator_worker.native.agent
python -m examples.multi_agent.orchestrator_worker.langchain.agent
python -m examples.multi_agent.orchestrator_worker.langgraph.agent
```

***

### 三、工作流模式（Workflow）

#### ⑦ Prompt Chaining（链式提示 · 多步骤顺序传递）

**核心思想**：将复杂任务拆成一串**顺序步骤**，每一步的输出自动作为下一步输入的一部分，步骤间可以通过 `{input}` 模板引用。通过「把大象装进冰箱分三步」的思路，让 LLM 在每一步只专注做一件事，效果和稳定性都比一次 prompt 强得多。

**典型场景**：内容生产流水线（想法→大纲→文章→摘要→推文）、ETL（抽取→清洗→结构化→报表）、多语言翻译+润色链。

**核心流程**：

```
initial_input
  ↓
Step 1（大纲生成）: system_prompt_1 + user_template_1.format(input=initial_input) → out1
  ↓
Step 2（内容润色）: system_prompt_2 + user_template_2.format(input=out1)        → out2
  ↓
Step 3（推文提炼）: system_prompt_3 + user_template_3.format(input=out2)        → out3（最终输出）
```

**源码路径**：

- 原生实现：[examples/workflow/prompt\_chaining/native/workflow.py](examples/workflow/prompt_chaining/native/workflow.py)

- LangChain 实现：[examples/workflow/prompt\_chaining/langchain/workflow.py](examples/workflow/prompt_chaining/langchain/workflow.py)

- LangGraph 实现：[examples/workflow/prompt\_chaining/langgraph/workflow.py](examples/workflow/prompt_chaining/langgraph/workflow.py)

**运行命令**：

```bash
python -m examples.workflow.prompt_chaining.native.workflow
python -m examples.workflow.prompt_chaining.langchain.workflow
python -m examples.workflow.prompt_chaining.langgraph.workflow
```

***

#### ⑧ Routing（智能路由 · 识别意图后分发）

**核心思想**：先让「路由 LLM」判断用户请求属于哪一类（类别列表可自由配置），再把请求分发给对应专业 Handler。这样不同类型的问题可以用不同的系统 Prompt、甚至不同的模型大小/温度，整体效果和成本都更优。

**典型场景**：客服系统（售后/售前/技术/投诉分桶）、多模型切换（简单问题用便宜模型，复杂问题用 GPT-4）、企业内部统一智能入口。

**核心流程**：

```
用户请求
  ↓
Router LLM（JSON 输出 {"route": "code_helper", "reason": "..."}）→ route_key
  ↓
switch route_key:
    "general_qa"       → Handler A（博学助手 Prompt）
    "code_helper"      → Handler B（资深工程师 Prompt）
    "creative_writing" → Handler C（创意写手 Prompt）
    "data_analysis"    → Handler D（数据分析师 Prompt）
  ↓
选中的 Handler 返回专业回答
```

**源码路径**：

- 原生实现：[examples/workflow/routing/native/workflow.py](examples/workflow/routing/native/workflow.py)

- LangChain 实现：[examples/workflow/routing/langchain/workflow.py](examples/workflow/routing/langchain/workflow.py)

- LangGraph 实现：[examples/workflow/routing/langgraph/workflow.py](examples/workflow/routing/langgraph/workflow.py)

**运行命令**：

```bash
python -m examples.workflow.routing.native.workflow
python -m examples.workflow.routing.langchain.workflow
python -m examples.workflow.routing.langgraph.workflow
```

***

#### ⑨ Parallelization（并行化 · 无依赖子任务并发执行）

**核心思想**：当一个大任务可以拆成若干**相互独立**的子任务时，用 `asyncio`（原生）或 `RunnableParallel`（LangChain）并发处理，最后把所有子结果汇总成最终回答。比串行逐个跑可以节省大量等待时间。

**典型场景**：云服务选型对比（AWS/Azure/GCP 各自分析）、竞品研究（N 家竞品信息并行搜集）、批量内容生成（10 条营销文案并发）。

**核心流程**：

```
主任务 + subtasks = [st1, st2, st3, st4]
  ↓
asyncio.gather( 或 RunnableParallel {
    task_1: chain(st1),
    task_2: chain(st2),
    task_3: chain(st3),
    task_4: chain(st4),
} → results_map（按原始顺序可还原）
  ↓
Synthesizer LLM 整合 results_map → 最终主任务答案
```

**源码路径**：

- 原生实现：[examples/workflow/parallelization/native/workflow.py](examples/workflow/parallelization/native/workflow.py)

- LangChain 实现：[examples/workflow/parallelization/langchain/workflow.py](examples/workflow/parallelization/langchain/workflow.py)

- LangGraph 实现：[examples/workflow/parallelization/langgraph/workflow.py](examples/workflow/parallelization/langgraph/workflow.py)

**运行命令**：

```bash
python -m examples.workflow.parallelization.native.workflow
python -m examples.workflow.parallelization.langchain.workflow
python -m examples.workflow.parallelization.langgraph.workflow
```

***

#### ⑩ Evaluator-Optimizer（生成-评估-改进 · 循环迭代到质量达标）

**核心思想**：类似一个小团队——Generator 负责出初稿，Evaluator 作为严格 QA 从多个维度（准确性/完整性/清晰度/实用性）打分并给出改进建议，Optimizer 根据建议逐条改进；循环进行直到 Evaluator 打出「≥ 阈值分数 + Pass: yes」或达到最大迭代。

**与 Reflection 的区别**：Reflection 是**同一个 LLM 角色切换**自我反思；Evaluator-Optimizer 是**三种角色（生成者/评估者/优化者）明确分工**，更接近工程化团队协作，评估维度和打分格式也更可量化。

**典型场景**：代码生成质量保障、Banner 文案 A/B 级优化、教学设计打磨、任何需要明确评分标准的生产内容。

**核心流程**：

```
用户需求 requirements
  ↓
Generator Chain → 初稿 current
  ↓
for i in 1..max_iterations:
    Evaluator Chain(req, current) → eval_text:
        Score: X/10
        Feedback: [问题1 / 问题2 ...]
        Pass: yes/no
    if Pass 或 score >= pass_score → break
    Optimizer Chain(req, current, eval_text) → current（改进版）
  ↓
输出达标版本
```

**源码路径**：

- 原生实现：[examples/workflow/evaluator\_optimizer/native/workflow.py](examples/workflow/evaluator_optimizer/native/workflow.py)

- LangChain 实现：[examples/workflow/evaluator\_optimizer/langchain/workflow.py](examples/workflow/evaluator_optimizer/langchain/workflow.py)

- LangGraph 实现：[examples/workflow/evaluator\_optimizer/langgraph/workflow.py](examples/workflow/evaluator_optimizer/langgraph/workflow.py)

**运行命令**：

```bash
python -m examples.workflow.evaluator_optimizer.native.workflow
python -m examples.workflow.evaluator_optimizer.langchain.workflow
python -m examples.workflow.evaluator_optimizer.langgraph.workflow
```

***

## 🧠 核心工具：`utils.llm_client.LLMClient`

为了屏蔽 OpenAI 与 Anthropic 两个 SDK 的调用差异，仓库封装了统一的 LLM 客户端。所有原生实现（`native/*`）都通过它进行调用。

### 基本用法

```python
from utils.llm_client import LLMClient

# 1. 最简方式（读取 .env 中的 DEFAULT_PROVIDER / OPENAI_API_KEY 等）
llm = LLMClient()
reply = llm.chat(
    messages=[{"role": "user", "content": "你好，请介绍自己"}],
    system_prompt="你是一个幽默的助手。"
)
print(reply)

# 2. 自定义参数覆盖
llm2 = LLMClient(
    provider="anthropic",        # 或 "openai"
    model="claude-3-5-sonnet-20240620",
    temperature=0.2,
    max_tokens=2048,
)
reply2 = llm2.chat(messages=[{"role": "user", "content": "Hello"}])
```

### 类结构速览

| 方法 / 构造参数                             | 说明                                                    | 默认值                                                                                             |
| ------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `provider`                            | LLM 提供商，支持 `"openai"` / `"anthropic"`                 | 从 `DEFAULT_PROVIDER` 读取，否则 `"openai"`                                                           |
| `model`                               | 模型名                                                   | OpenAI: `OPENAI_MODEL` → `gpt-4o-mini`Anthropic: `ANTHROPIC_MODEL` → `claude-3-sonnet-20240229` |
| `temperature`                         | 温度 0-2                                                | 从 `DEFAULT_TEMPERATURE` → `0.7`                                                                 |
| `max_tokens`                          | 单次最大输出 token                                          | 从 `DEFAULT_MAX_TOKENS` → `4096`                                                                 |
| `.chat(messages, system_prompt=None)` | 发送对话请求。`messages` 格式为 `[{"role":..., "content":...}]` | 返回 LLM 的字符串回答                                                                                   |

***

## 🧩 扩展指南

### 1. 如何新增一个 LLM Provider（如 通义千问 / DeepSeek / 本地模型）

> 以兼容 OpenAI 协议的模型最简单。如果你的中转服务兼容 OpenAI 格式，**不需要改代码**，只需要在 `.env` 设置：
>
> ```dotenv
> OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
> OPENAI_API_KEY=sk-your-dashscope-key
> OPENAI_MODEL=qwen-plus
> ```
>
> 直接就能用。

如果 Provider 不兼容 OpenAI 协议（如原生的 Bedrock / Gemini REST API），则按以下步骤扩展 `utils/llm_client.py`：

1. 在 `LLMClient.__init__` 新增一个分支 `elif self.provider == "your_provider"`，调用 `_init_your_provider()`；
2. 新增 `_init_your_provider()` 方法，从环境变量读取 API Key 并初始化对应 SDK 客户端；
3. 在 `LLMClient.chat()` 的 `if/else` 增加 `_chat_your_provider()` 分支；
4. 实现 `_chat_your_provider(messages, system_prompt)`，按厂商 SDK 格式调用并**返回纯文本字符串**。

### 2. 如何给 ReAct 等 Agent 新增一个工具函数

以 `examples/single_agent/react/native/agent.py` 为例：

```python
# 1. 定义工具函数（函数签名接受一个 str 参数，返回 str）
def weather_forecast(city: str) -> str:
    """模拟天气查询工具（实际项目中请替换为真实 API 调用）"""
    return f"[天气] {city} 未来三天：晴→多云→小雨，气温 18-26°C。"

# 2. 在 AVAILABLE_TOOLS 字典注册
AVAILABLE_TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
    "weather_forecast": weather_forecast,   # 👈 新增
}

# 3. 在 System Prompt 中把工具加入说明
# （否则 LLM 不知道可以调用它）
SYSTEM_PROMPT = """...
你可以使用以下工具：
- web_search(query): ...
- calculator(expression): ...
- weather_forecast(city): 查询指定城市天气  # 👈 新增
"""
```

LangChain 版本更简单，给新函数加 `@tool` 装饰器并放进 `TOOLS` 列表即可，LangChain 会自动把 docstring 作为工具描述告诉 LLM。

### 3. 如何新增一种范式（如「Tree of Thoughts」）

1. 建目录：`examples/<分类>/<你的范式名>/`，下挂 `native/` 和 `langchain/` 两个子目录；
2. 每个子目录中创建 `agent.py`（或 `workflow.py`），包含完整的类实现、`main()` 演示入口；
3. 在相应的父级目录添加 `__init__.py`（空文件即可），保证 `python -m examples.xxx.yyy.zzz` 能正常运行；
4. 回到本 README 的「📚 范式大全」章节补登记：核心思想 / 场景 / 流程 / 源码路径 / 运行命令。

***

## ❓ FAQ

### Q1：运行时报 `ValueError: 未找到 OPENAI_API_KEY 环境变量` 怎么办？

A：确认已经把 `.env.example` 复制为 `.env` 并填入真实 Key；同时确认 `python` 运行时的**当前工作目录**是仓库根目录（因为 `LLMClient` 内部用 `dotenv.load_dotenv()` 从当前目录找 `.env`）。推荐始终用 `python -m examples.xxx.yyy` 从仓库根目录运行。

### Q2：想使用本地 / 开源 / 兼容 OpenAI 协议的模型怎么配？

A：在 `.env` 中设置两行即可，其他代码不用改：

```dotenv
OPENAI_API_KEY=sk-任意字符串   # 本地模型通常不校验 Key，随便填一个就行
OPENAI_BASE_URL=http://127.0.0.1:11434/v1  # 示例：Ollama 默认端口
OPENAI_MODEL=qwen2.5:7b
```

常见兼容服务：**Ollama、vLLM、LM Studio、OneAPI、硅基流动、阿里云百炼 DashScope** 等。

### Q3：原生实现和 LangChain 实现应该学哪个？

A：建议顺序：**先读原生实现（100 行内看懂循环逻辑）→ 再对照 LangChain 实现理解框架封装了什么 → 再读 LangGraph 实现看懂图编排如何显式建模循环/并发/路由**。生产项目可选 LangChain 或 LangGraph 版，学习/研究选原生版。三套实现输入输出等价，可以互相当「金标准参考答案」。

### Q4：如何调整模型温度 / 最大输出 token 数？

A：三种方式优先级从高到低：

1. 在代码里显式传给构造函数：`LLMClient(temperature=0.0, max_tokens=8192)`；
2. 在 `.env` 中设置 `DEFAULT_TEMPERATURE` / `DEFAULT_MAX_TOKENS`（全局生效）；
3. 不改配置，使用默认值（0.7 / 4096）。

### Q5：Agent 输出 `达到最大步数限制，未得出最终答案` 是怎么回事？

A：为了防止 Agent 进入无限循环/无限反思，所有范式都配置了 `max_steps` / `max_iterations`（通常默认 3\~10）。如果你确认任务需要更多轮数，可以在构造时调大：

```python
agent = ReActAgent(max_steps=30)
agent = ReflectionAgent(max_iterations=5)
```

更推荐的做法是优化 Prompt / 工具实现，让 Agent 更快收敛。

### Q6：这些示例能直接搬进生产项目吗？

A：**可以作为架构模板，不建议原封不动搬**。生产环境还需要补充：① 日志和可观测性（LangSmith / LangFuse / Phoenix）；② 结构化输出校验（Pydantic / Instructor）；③ 工具调用的重试/超时/限流（Tenacity）；④ 对话历史的存储与检索（Redis / DB + RAG）；⑤ 成本监控与错误告警。本仓库 20 个示例的最大价值是提供**每种范式的最小闭环**，你可以在此基础上按生产需求逐步加固。

***

## 🤝 贡献指南

欢迎 Issue / PR！建议的贡献方向：

- 新增范式：Tree of Thoughts、Self-Consistency、Mixture-of-Agents、Graph-based Multi-Agent 等

- 新增 Provider 支持（DashScope、ZhipuAI、Gemini、Bedrock 等）

- 新增真实可用工具（搜索 API、计算器 API、MCP 工具等）

- 修复 Bug、补充更多 LangGraph / LlamaIndex 等价实现

最小 PR 参考「扩展指南 - 3. 如何新增一种范式」四步曲即可。

***

## 📚 参考论文与资料

| 范式                                        | 参考来源                                                                                                                                    |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **ReAct**                                 | Yao et al., *[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)* (ICLR 2023)                |
| **Reflection / Reflexion**                | Shinn et al., *[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)* (2023)                |
| **Plan-and-Execute**                      | Wang et al., *[Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091)* 以及 LangGraph Plan-and-Execute 范式                            |
| **Multi-Agent Debate**                    | Du et al., *[Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325)* (2023) |
| **Mixture-of-Agents（Multi-Brain 相关）**     | Wang et al., *[Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692)* (2024)                  |
| **Evaluator-Optimizer / Self-Refinement** | Madaan et al., *[Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)* (2023)                        |

***

## 📄 License

项目 License **待补充**（当前仓库无 `LICENSE` 文件）。如需商用建议联系作者确认。

***

## 🧑‍💻 致谢

- 感谢 **OpenAI** / **Anthropic** 提供强大的 LLM API；

- 感谢 **LangChain 团队**打造的 `langchain-core` / `langchain-community` 框架生态；

- 感谢所有探索 LLM Agent 范式的研究者与开源贡献者，你们的工作让这个仓库的内容成为可能。

***

> 如果这个仓库帮助你快速学懂了某个 Agent 范式，欢迎 ⭐ Star / 分享给朋友！

