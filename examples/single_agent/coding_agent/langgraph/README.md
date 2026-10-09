# Coding Agent（LangGraph）示例包使用文档

## 1. 简介

Coding Agent（代码生成智能体）示例，基于 LangGraph 实现「生成/修改 → 执行 → 评估 →（未通过则修复）→ 再执行」的迭代循环，直到代码通过验收或达到最大迭代次数，最终将代码保存到本地。

智能体接收一个编程任务描述后：

1. 由 LLM 生成候选代码（或基于指定文件进行修改）；
2. 在沙箱中执行该代码并捕获 stdout / 错误输出；
3. 由 LLM 评审执行结果，判定是否通过验收；
4. 若未通过，LLM 依据反馈修复代码并再次执行；
5. 如此循环，直至通过或达到 `max_iterations` 上限；
6. 最终代码通过 `save` 节点写入本地文件。

## 2. 目录结构与四大核心组件

```
examples/single_agent/coding_agent/langgraph/
├── state.py        # 智能体状态管理模块（CodingAgentState + 迭代历史，跟踪代码生成上下文）
├── tools.py        # 工具调用接口（run_python 沙箱执行 / read_file 读取 / save_code 保存）
├── prompts.py      # 提示词模块（生成/修改/评估/修复系统提示词 + 代码围栏剥离）
├── agent.py        # 决策逻辑（evaluate + should_continue）与循环执行（StateGraph），含 save 保存节点
└── tests/          # 无需 API Key 的单元测试
```

各组件职责：

- **state.py**：智能体状态管理模块。定义 `CodingAgentState`（`TypedDict`），字段含 `task`、`code`、`execution_output`、`execution_error`、`feedback`、`passed`、`iteration`、`max_iterations`、`target_file`、`output_path`、`history`（历史用归约器累积），用于跨节点传递并跟踪代码生成上下文。
- **tools.py**：工具调用接口。`run_python(code, timeout=30)` 通过 subprocess 在沙箱中执行 Python 代码，支持超时保护与错误捕获；`read_file(file_path)` 读取指定文件内容；`save_code(file_path, content)` 将代码写入本地文件（自动创建父目录）。三者均返回结构化 dict，不抛出未捕获异常。
- **prompts.py**：提示词模块。提供 `GENERATE_PROMPT`（从零生成）、`MODIFY_PROMPT`（修改现有代码）、`EVALUATE_PROMPT`（评估）、`FIX_PROMPT`（修复）四类系统提示词，以及 `strip_code_fence(text)` 用于剥离模型输出中的 Markdown 代码围栏。
- **agent.py**：决策逻辑单元与循环执行机制。`evaluate` 节点结合 LLM 评审与执行结果给出 `passed` 判定；`should_continue` 作为条件边决定「继续修复（fix）」还是「保存结束（save）」；`save` 节点把最终代码写入 `output_path`；最后通过 `StateGraph` 将各节点串成完整的迭代循环。
- **tests/**：无需 API Key 的单元测试，覆盖工具行为、决策逻辑、围栏剥离与图构建。

## 3. 依赖安装

在项目根目录依次安装：

```bash
pip install -r requirements.txt
pip install -r requirements-langgraph.txt        # 或使用本示例 requirements.txt
pip install -r examples/single_agent/coding_agent/requirements.txt
```

## 4. 环境变量配置

复制根目录 `.env.example` 为 `.env`，并至少配置以下变量：

| 变量名             | 必填 | 说明                                   |
| ------------------ | ---- | -------------------------------------- |
| `OPENAI_API_KEY`   | 是   | OpenAI（或兼容）API 密钥               |
| `OPENAI_MODEL`     | 否   | 使用的模型名，默认示例内置             |
| `OPENAI_BASE_URL`  | 否   | 自定义 API 地址（如使用代理或兼容服务） |

## 5. 运行

```bash
# 方式一：模块方式（推荐，从项目根目录）
python -m examples.single_agent.coding_agent.langgraph.agent

# 方式二：脚本方式
cd examples/single_agent/coding_agent/langgraph
python agent.py
```

`main()` 会依次演示两个场景：

1. **从零生成**：编写 fibonacci(n) 函数并打印前 10 项，最终代码保存到 `<项目根>/outputs/coding_agent_output.py`；
2. **修改指定文件**：创建 `outputs/sample_calc.py` 示例脚本，再要求 Agent 把 `multiply` 函数改为三参数并写回该文件。

## 6. 最终代码保存

- `save` 节点会在循环结束后把验收通过的最终代码写入本地文件，并在控制台打印保存路径。
- 默认保存路径：
  - 从零生成模式：`<项目根>/outputs/coding_agent_output.py`；
  - 修改指定文件模式：就地写回 `target_file` 指向的文件。
- 可通过 `run(..., output_path="...")` 显式指定保存路径（两种模式均生效）。

## 7. 修改指定文件

通过 `run(..., target_file="路径")` 开启修改模式：

```python
from examples.single_agent.coding_agent.langgraph.agent import run

# 读取 demo.py，按任务修改后写回 demo.py
run("把函数 add 改为支持三个参数", target_file="demo.py")

# 修改后另存到新文件（不覆盖原文件）
run("把函数 add 改为支持三个参数", target_file="demo.py", output_path="demo_v2.py")
```

行为说明：

- `target_file` 的内容会被读取并作为初始代码进入「执行 → 评估 → 修复」循环；
- 生成/修复提示词会切换到「修改现有代码」模式（`MODIFY_PROMPT`）；
- 最终代码默认写回 `target_file`（就地修改）；如需保留原文件，请显式传 `output_path`。

> 注意：就地修改会覆盖原文件，请确认文件可被覆盖，或使用 `output_path` 另存。

## 8. 预期输出观察点

运行后，控制台会依次出现如下节点流转日志：

1. `[节点] generate`：生成初始代码（或基于现有代码完成修改）；
2. `[节点] execute`：沙箱执行，显示 stdout / 错误输出；
3. `[节点] evaluate`：评审并给出 `passed` 判定；
4. 若未通过：`[节点] fix` 修复代码；
5. 再次进入 `execute` / `evaluate`，如此循环……
6. `[节点] save`：把最终代码写入本地文件。

最终打印「验收结果」「保存路径」与「最终代码」。

> 说明：演示任务为「编写 fibonacci(n) 函数并打印前 10 项」，正常应在 1 轮内通过。

## 9. 测试

```bash
python -m pytest examples/single_agent/coding_agent/tests/ -q
# 或（无需 pytest）
python examples/single_agent/coding_agent/tests/test_coding_agent.py
```

说明：测试覆盖工具行为（执行/读取/保存）、决策逻辑、围栏剥离与图构建，不调用 LLM、不需要 API Key。

## 10. 可调参数

- `run(task, max_iterations=3, target_file=None, output_path=None)`：
  - `max_iterations`：最大修复轮次，超过后即使未通过也会结束循环；
  - `target_file`：指定要修改的目标文件路径（为空则从零生成）；
  - `output_path`：最终代码保存路径（为空时按默认规则解析）。
- `run_python(code, timeout=30)`：`timeout` 控制单次代码执行超时（秒），超时后强制终止并返回「超时」错误信息。
