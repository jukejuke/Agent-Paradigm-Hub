"""
ReAct (Reason + Act) Agent - LangGraph 实现
==============================================

【什么是 ReAct？】
ReAct 是一种让 AI 交替进行「推理（Reason）」与「行动（Act）」的模式。
核心思想：当模型遇到复杂问题时，不要一次性直接回答，而是：
    1. 先思考（Thought）：我该怎么做？
    2. 再行动（Act）：调用某个工具（如搜索、计算）来获取信息；
    3. 观察结果（Observation）：查看工具返回了什么；
    4. 循环上述步骤，直到模型觉得自己有足够信息给出最终答案。

【LangGraph 如何实现 ReAct？】
LangGraph 使用「状态图（StateGraph）」把 agent 和工具之间的循环显式建模出来：
- agent 节点：由 LLM 推理并决定下一步行动（是否调用工具）；
- tools 节点：执行 agent 选择调用的工具；
两者通过「条件边」连接，循环往复，直到 agent 给出最终答案。

【术语速查】
- 节点（Node）：图中的一步操作，本质就是一个 Python 函数。
- 边（Edge）：连接节点，决定数据如何流动。
- 条件边（Conditional Edge）：根据某个函数的结果，动态选择下一步走向。
- 状态（State）：在整个图执行过程中共享的数据容器，这里就是「消息列表」。

参考: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
"""

import os
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from pathlib import Path
from dotenv import load_dotenv

# 定位项目根目录并加载 .env（支持从任意目录直接运行本文件）
# parents[0] 是 agent.py 所在目录，parents[4] 往上第 5 层即项目根目录，
# 根目录下通常放着 .env 文件（里面有 OPENAI_API_KEY 等配置）
load_dotenv(Path(__file__).resolve().parents[4] / ".env")


# ==============================================================================
# 定义工具
# ==============================================================================
# 在 ReAct 模式中，工具就是 agent 的「手脚」。
# 使用 @tool 装饰器，可以把一个普通函数变成 LangChain 认知的「工具」，
# 这样 LLM 在推理时就知道有这些工具可用，并能按函数签名生成调用参数。

@tool
def web_search(query: str) -> str:
    """
    模拟网络搜索工具

    Args:
        query: 搜索关键词

    Returns:
        中文模拟搜索结果
    """
    # 这里只是返回一段模拟文本，实际项目中可以调用真实的搜索引擎 API
    return f"[搜索结果] 关于 '{query}' 的信息：这是一个模拟的搜索结果。"


@tool
def calculator(expression: str) -> str:
    """
    安全计算数学表达式（使用 ast 解析，禁止 eval）

    Args:
        expression: 数学表达式字符串，例如 "15 * 27"

    Returns:
        计算结果字符串，格式为 "[计算结果] 表达式 = 结果"
    """
    import ast
    import operator

    # 安全的运算符映射，仅允许白名单内的运算符
    # 例如：遇到 "+"（ast.Add）就执行 operator.add（即 +）
    _SAFE_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }

    def _safe_eval(node):
        """递归安全求值 AST 节点，仅允许常量、二元运算与一元正负号"""
        # 说明：为什么不用 eval()？因为 eval() 可以执行任意代码，很危险。
        # 这里改用 ast 把表达式解析成「语法树」，再只允许白名单内的运算节点，
        # 这样用户无论输入什么，最多只能做加减乘除，无法执行恶意代码。
        if isinstance(node, ast.Expression):
            # 整个表达式的最外层，去掉外壳继续求值内部内容
            return _safe_eval(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            # 遇到数字常量（如 15、27），直接返回该数字
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            # 遇到二元运算（如 15 * 27），先求左值、再求右值，最后做运算
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            return _SAFE_OPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            # 遇到一元运算符（如 -5），取操作数并加上正负号
            operand = _safe_eval(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        else:
            # 白名单以外的语法（如函数调用、变量赋值）一律拒绝
            raise ValueError(f"不支持的表达式类型: {type(node).__name__}")

    try:
        # 把字符串解析为 AST（抽象语法树），mode="eval" 表示只解析一个表达式
        tree = ast.parse(expression, mode="eval")
        # 对语法树进行安全求值
        result = _safe_eval(tree)
        return f"[计算结果] {expression} = {result}"
    except Exception as e:
        # 解析或计算失败时，返回错误信息（而不是抛出异常导致整个程序崩溃）
        return f"[计算错误] {e}"


# 注册可用工具
# 把上面定义的工具收集到列表里，LangGraph 的 ToolNode 和 LLM 绑定工具都会用到它
TOOLS = [web_search, calculator]


# ==============================================================================
# 状态定义
# ==============================================================================

class ReActState(TypedDict):
    """ReAct Agent 的状态，仅包含消息列表"""

    # TypedDict 是 Python 提供的「带类型的字典」，用来给字典里的字段做类型标注。
    #
    # messages: 对话过程中累积的所有消息（用户消息、AI 消息、工具返回消息）。
    # Annotated[list[BaseMessage], add_messages] 是 LangGraph 的「消息归约器」：
    #   - 普通情况下，节点返回的字典会「覆盖」状态字段；
    #   - 但加上 add_messages 后，新消息会「追加」进原来的列表，
    #     这样 agent 每次都能看到完整的对话历史，而不是只看到最新一条。
    messages: Annotated[list[BaseMessage], add_messages]


# 统一创建 LLM 模型实例
# 从 .env 中读取配置：模型名称、API Key、API 地址。
# 如果没读到就用默认值兜底（gpt-4o-mini）。
# temperature=0.7 控制输出的随机性，值越大回答越发散，越小越确定。
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0.7)


# ==============================================================================
# 节点函数
# ==============================================================================
# 节点函数是状态图的基本单元，输入是当前状态，输出是一个「要更新到状态里的字典」。

def agent_node(state: ReActState) -> dict:
    """
    agent 节点：让 LLM 推理并决定下一步行动

    Args:
        state: 当前图状态，包含消息列表

    Returns:
        包含新消息的字典，用于更新图状态
    """
    # bind_tools：把工具清单「绑定」到 LLM 上，
    # 这样模型在回答时不仅会生成文本，还可能在文本里附带 tool_calls（工具调用指令）。
    llm_with_tools = llm.bind_tools(TOOLS)
    # 把完整对话历史交给模型推理
    resp = llm_with_tools.invoke(state["messages"])
    # 返回新生成的 AI 消息；由于状态字段配了 add_messages，它会被追加到历史中
    return {"messages": [resp]}


def should_continue(state: ReActState) -> str:
    """
    条件边判断函数：决定下一步是继续调用工具还是结束

    Args:
        state: 当前图状态

    Returns:
        "tools" 表示继续调用工具，END 表示结束
    """
    # 只看最近一条消息（即 agent 节点刚生成的那条 AI 消息）
    last_msg = state["messages"][-1]
    # 如果 AI 消息里带有工具调用指令（tool_calls），说明模型还想用工具 → 去 tools 节点
    if last_msg.tool_calls:
        return "tools"
    # 否则说明模型已经给出最终答案、不再需要工具 → 结束整个流程
    return END


# ==============================================================================
# 构建 ReAct 状态图
# ==============================================================================

def build_react_graph():
    """
    构建并编译 ReAct 状态图

    Returns:
        编译后的 LangGraph 图对象（CompiledGraph）
    """
    # 1. 创建状态图，并指定状态类型 ReActState
    graph = StateGraph(ReActState)
    # 2. 添加两个节点：
    #    - "agent" 节点：负责推理（agent_node）
    #    - "tools" 节点：执行工具（ToolNode 是 LangGraph 内置的、自动执行 TOOLS 的节点）
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    # 3. 设置图的入口：流程一开始从 "agent" 节点进入
    graph.set_entry_point("agent")
    # 4. 添加条件边：从 "agent" 节点出发，根据 should_continue 的返回值决定去向
    #    - 返回 "tools" → 跳到 "tools" 节点执行工具
    #    - 返回 END    → 直接结束流程
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    # 5. 工具执行完毕后，回到 "agent" 节点继续推理（形成循环：
    #    agent →(要工具? no) 结束
    #            →(要工具? yes) tools → agent → ...
    graph.add_edge("tools", "agent")
    # 6. 编译成可运行的图（compile 会做校验，返回 CompiledGraph 对象）
    return graph.compile()


# ==============================================================================
# 运行入口
# ==============================================================================

def run(question: str) -> str:
    """
    运行 ReAct Agent，打印中间思考过程并返回最终答案

    Args:
        question: 用户的问题

    Returns:
        最终答案字符串
    """
    graph = build_react_graph()
    # 打印一个简单的分隔线，让控制台输出更清晰
    print(f"\n{'=' * 50}")
    print(f"用户问题: {question}")
    print(f"{'=' * 50}\n")

    final_answer = ""
    # 以 updates 模式逐步输出每个节点的状态更新，从而观察 ReAct 的思考过程
    # stream 会「流式」执行图：每跑完一个节点就 yield 一次，而不是全部跑完才返回，
    # 这样我们就能看到 agent 是怎么一步步思考、调用工具、拿到结果的。
    for chunk in graph.stream(
        {"messages": [HumanMessage(content=question)]},  # 初始状态：只有用户的一条问题
        stream_mode="updates",
    ):
        # 每个 chunk 形如 {节点名: 该节点返回的状态更新}
        for node_name, update in chunk.items():
            print(f"Node: {node_name}")
            # print(f"\n[{node_name}] {update}")
            for msg in update.get("messages", []):
                # AI 消息：展示模型的思考与工具调用决策
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        # 模型决定调用工具：打印工具名和参数
                        for tc in msg.tool_calls:
                            print(f"Thought: 调用工具 {tc['name']}，参数 {tc['args']}")
                    elif msg.content:
                        # 模型给出最终文字答案
                        final_answer = msg.content
                        print(f"Final Answer: {msg.content}")
                # 工具消息：展示工具执行结果
                elif isinstance(msg, ToolMessage):
                    print(f"Observation ({msg.name}): {msg.content}\n")

    return final_answer

def main():
    """演示 LangGraph ReAct Agent 的使用"""
    # 这个问题同时用到了「计算」和「搜索」两个工具，很适合演示 ReAct 的循环过程：
    # 1. 模型先调用 calculator 算出 15 * 27 = 405；
    # 2. 再调用 web_search 搜索 405 的含义；
    # 3. 最后综合信息给出最终答案。
    question = "计算 15 * 27 的结果，然后搜索了解一下这个数字的含义"
    answer = run(question)
    print(f"\n{'='*50}")
    print(f"最终答案: {answer}")


# 当直接运行本文件（python agent.py）时才执行 main()；
# 如果是被其它文件 import，则不会执行，避免副作用。
if __name__ == "__main__":
    main()
