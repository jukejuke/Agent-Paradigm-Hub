"""
ReAct (Reason + Act) Agent - LangGraph 实现
==============================================

核心思想：推理（Reasoning）与行动（Acting）交替进行。
LangGraph 使用状态图（StateGraph）显式建模 agent 与工具之间的循环：
- agent 节点：由 LLM 推理并决定下一步行动（是否调用工具）；
- tools 节点：执行 agent 选择调用的工具；
两者通过条件边连接，循环往复，直到 agent 给出最终答案。

参考: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
"""

import os
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage

from pathlib import Path
from dotenv import load_dotenv

# 定位项目根目录并加载 .env（支持从任意目录直接运行本文件）
load_dotenv(Path(__file__).resolve().parents[4] / ".env")


# ==============================================================================
# 定义工具
# ==============================================================================

@tool
def web_search(query: str) -> str:
    """
    模拟网络搜索工具

    Args:
        query: 搜索关键词

    Returns:
        中文模拟搜索结果
    """
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
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            return _SAFE_OPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = _safe_eval(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        else:
            raise ValueError(f"不支持的表达式类型: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return f"[计算结果] {expression} = {result}"
    except Exception as e:
        return f"[计算错误] {e}"


# 注册可用工具
TOOLS = [web_search, calculator]


# ==============================================================================
# 状态定义
# ==============================================================================

class ReActState(TypedDict):
    """ReAct Agent 的状态，仅包含消息列表"""

    messages: Annotated[list[BaseMessage], add_messages]


# 统一创建 LLM 模型实例
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0.7)


# ==============================================================================
# 节点函数
# ==============================================================================

def agent_node(state: ReActState) -> dict:
    """
    agent 节点：让 LLM 推理并决定下一步行动

    Args:
        state: 当前图状态，包含消息列表

    Returns:
        包含新消息的字典，用于更新图状态
    """
    llm_with_tools = llm.bind_tools(TOOLS)
    resp = llm_with_tools.invoke(state["messages"])
    return {"messages": [resp]}


def should_continue(state: ReActState) -> str:
    """
    条件边判断函数：决定下一步是继续调用工具还是结束

    Args:
        state: 当前图状态

    Returns:
        "tools" 表示继续调用工具，END 表示结束
    """
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
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
    graph = StateGraph(ReActState)
    # 添加 agent 节点（推理）与 tools 节点（执行工具）
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    # 入口为 agent 节点
    graph.set_entry_point("agent")
    # agent 节点根据 should_continue 决定去 tools 还是结束
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    # 工具执行完毕后回到 agent 节点继续推理
    graph.add_edge("tools", "agent")
    return graph.compile()


# ==============================================================================
# 运行入口
# ==============================================================================

def run(question: str) -> str:
    """
    运行 ReAct Agent 并返回最终答案

    Args:
        question: 用户的问题

    Returns:
        最终答案字符串
    """
    graph = build_react_graph()
    result = graph.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content

def main():
    """演示 LangGraph ReAct Agent 的使用"""
    question = "计算 15 * 27 的结果，然后搜索了解一下这个数字的含义"
    answer = run(question)
    print(f"\n{'='*50}")
    print(f"最终答案: {answer}")


if __name__ == "__main__":
    main()
