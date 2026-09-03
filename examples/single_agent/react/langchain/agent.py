"""
ReAct (Reason + Act) Agent - LangChain 实现
==============================================

使用 LangChain 框架实现的 ReAct Agent。
LangChain 1.x 使用 create_agent（底层基于 langgraph 状态图）来处理 ReAct 逻辑。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage

# 定位项目根目录并加载 .env（支持从任意目录直接运行本文件）
load_dotenv(Path(__file__).resolve().parents[4] / ".env")


# ==============================================================================
# 定义工具
# ==============================================================================

@tool
def web_search(query: str) -> str:
    """搜索网络获取信息"""
    return f"[搜索结果] 关于 '{query}' 的信息：这是一个模拟的搜索结果。"


@tool
def calculator(expression: str) -> str:
    """安全计算数学表达式（使用 ast 解析，避免 eval 安全风险）"""
    import ast
    import operator

    _SAFE_OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }

    def _safe_eval(node):
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = _safe_eval(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        else:
            raise ValueError(f"不支持的表达式类型: {type(node).__name__}")

    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
        return f"[计算结果] {expression} = {result}"
    except Exception as e:
        return f"[计算错误] {e}"


TOOLS = [web_search, calculator]


# ==============================================================================
# ReAct Agent 实现
# ==============================================================================

def build_react_agent(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
):
    """
    创建基于 LangChain create_agent 的 ReAct Agent（底层为 langgraph 状态图）

    Args:
        model_name: 使用的模型名称
        temperature: 温度参数

    Returns:
        编译后的 ReAct Agent 状态图
    """
    # 初始化 LLM（从 .env 读取 api_key / base_url / model）
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", model_name),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=temperature,
    )

    # 使用 LangChain 1.x 的 create_agent 构建 ReAct Agent
    # 工具调用协议由框架自动处理，system_prompt 仅约束推理/输出风格
    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=(
            "你是一个 ReAct（Reason + Act）智能体。请遵循以下格式逐步推理并解决问题：\n"
            "Thought: 思考下一步该做什么\n"
            "Action: 选择要调用的工具\n"
            "Action Input: 工具入参\n"
            "Observation: 工具返回结果\n"
            "……（可多轮重复）\n"
            "Final Answer: 给出最终答案"
        ),
    )

    return agent


def run(question: str, agent=None) -> str:
    """
    运行 ReAct Agent，打印中间思考过程并返回最终答案

    Args:
        question: 用户的问题
        agent: 编译后的 ReAct Agent，默认创建新实例

    Returns:
        最终答案字符串
    """
    if agent is None:
        agent = build_react_agent()

    final_answer = ""
    # 以 updates 模式逐步输出每个节点的状态更新，从而观察 ReAct 的思考过程
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
    ):
        for update in chunk.values():
            for msg in update.get("messages", []):
                # AI 消息：展示模型的思考与工具调用决策
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"Thought: 调用工具 {tc['name']}，参数 {tc['args']}")
                    elif msg.content:
                        final_answer = msg.content
                        print(f"Final Answer: {msg.content}")
                # 工具消息：展示工具执行结果
                elif isinstance(msg, ToolMessage):
                    print(f"Observation ({msg.name}): {msg.content}\n")

    return final_answer


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 LangChain ReAct Agent 的使用"""
    answer = run("计算 15 * 27 的结果，然后搜索了解一下这个数字的含义")
    print(f"\n{'='*50}")
    print(f"最终答案: {answer}")


if __name__ == "__main__":
    main()
