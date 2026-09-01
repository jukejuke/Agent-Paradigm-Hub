"""
ReAct (Reason + Act) Agent - LangChain 实现
==============================================

使用 LangChain 框架实现的 ReAct Agent。
LangChain 内置了 AgentExecutor 来处理 ReAct 逻辑。
"""

from typing import Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


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

def create_react_agent_executor(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_iterations: int = 10,
) -> AgentExecutor:
    """
    创建 LangChain ReAct Agent 执行器

    Args:
        model_name: 使用的模型名称
        temperature: 温度参数
        max_iterations: 最大迭代次数

    Returns:
        配置好的 AgentExecutor 实例
    """
    # 初始化 LLM
    llm = ChatOpenAI(model=model_name, temperature=temperature)

    # 使用 LangChain 内置的 ReAct prompt
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = ChatPromptTemplate.from_template(template)

    # 创建 ReAct Agent
    agent = create_react_agent(llm, TOOLS, prompt)

    # 创建 AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
    )

    return agent_executor


def run(question: str, agent_executor: Optional[AgentExecutor] = None) -> str:
    """
    运行 ReAct Agent 解决问题

    Args:
        question: 用户的问题
        agent_executor: Agent 执行器，默认创建新实例

    Returns:
        最终答案字符串
    """
    if agent_executor is None:
        agent_executor = create_react_agent_executor()

    result = agent_executor.invoke({"input": question})
    return result["output"]


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
