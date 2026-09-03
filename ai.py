import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage

load_dotenv()

# ===================== 1. 定义工具 =====================
@tool
def search(query: str) -> str:
    """搜索外部信息，用于查询事实数据
    Args:
        query: 查询关键词
    """
    mock_db = {
        "2025成都GDP": "2025年成都GDP约2.45万亿元",
        "成都人口2025": "2025成都常住人口2180万"
    }
    return mock_db.get(query, f"搜索结果：未查询到【{query}】相关数据")


@tool
def calculate(expr: str) -> str:
    """数学表达式计算
    Args:
        expr: 数学表达式，例如 "24500 / 2180"
    """
    try:
        res = eval(expr, {"__builtins__": None}, {})
        return f"计算结果：{res:.4f}"
    except Exception as e:
        return f"计算错误: {str(e)}"


tools = [search, calculate]

# ===================== 2. Graph State 状态定义 =====================
class ReActState(TypedDict):
    # ⚠️ add_messages reducer：把新旧 messages 列表**拼接**而不是覆盖
    # 没有它的话，LangGraph 默认只保留最新节点产出的 messages，历史 HumanMessage / AIMessage 会丢
    messages: Annotated[list[BaseMessage], add_messages]


# ===================== 3. Agent节点：大模型推理节点 =====================
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.getenv("OPENAI_API_KEY","sk-192744cf18564c6c8a6a947726948f2b"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
).bind_tools(tools)


def agent_node(state: ReActState):
    """ReAct Thought 节点：模型思考，输出tool_call或者最终回答"""
    messages = state["messages"]
    resp = llm.invoke(messages)
    return {"messages": [resp]}


# ===================== 4. 路由函数：判断是否继续循环 =====================
def should_continue(state: ReActState):
    """
    ReAct循环终止判断
    return: "tools" 继续调用工具 | END 结束
    """
    last_msg = state["messages"][-1]
    # 如果存在tool_calls，就走到工具节点；否则结束
    if last_msg.tool_calls:
        return "tools"
    return END


# ===================== 5. 构建ReAct图 =====================
def build_react_graph():
    graph_builder = StateGraph(ReActState)

    # 添加节点
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(tools))

    # 入口
    graph_builder.set_entry_point("agent")

    # 条件边：agent执行完，判断走工具还是结束
    graph_builder.add_conditional_edges(
        source="agent",
        path=should_continue,
        path_map={
            "tools": "tools",
            END: END
        }
    )
    # 工具执行完，回到agent继续思考（ReAct循环）
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile()


if __name__ == "__main__":
    react_agent = build_react_graph()

    user_query = "成都2025年人均GDP是多少？"
    # ⚠️ 必须传 HumanMessage 对象，不能传 tuple！
    # tuple 会被 LangGraph 原样存入 state，后续 ToolNode/agent_node 迭代就会崩
    result = react_agent.invoke({
        "messages": [HumanMessage(content=user_query)]
    })

    print("==== 完整对话 ====")
    for m in result["messages"]:
        m.pretty_print()

    print("\n==== 最终回答 ====")
    print(result["messages"][-1].content)
