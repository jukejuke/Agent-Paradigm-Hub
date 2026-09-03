"""
Routing - LangGraph 实现
=========================
核心思想：先判断用户请求的类别（路由），再分发到对应的专业处理器。
流程：router（LLM 判断类别）-> 条件边分发 -> general_qa / code_helper /
creative_writing / data_analysis 四个专业 handler -> 写入 final 并结束。
"""

import os
import json

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from pathlib import Path
from dotenv import load_dotenv

# 定位项目根目录并加载 .env（支持从任意目录直接运行本文件）
load_dotenv(Path(__file__).resolve().parents[4] / ".env")


# ==============================================================================
# 状态与路由定义
# ==============================================================================

class RoutingState(TypedDict):
    """Routing 图状态：request 用户请求、route 路由类别、final 最终回复。"""

    request: str  # 用户请求
    route: str    # 路由判断出的类别 key
    final: str    # 最终回复


# 统一模型：默认 gpt-4o-mini，可用环境变量 OPENAI_MODEL 覆盖
_LLM = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0.7)

# 4 种路由及对应中文 handler system prompt
_HANDLER_PROMPTS = {
    "general_qa": "你是一个博学助手，请用通俗易懂的方式回答用户的问题。",
    "code_helper": "你是一个资深工程师，擅长代码编写和调试。请给出完整可运行的代码和解释。",
    "creative_writing": "你是一个创意写作专家，文笔生动有趣，善于打动读者。",
    "data_analysis": "你是一个数据分析师，注重逻辑和数据支撑，给出清晰的分析思路。",
}

# 路由判断中文 prompt：仅输出 JSON
_ROUTING_SYSTEM = (
    "你是一个请求路由器。请判断用户请求属于以下哪个类别：\n"
    "- general_qa: 一般性的问答、解释、建议\n"
    "- code_helper: 代码编写、调试、重构、技术问题\n"
    "- creative_writing: 文案创作、故事生成、营销内容\n"
    "- data_analysis: 数据分析、统计问题、数字处理\n\n"
    '请仅输出 JSON 格式：{"route": "类别key", "reason": "选择理由"}，不要输出其他内容。'
)


# ==============================================================================
# 节点实现
# ==============================================================================

def router_node(state: RoutingState) -> dict:
    """路由节点：LLM 判断请求类别。Args: state 当前图状态。Returns: {"route": 类别key}，解析失败 fallback general_qa。"""
    messages = [
        SystemMessage(content=_ROUTING_SYSTEM),
        HumanMessage(content=state["request"]),
    ]
    content = ""
    for chunk in _LLM.stream(messages):
        content += chunk.content

    route_key = "general_qa"
    try:
        # 截取首个 { 到最后一个 } 之间的 JSON 片段并解析
        start = content.find("{")
        end = content.rfind("}") + 1
        route_key = json.loads(content[start:end]).get("route", "general_qa")
    except (json.JSONDecodeError, ValueError):
        print(f"Warning: 路由解析失败，默认走 general_qa。原始回复: {content}")

    # 兜底：非法 key 统一回退
    if route_key not in _HANDLER_PROMPTS:
        route_key = "general_qa"

    return {"route": route_key}


def route_decision(state: RoutingState) -> str:
    """条件边决策：返回对应 handler 节点名。Args: state 当前图状态。Returns: 节点名，非法时回退 general_qa。"""
    route = state.get("route", "general_qa")
    return route if route in _HANDLER_PROMPTS else "general_qa"


def general_qa_node(state: RoutingState) -> dict:
    """通用问答处理器（博学助手）。Args: state 当前图状态。Returns: {"final": 处理结果}。"""
    messages = [SystemMessage(content=_HANDLER_PROMPTS["general_qa"]), HumanMessage(content=state["request"])]
    final = ""
    for chunk in _LLM.stream(messages):
        final += chunk.content
    return {"final": final}


def code_helper_node(state: RoutingState) -> dict:
    """编程助手处理器（资深工程师）。Args: state 当前图状态。Returns: {"final": 处理结果}。"""
    messages = [SystemMessage(content=_HANDLER_PROMPTS["code_helper"]), HumanMessage(content=state["request"])]
    final = ""
    for chunk in _LLM.stream(messages):
        final += chunk.content
    return {"final": final}


def creative_writing_node(state: RoutingState) -> dict:
    """创意写作处理器（创意写作专家）。Args: state 当前图状态。Returns: {"final": 处理结果}。"""
    messages = [SystemMessage(content=_HANDLER_PROMPTS["creative_writing"]), HumanMessage(content=state["request"])]
    final = ""
    for chunk in _LLM.stream(messages):
        final += chunk.content
    return {"final": final}


def data_analysis_node(state: RoutingState) -> dict:
    """数据分析处理器（数据分析师）。Args: state 当前图状态。Returns: {"final": 处理结果}。"""
    messages = [SystemMessage(content=_HANDLER_PROMPTS["data_analysis"]), HumanMessage(content=state["request"])]
    final = ""
    for chunk in _LLM.stream(messages):
        final += chunk.content
    return {"final": final}


# ==============================================================================
# 图构建与入口
# ==============================================================================

def build_routing_graph():
    """构建并编译 Routing 图：router -> 条件边分发 -> 4 个 handler -> END。Returns: 编译后的 LangGraph 图对象。"""
    graph = StateGraph(RoutingState)

    graph.add_node("router", router_node)
    graph.add_node("general_qa", general_qa_node)
    graph.add_node("code_helper", code_helper_node)
    graph.add_node("creative_writing", creative_writing_node)
    graph.add_node("data_analysis", data_analysis_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "general_qa": "general_qa",
            "code_helper": "code_helper",
            "creative_writing": "creative_writing",
            "data_analysis": "data_analysis",
        },
    )
    graph.add_edge("general_qa", END)
    graph.add_edge("code_helper", END)
    graph.add_edge("creative_writing", END)
    graph.add_edge("data_analysis", END)

    return graph.compile()


def run(request: str) -> str:
    """运行完整的 Routing 工作流，流式打印底层 LLM 输出。Args: request 用户请求。Returns: 最终回复。"""
    graph = build_routing_graph()
    print(f"\n{'=' * 50}")
    print(f"请求: {request}")
    print(f"{'=' * 50}\n")

    final_state = None
    # updates 模式打印每个节点的状态更新，values 模式用于取最终状态
    for mode, payload in graph.stream({"request": request}, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node_name, update in payload.items():
                print(f"\n[{node_name}] {update}")
        else:  # mode == "values"
            final_state = payload

    print("\n")
    return final_state["final"]


def main():
    """演示 LangGraph Routing 的使用。"""
    test_requests = [
        "用 Python 写一个快速排序算法",
        "帮我想一个奶茶店的 slogan",
        "请解释一下什么是 Transformer 架构",
    ]

    for req in test_requests:
        print(f"\n{'#' * 50}")
        print(f"请求: {req}")
        result = run(req)
        print(f"最终回复:\n{result}")


if __name__ == "__main__":
    main()
