"""
Multi-Agent Debate - LangGraph 实现
====================================

核心思想：正方与反方围绕辩题进行多轮辩论，
最后由中立裁判综合双方观点进行裁决。

使用 LangGraph 的 StateGraph 将辩论流程建模为状态图：
正方发言 -> 反方发言 -> ... -> 裁判裁决。
"""

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import os

from pathlib import Path
from dotenv import load_dotenv

# 定位项目根目录并加载 .env（支持从任意目录直接运行本文件）
load_dotenv(Path(__file__).resolve().parents[4] / ".env")


# ==============================================================================
# 状态定义
# ==============================================================================

class DebateState(TypedDict):
    """辩论状态"""
    topic: str        # 辩论主题
    history: str      # 对话历史累积
    round: int        # 当前轮次
    verdict: str      # 裁判裁决结果


# ==============================================================================
# 节点实现
# ==============================================================================

def _get_llm() -> ChatOpenAI:
    """
    获取统一的 LLM 实例

    Returns:
        ChatOpenAI 实例
    """
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0.7)


def pro_node(state: DebateState) -> dict:
    """
    正方发言节点

    Args:
        state: 当前辩论状态

    Returns:
        更新后的 history 与 round
    """
    llm = _get_llm()
    messages = [
        SystemMessage(
            content=(
                "你是辩论的正方。请坚定地支持辩题的观点，有理有据地阐述理由，"
                "并针对反方可能提出的质疑进行反驳。发言控制在 150 字以内。"
            )
        ),
        HumanMessage(
            content=f"辩论主题: {state['topic']}\n\n前序辩论:\n{state['history']}\n\n请正方发言:"
        ),
    ]
    content = ""
    for chunk in llm.stream(messages):
        content += chunk.content
    return {
        "history": state["history"] + "\n正方: " + content,
        "round": state["round"] + 1,
    }


def con_node(state: DebateState) -> dict:
    """
    反方发言节点

    Args:
        state: 当前辩论状态

    Returns:
        更新后的 history
    """
    llm = _get_llm()
    messages = [
        SystemMessage(
            content=(
                "你是辩论的反方。请坚定地反对辩题的观点，有理有据地阐述理由，"
                "并针对正方提出的论点进行反驳。发言控制在 150 字以内。"
            )
        ),
        HumanMessage(
            content=f"辩论主题: {state['topic']}\n\n前序辩论:\n{state['history']}\n\n请反方发言:"
        ),
    ]
    content = ""
    for chunk in llm.stream(messages):
        content += chunk.content
    return {
        "history": state["history"] + "\n反方: " + content,
    }


def should_continue(state: DebateState) -> str:
    """
    判断辩论是否继续

    Args:
        state: 当前辩论状态

    Returns:
        "judge" 表示进入裁判裁决，"pro" 表示继续下一轮
    """
    if state["round"] >= 2:
        return "judge"
    return "pro"


def judge_node(state: DebateState) -> dict:
    """
    裁判裁决节点

    Args:
        state: 当前辩论状态

    Returns:
        更新后的 verdict
    """
    llm = _get_llm()
    messages = [
        SystemMessage(
            content=(
                "你是一位中立公正的辩论裁判。请通读完整的辩论记录，总结双方核心论点，"
                "评价双方论证的强弱，并给出综合判断（不偏袒任何一方）。"
            )
        ),
        HumanMessage(
            content=f"辩论主题: {state['topic']}\n\n完整辩论记录:\n{state['history']}\n\n请给出裁决:"
        ),
    ]
    verdict = ""
    for chunk in llm.stream(messages):
        verdict += chunk.content
    return {"verdict": verdict}


# ==============================================================================
# 图构建与入口
# ==============================================================================

def build_debate_graph():
    """
    构建辩论状态图

    Returns:
        编译后的 LangGraph 图
    """
    graph = StateGraph(DebateState)

    graph.add_node("pro", pro_node)
    graph.add_node("con", con_node)
    graph.add_node("judge", judge_node)

    graph.set_entry_point("pro")
    graph.add_edge("pro", "con")
    graph.add_conditional_edges("con", should_continue, {"pro": "pro", "judge": "judge"})
    graph.add_edge("judge", END)

    return graph.compile()


def run(topic: str) -> str:
    """
    运行辩论并返回裁决结果

    Args:
        topic: 辩论主题

    Returns:
        裁判裁决文本
    """
    graph = build_debate_graph()
    print(f"\n{'=' * 50}")
    print(f"辩论主题: {topic}")
    print(f"{'=' * 50}\n")

    final_state = None
    # updates 模式打印每个节点的状态更新，values 模式用于取最终状态
    for mode, payload in graph.stream(
        {"topic": topic, "round": 0, "history": ""},
        stream_mode=["updates", "values"],
    ):
        if mode == "updates":
            for node_name, update in payload.items():
                print(f"\n[{node_name}] {update}")
        else:  # mode == "values"
            final_state = payload

    print("\n")
    return final_state["verdict"]


def main():
    """演示入口：以固定辩题运行辩论并打印裁决结果"""
    topic = "AI 是否会取代大多数程序员的工作"
    verdict = run(topic)
    print(f"辩论主题: {topic}")
    print(f"裁判裁决:\n{verdict}")


if __name__ == "__main__":
    main()
