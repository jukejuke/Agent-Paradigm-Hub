"""
Reflection - LangGraph 实现
============================

核心思想：让 LLM 先生成答案初稿，再以「严厉批评者」角色对初稿进行
自我批评，最后根据批评意见改进答案，循环迭代直到满意或达到最大迭代次数。

流程：generate（生成初稿）-> reflect（自我批评）-> refine（按批评改进）
      -> reflect -> refine -> ... 直到 should_continue 判定结束。
"""

import os

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from pathlib import Path
from dotenv import load_dotenv

# 定位项目根目录并加载 .env（支持从任意目录直接运行本文件）
load_dotenv(Path(__file__).resolve().parents[4] / ".env")


# ==============================================================================
# 状态定义
# ==============================================================================

class ReflectionState(TypedDict):
    """Reflection 图状态"""
    question: str      # 用户问题
    draft: str         # 当前答案（初稿或改进稿）
    critique: str      # 批评意见
    iteration: int     # 当前迭代轮次


# ==============================================================================
# 模型与提示词
# ==============================================================================

# 统一模型：默认 gpt-4o-mini，可用环境变量 OPENAI_MODEL 覆盖
_LLM = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.7,
)

_GENERATE_SYSTEM = "你是一个善于解决问题的助手。请直接给出答案初稿，无需追求完美。"

_REFLECT_SYSTEM = (
    "你是一个严厉的批评者。请审视下面的回答，找出其中的逻辑错误、事实错误、"
    "遗漏的关键点以及可以改进的地方。请用中文逐条列出批评意见。"
    "如果回答已经很好，请说「答案已经很好，无需改进」。"
)

_REFINE_SYSTEM = (
    "你是一个改进专家。请根据批评意见逐条改进下面的回答，"
    "确保每个问题都得到解决，并输出改进后的完整答案。"
)


# ==============================================================================
# 图节点
# ==============================================================================

def generate_node(state: ReflectionState) -> dict:
    """
    生成节点：LLM 根据问题生成答案初稿

    Args:
        state: 当前图状态

    Returns:
        包含初稿 draft 与初始迭代次数 iteration=1 的状态更新
    """
    messages = [
        SystemMessage(content=_GENERATE_SYSTEM),
        HumanMessage(content=state["question"]),
    ]
    draft = ""
    for chunk in _LLM.stream(messages):
        draft += chunk.content
    return {"draft": draft, "iteration": 1}


def reflect_node(state: ReflectionState) -> dict:
    """
    反思节点：LLM 以「严厉批评者」角色审阅 draft，输出中文批评意见

    Args:
        state: 当前图状态

    Returns:
        包含批评意见 critique 的状态更新
    """
    messages = [
        SystemMessage(content=_REFLECT_SYSTEM),
        HumanMessage(content=f"问题: {state['question']}\n\n回答:\n{state['draft']}\n\n请批评。"),
    ]
    critique = ""
    for chunk in _LLM.stream(messages):
        critique += chunk.content
    return {"critique": critique}


def refine_node(state: ReflectionState) -> dict:
    """
    改进节点：LLM 根据 critique 改进 draft，并递增迭代次数

    Args:
        state: 当前图状态

    Returns:
        包含改进后 draft 与 iteration+1 的状态更新
    """
    messages = [
        SystemMessage(content=_REFINE_SYSTEM),
        HumanMessage(
            content=(
                f"问题: {state['question']}\n\n"
                f"上一轮回答:\n{state['draft']}\n\n"
                f"批评意见:\n{state['critique']}\n\n请改进。"
            )
        ),
    ]
    refined = ""
    for chunk in _LLM.stream(messages):
        refined += chunk.content
    return {"draft": refined, "iteration": state["iteration"] + 1}


# ==============================================================================
# 路由函数
# ==============================================================================

def should_continue(state: ReflectionState):
    """
    判断是否继续反思循环

    Args:
        state: 当前图状态

    Returns:
        "refine" 表示继续改进，END 表示结束循环
    """
    critique = state["critique"]
    # 批评意见表明答案已足够好时结束
    for keyword in ("已经很好", "无需改进", "没有明显问题"):
        if keyword in critique:
            return END
    # 达到最大迭代次数时结束
    if state["iteration"] >= 3:
        return END
    return "refine"


# ==============================================================================
# 图构建
# ==============================================================================

def build_reflection_graph():
    """
    构建 Reflection 图

    Returns:
        编译后的 LangGraph 图
    """
    graph = StateGraph(ReflectionState)

    # 添加节点
    graph.add_node("generate", generate_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("refine", refine_node)

    # 入口为 generate
    graph.set_entry_point("generate")

    # generate -> reflect
    graph.add_edge("generate", "reflect")

    # reflect 后根据 should_continue 决定：改进或结束
    graph.add_conditional_edges(
        "reflect",
        should_continue,
        {"refine": "refine", END: END},
    )

    # refine -> reflect，形成「反思-改进」循环
    graph.add_edge("refine", "reflect")

    return graph.compile()


# ==============================================================================
# 入口函数
# ==============================================================================

def run(question: str) -> str:
    """
    运行 Reflection Agent

    Args:
        question: 用户问题

    Returns:
        经过反思改进后的最终答案
    """
    graph = build_reflection_graph()
    print(f"\n{'=' * 50}")
    print(f"用户问题: {question}")
    print(f"{'=' * 50}\n")

    final_state = None
    # updates 模式打印每个节点的状态更新，values 模式用于取最终状态
    for mode, payload in graph.stream({"question": question}, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node_name, update in payload.items():
                print(f"\n[{node_name}] {update}")
        else:  # mode == "values"
            final_state = payload

    print("\n")
    return final_state["draft"]


def main():
    """演示 LangGraph Reflection 的使用"""
    question = "写一段关于「为什么程序员应该学习 AI」的短文"
    answer = run(question)
    print(f"{'=' * 50}")
    print(f"最终答案:\n{answer}")


if __name__ == "__main__":
    main()
