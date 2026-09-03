"""
Plan-and-Execute - LangGraph 实现
====================================

核心思想：先规划，后执行，最后汇总。
第一阶段让 LLM 输出完成任务的完整步骤列表 JSON；
第二阶段在节点内部逐步执行每个步骤；
第三阶段基于问题、计划与执行结果生成最终总结。

与 ReAct 的区别：ReAct 是边想边做，Plan-and-Execute 是先想好了再做。
"""

import os, json
from typing import TypedDict, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

from pathlib import Path
from dotenv import load_dotenv

# 定位项目根目录并加载 .env（支持从任意目录直接运行本文件）
load_dotenv(Path(__file__).resolve().parents[4] / ".env")


# ==============================================================================
# 状态定义
# ==============================================================================

class PlanExecuteState(TypedDict):
    """Plan-and-Execute 图的状态定义"""

    question: str          # 用户问题
    plan: list[str]        # 规划步骤列表
    results: list[str]     # 每步执行结果
    final: str             # 最终答案


# 统一模型实例
LLM = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0.7)


# ==============================================================================
# 节点函数
# ==============================================================================

def planner_node(state: PlanExecuteState) -> dict:
    """
    规划节点：让 LLM 输出完成任务的步骤列表 JSON。

    Args:
        state: 当前图状态，包含用户问题 question。

    Returns:
        字典 {"plan": [...]}，更新后的计划步骤列表。
    """
    question = state["question"]

    prompt = (
        "你是一个任务规划专家。请为用户的问题制定清晰的执行步骤。\n"
        "要求：仅输出一个 JSON 数组字符串，数组中的每个元素是一个步骤描述。\n"
        '例如：["步骤1：收集相关信息", "步骤2：分析信息", "步骤3：给出结论"]\n\n'
        f"用户问题：{question}"
    )

    response = LLM.invoke([HumanMessage(content=prompt)])
    raw = str(response.content)

    # 解析 JSON 步骤列表，失败时回退到默认三步
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        steps = json.loads(raw[start:end])
        if not isinstance(steps, list) or not steps:
            raise ValueError("解析结果为空或非列表")
    except (json.JSONDecodeError, ValueError, TypeError):
        print("Warning: JSON 解析失败，使用默认三步计划")
        steps = [
            f"收集与「{question}」相关的背景信息",
            "整理并分析收集到的信息",
            "组织语言，输出最终结论",
        ]

    print(f"📋 生成的计划: {steps}")
    return {"plan": steps}


def executor_node(state: PlanExecuteState) -> dict:
    """
    执行节点：在节点内部遍历计划步骤，逐步调用 LLM 执行。

    Args:
        state: 当前图状态，包含 question 和 plan。

    Returns:
        字典 {"results": [...]}，每一步的执行结果列表。
    """
    question = state["question"]
    plan = state["plan"]
    results = []

    # 逐步执行计划中的每个步骤
    for step in plan:
        messages = [
            SystemMessage(content="你是执行者，只完成当前这一步任务。"),
            HumanMessage(content=f"原始问题：{question}\n当前步骤：{step}"),
        ]
        response = LLM.invoke(messages)
        results.append(str(response.content))
        print(f"--- 执行步骤完成: {step}")

    return {"results": results}


def summarizer_node(state: PlanExecuteState) -> dict:
    """
    汇总节点：基于问题、计划与执行结果生成最终总结。

    Args:
        state: 当前图状态，包含 question、plan 和 results。

    Returns:
        字典 {"final": ...}，最终答案文本。
    """
    question = state["question"]
    plan = state["plan"]
    results = state["results"]

    messages = [
        SystemMessage(content="你是总结者，请基于计划与执行结果给出最终答案。"),
        HumanMessage(
            content=(
                f"用户问题：{question}\n"
                f"执行计划：{json.dumps(plan, ensure_ascii=False)}\n"
                f"执行结果：{json.dumps(results, ensure_ascii=False)}\n\n"
                "请输出最终总结。"
            )
        ),
    ]
    response = LLM.invoke(messages)
    return {"final": str(response.content)}


# ==============================================================================
# 图构建与入口
# ==============================================================================

def build_plan_execute_graph():
    """
    构建并编译 Plan-and-Execute 图。

    Returns:
        编译后的 LangGraph 图对象。
    """
    builder = StateGraph(PlanExecuteState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("summarizer", summarizer_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "summarizer")
    builder.add_edge("summarizer", END)

    return builder.compile()


def run(question: str) -> str:
    """
    运行完整的 Plan-and-Execute 流程。

    Args:
        question: 用户问题。

    Returns:
        最终答案文本。
    """
    graph = build_plan_execute_graph()
    result = graph.invoke({"question": question})
    return result["final"]


def main():
    """演示 Plan-and-Execute LangGraph 实现。"""
    question = "写一份关于 AI Agent 发展趋势的简短报告"
    answer = run(question)
    print("=" * 50)
    print(f"最终答案:\n{answer}")


if __name__ == "__main__":
    main()
