"""
Parallelization - LangGraph 实现
=================================
核心思想：将一个大任务拆分成多个相互独立的子任务，
并行执行（通过 LangGraph 分支节点实现），最后汇总结果。
"""

import os
import json
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class ParallelizationState(TypedDict):
    """Parallelization 图的状态定义"""

    task: str            # 主任务
    subtasks: list[str]  # 拆分后的子任务列表
    result_1: str        # 子任务 1 的处理结果
    result_2: str        # 子任务 2 的处理结果
    result_3: str        # 子任务 3 的处理结果
    final: str           # 最终汇总答案


# 统一模型实例
LLM = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.7)


def split_node(state: ParallelizationState) -> dict:
    """拆分节点：将主任务拆分成 3 个相互独立的子任务。

    Args:
        state: 当前图状态，包含主任务 task。

    Returns:
        字典 {"subtasks": [...]}，3 个独立子任务列表。
    """
    task = state["task"]
    prompt = (
        "你是一个任务拆分专家。请将下面的主任务拆分成 3 个相互独立、"
        "可以并行执行的子任务。\n"
        "要求：仅输出一个 JSON 数组字符串，数组长度为 3，每个元素是一个子任务描述。\n"
        '例如：["子任务1", "子任务2", "子任务3"]\n\n'
        f"主任务：{task}"
    )
    response = LLM.invoke([HumanMessage(content=prompt)])
    raw = str(response.content)
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        subtasks = json.loads(raw[start:end])
        if not isinstance(subtasks, list) or not subtasks:
            raise ValueError("解析结果为空或非列表")
        subtasks = [str(s) for s in subtasks[:3]]
    except (json.JSONDecodeError, ValueError, TypeError):
        print("Warning: JSON 解析失败，使用默认 3 个子任务")
        subtasks = [
            f"收集「{task}」相关的背景信息",
            f"分析「{task}」的关键要点",
            f"总结并组织「{task}」的最终结论",
        ]
    while len(subtasks) < 3:
        subtasks.append(f"补充分析「{task}」的剩余要点")
    print(f"📋 拆分的子任务: {subtasks}")
    return {"subtasks": subtasks}


def worker_1_node(state: ParallelizationState) -> dict:
    """并行工作节点 1：处理第一个子任务。

    Args:
        state: 当前图状态。

    Returns:
        {"result_1": 第一个子任务的处理结果}。
    """
    messages = [
        SystemMessage(content="你是高效的处理者，只完成分配给你的子任务。"),
        HumanMessage(content=state["subtasks"][0]),
    ]
    response = LLM.invoke(messages)
    print("✅ [子任务 1] 完成")
    return {"result_1": str(response.content)}


def worker_2_node(state: ParallelizationState) -> dict:
    """并行工作节点 2：处理第二个子任务。

    Args:
        state: 当前图状态。

    Returns:
        {"result_2": 第二个子任务的处理结果}。
    """
    messages = [
        SystemMessage(content="你是高效的处理者，只完成分配给你的子任务。"),
        HumanMessage(content=state["subtasks"][1]),
    ]
    response = LLM.invoke(messages)
    print("✅ [子任务 2] 完成")
    return {"result_2": str(response.content)}


def worker_3_node(state: ParallelizationState) -> dict:
    """并行工作节点 3：处理第三个子任务。

    Args:
        state: 当前图状态。

    Returns:
        {"result_3": 第三个子任务的处理结果}。
    """
    messages = [
        SystemMessage(content="你是高效的处理者，只完成分配给你的子任务。"),
        HumanMessage(content=state["subtasks"][2]),
    ]
    response = LLM.invoke(messages)
    print("✅ [子任务 3] 完成")
    return {"result_3": str(response.content)}


def synthesize_node(state: ParallelizationState) -> dict:
    """汇总节点：整合三个子任务结果，生成最终答案。

    Args:
        state: 当前图状态，包含 task 和三个子任务结果。

    Returns:
        字典 {"final": ...}，最终汇总答案文本。
    """
    combined = (
        f"子任务 1 结果：{state['result_1']}\n\n"
        f"子任务 2 结果：{state['result_2']}\n\n"
        f"子任务 3 结果：{state['result_3']}"
    )
    messages = [
        SystemMessage(content="你是内容整合专家，请整合多个结果形成连贯的最终回答。"),
        HumanMessage(
            content=f"主任务：{state['task']}\n\n各子任务结果：\n{combined}\n\n请输出最终汇总答案。"
        ),
    ]
    response = LLM.invoke(messages)
    return {"final": str(response.content)}


def build_parallelization_graph():
    """构建并编译 Parallelization 图：split → 三个并行 worker → synthesize。

    Returns:
        编译后的 LangGraph 图对象。
    """
    builder = StateGraph(ParallelizationState)

    builder.add_node("split", split_node)
    builder.add_node("worker_1", worker_1_node)
    builder.add_node("worker_2", worker_2_node)
    builder.add_node("worker_3", worker_3_node)
    builder.add_node("synthesize", synthesize_node)

    builder.set_entry_point("split")
    builder.add_edge("split", "worker_1")
    builder.add_edge("split", "worker_2")
    builder.add_edge("split", "worker_3")
    builder.add_edge("worker_1", "synthesize")
    builder.add_edge("worker_2", "synthesize")
    builder.add_edge("worker_3", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


def run(task: str) -> str:
    """运行完整的 Parallelization 流程。

    Args:
        task: 主任务描述。

    Returns:
        最终汇总答案文本。
    """
    graph = build_parallelization_graph()
    result = graph.invoke({"task": task})
    return result["final"]


def main():
    """演示 Parallelization LangGraph 实现。"""
    task = "对比 Python、Java、Go 三门语言各自的优缺点"
    answer = run(task)
    print("=" * 50)
    print(f"最终答案:\n{answer}")


if __name__ == "__main__":
    main()
