"""
Orchestrator-Worker - LangGraph 实现
====================================

核心思想：总控编排者（Orchestrator）将复杂任务拆解为多个子任务，
多个专业 Worker（搜索研究员、数据分析师、报告撰写者）并行执行各自职责，
最后由汇总节点（Aggregator）整合所有结果，生成最终报告。

使用 LangGraph 的 StateGraph 将流程建模为状态图：
orchestrator -> 并行三 worker -> aggregator。
"""

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import os, json


# ==============================================================================
# Worker 角色定义
# ==============================================================================

# 三个专业 Worker 角色常量，value 为对应的中文 system prompt
WORKERS = {
    "researcher": "你是搜索研究员，负责围绕任务搜索、收集和整理相关信息，输出结构化的事实与资料要点。",
    "analyst": "你是数据分析师，负责对收集到的信息进行深入分析，提炼关键趋势、洞察与结论。",
    "writer": "你是报告撰写者，负责将资料与分析结果整合为条理清晰、语言流畅的中文报告。",
}


# ==============================================================================
# 状态定义
# ==============================================================================

class OrchestratorWorkerState(TypedDict):
    """Orchestrator-Worker 状态"""
    task: str             # 用户任务
    steps: list[str]      # 编排者拆解出的子任务
    researcher_result: str  # 搜索研究员结果
    analyst_result: str     # 数据分析师结果
    writer_result: str      # 报告撰写者结果
    final: str              # 最终汇总报告


# ==============================================================================
# 工具函数
# ==============================================================================

def _get_llm() -> ChatOpenAI:
    """
    获取统一的 LLM 实例

    Returns:
        ChatOpenAI 实例
    """
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.7)


# ==============================================================================
# 节点实现
# ==============================================================================

def orchestrator_node(state: OrchestratorWorkerState) -> dict:
    """
    编排者节点：将任务拆解为 JSON 步骤数组

    Args:
        state: 当前状态

    Returns:
        包含 steps 的更新字典
    """
    llm = _get_llm()
    messages = [
        SystemMessage(
            content=(
                "你是任务编排专家。请将用户的任务拆解为若干子任务步骤，"
                "以 JSON 字符串数组格式输出，例如：[\"步骤1\", \"步骤2\", \"步骤3\"]。"
                "只输出 JSON 数组，不要输出其他内容。"
            )
        ),
        HumanMessage(content=f"请拆解以下任务:\n{state['task']}"),
    ]
    response = llm.invoke(messages)
    content = response.content

    try:
        start = content.find("[")
        end = content.rfind("]") + 1
        steps = json.loads(content[start:end])
        if not isinstance(steps, list):
            raise ValueError("解析结果不是列表")
    except (json.JSONDecodeError, ValueError):
        steps = [
            f"搜索并收集关于「{state['task']}」的相关信息",
            "分析收集到的信息，提炼关键趋势与洞察",
            "整合分析结果，撰写最终报告",
        ]
    return {"steps": steps}


def researcher_node(state: OrchestratorWorkerState) -> dict:
    """
    搜索研究员节点：以研究员视角执行信息收集

    Args:
        state: 当前状态

    Returns:
        更新后的 researcher_result
    """
    llm = _get_llm()
    steps = "\n".join(f"- {s}" for s in state["steps"])
    messages = [
        SystemMessage(content=WORKERS["researcher"]),
        HumanMessage(
            content=f"任务: {state['task']}\n\n拆解出的子任务:\n{steps}\n\n请给出信息收集结果:"
        ),
    ]
    response = llm.invoke(messages)
    return {"researcher_result": response.content}


def analyst_node(state: OrchestratorWorkerState) -> dict:
    """
    数据分析师节点：以分析师视角对信息进行深入分析

    Args:
        state: 当前状态

    Returns:
        更新后的 analyst_result
    """
    llm = _get_llm()
    steps = "\n".join(f"- {s}" for s in state["steps"])
    messages = [
        SystemMessage(content=WORKERS["analyst"]),
        HumanMessage(
            content=f"任务: {state['task']}\n\n拆解出的子任务:\n{steps}\n\n请给出分析结果:"
        ),
    ]
    response = llm.invoke(messages)
    return {"analyst_result": response.content}


def writer_node(state: OrchestratorWorkerState) -> dict:
    """
    报告撰写者节点：以撰写者视角整合内容

    Args:
        state: 当前状态

    Returns:
        更新后的 writer_result
    """
    llm = _get_llm()
    steps = "\n".join(f"- {s}" for s in state["steps"])
    messages = [
        SystemMessage(content=WORKERS["writer"]),
        HumanMessage(
            content=f"任务: {state['task']}\n\n拆解出的子任务:\n{steps}\n\n请给出撰写内容:"
        ),
    ]
    response = llm.invoke(messages)
    return {"writer_result": response.content}


def aggregator_node(state: OrchestratorWorkerState) -> dict:
    """
    汇总节点：读取任务与三个 Worker 结果，汇总为最终报告

    Args:
        state: 当前状态

    Returns:
        更新后的 final
    """
    llm = _get_llm()
    messages = [
        SystemMessage(
            content="你是内容整合专家。请将搜索研究员、数据分析师、报告撰写者的结果"
                    "整合为一份完整、通顺、条理清晰的中文最终报告。"
        ),
        HumanMessage(
            content=(
                f"任务: {state['task']}\n\n"
                f"搜索研究员结果:\n{state['researcher_result']}\n\n"
                f"数据分析师结果:\n{state['analyst_result']}\n\n"
                f"报告撰写者结果:\n{state['writer_result']}\n\n"
                "请输出最终报告:"
            )
        ),
    ]
    response = llm.invoke(messages)
    return {"final": response.content}


# ==============================================================================
# 图构建与入口
# ==============================================================================

def build_orchestrator_worker_graph():
    """
    构建 Orchestrator-Worker 状态图

    Returns:
        编译后的 LangGraph 图
    """
    graph = StateGraph(OrchestratorWorkerState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    graph.add_node("aggregator", aggregator_node)

    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "researcher")
    graph.add_edge("orchestrator", "analyst")
    graph.add_edge("orchestrator", "writer")
    graph.add_edge("researcher", "aggregator")
    graph.add_edge("analyst", "aggregator")
    graph.add_edge("writer", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


def run(task: str) -> str:
    """
    运行 Orchestrator-Worker 流程并返回最终报告

    Args:
        task: 用户任务

    Returns:
        最终报告文本
    """
    graph = build_orchestrator_worker_graph()
    result = graph.invoke({"task": task})
    return result["final"]


def main():
    """演示入口：以固定任务运行并打印最终报告"""
    task = "写一份关于 AI Agent 发展趋势的简短报告"
    final = run(task)
    print(f"任务: {task}")
    print(f"最终报告:\n{final}")


if __name__ == "__main__":
    main()
