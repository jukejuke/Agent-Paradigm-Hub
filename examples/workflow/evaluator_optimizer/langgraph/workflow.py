"""
Evaluator-Optimizer - LangGraph 实现
====================================

核心思想：生成器先生成内容初稿，评估器以「严格 QA」角色按准确性/完整性/清晰度/实用性
四个维度打分并给出反馈，优化器根据反馈改进内容；随后再次评估，循环迭代直到达标或
达到最大迭代次数。流程：generate -> evaluate -> optimize -> evaluate ... 循环。
"""

import os
import re

from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from pathlib import Path
from dotenv import load_dotenv

# 定位项目根目录并加载 .env（支持从任意目录直接运行本文件）
load_dotenv(Path(__file__).resolve().parents[4] / ".env")


class EvaluatorOptimizerState(TypedDict):
    """Evaluator-Optimizer 图状态"""
    requirements: str   # 用户需求
    current: str        # 当前内容（初稿或改进稿）
    feedback: str       # 评估反馈（含 Score 与 Pass）
    pass_score: int     # 通过的最低分数
    iteration: int      # 当前迭代轮次


# 统一模型：默认 gpt-4o-mini，可用环境变量 OPENAI_MODEL 覆盖
_LLM = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0.7)

_GENERATE_SYSTEM = "你是一个内容生成助手。请根据用户需求快速生成初稿，无需追求完美。"

_EVALUATE_SYSTEM = (
    "你是一个严格的 QA 评估器。请从以下四个维度给内容打分（1-10 分）：\n"
    "1. 准确性 (Accuracy)\n"
    "2. 完整性 (Completeness)\n"
    "3. 清晰度 (Clarity)\n"
    "4. 实用性 (Practicality)\n\n"
    "请严格按照以下格式输出（用中文）：\n"
    "Score: X/10\n"
    "Feedback: [具体的改进建议，分点列出]\n"
    "Pass: yes/no"
)

_OPTIMIZE_SYSTEM = (
    "你是一个内容优化专家。请根据评估反馈逐条改进当前内容，"
    "确保反馈中的每个问题都得到解决，并输出改进后的完整内容。"
)


def generate_node(state: EvaluatorOptimizerState) -> dict:
    """
    生成节点：LLM 根据需求生成内容初稿

    Args:
        state: 当前图状态

    Returns:
        包含初稿 current 与初始迭代次数 iteration=1 的状态更新
    """
    messages = [
        SystemMessage(content=_GENERATE_SYSTEM),
        HumanMessage(content=state["requirements"]),
    ]
    draft = ""
    for chunk in _LLM.stream(messages):
        draft += chunk.content
    return {"current": draft, "iteration": 1}


def evaluate_node(state: EvaluatorOptimizerState) -> dict:
    """
    评估节点：LLM 以「严格 QA」角色对当前内容打分并给出反馈

    Args:
        state: 当前图状态

    Returns:
        包含评估文本 feedback 的状态更新
    """
    messages = [
        SystemMessage(content=_EVALUATE_SYSTEM),
        HumanMessage(content=f"需求: {state['requirements']}\n\n当前内容:\n{state['current']}\n\n请评估。"),
    ]
    feedback = ""
    for chunk in _LLM.stream(messages):
        feedback += chunk.content
    return {"feedback": feedback}


def optimize_node(state: EvaluatorOptimizerState) -> dict:
    """
    优化节点：LLM 根据评估反馈改进当前内容，并递增迭代次数

    Args:
        state: 当前图状态

    Returns:
        包含改进后 current 与 iteration+1 的状态更新
    """
    messages = [
        SystemMessage(content=_OPTIMIZE_SYSTEM),
        HumanMessage(
            content=(
                f"需求: {state['requirements']}\n\n"
                f"当前内容:\n{state['current']}\n\n"
                f"评估反馈:\n{state['feedback']}\n\n请改进。"
            )
        ),
    ]
    improved = ""
    for chunk in _LLM.stream(messages):
        improved += chunk.content
    return {"current": improved, "iteration": state["iteration"] + 1}


def should_continue(state: EvaluatorOptimizerState):
    """
    判断是否继续优化循环

    Args:
        state: 当前图状态

    Returns:
        "optimize" 表示继续优化，END 表示结束循环
    """
    feedback = state["feedback"]
    score_match = re.search(r"Score:\s*(\d+)", feedback)
    score = int(score_match.group(1)) if score_match else 0
    pass_match = re.search(r"Pass:\s*(yes|no)", feedback, re.IGNORECASE)
    passed = bool(pass_match and pass_match.group(1).lower() == "yes")
    if passed or score >= state["pass_score"] or state["iteration"] >= 3:
        return END
    return "optimize"


def build_evaluator_optimizer_graph():
    """
    构建 Evaluator-Optimizer 图

    Returns:
        编译后的 LangGraph 图
    """
    graph = StateGraph(EvaluatorOptimizerState)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("optimize", optimize_node)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", should_continue, {"optimize": "optimize", END: END})
    graph.add_edge("optimize", "evaluate")
    return graph.compile()


def run(requirements: str) -> str:
    """
    运行 Evaluator-Optimizer 工作流

    Args:
        requirements: 需求描述

    Returns:
        经过评估与优化后的最终内容
    """
    graph = build_evaluator_optimizer_graph()
    print(f"\n{'=' * 50}")
    print(f"需求: {requirements}")
    print(f"{'=' * 50}\n")

    final_state = None
    # updates 模式打印每个节点的状态更新，values 模式用于取最终状态
    for mode, payload in graph.stream(
        {"requirements": requirements, "pass_score": 7},
        stream_mode=["updates", "values"],
    ):
        if mode == "updates":
            for node_name, update in payload.items():
                print(f"\n[{node_name}] {update}")
        else:  # mode == "values"
            final_state = payload

    print("\n")
    return final_state["current"]


def main():
    """演示 LangGraph Evaluator-Optimizer 的使用"""
    requirements = "写一句奶茶店开业促销 slogan"
    result = run(requirements)
    print(f"{'=' * 50}")
    print(f"最终版本:\n{result}")


if __name__ == "__main__":
    main()
