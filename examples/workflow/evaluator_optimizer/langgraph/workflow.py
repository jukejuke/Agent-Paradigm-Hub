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
    # ============ 一、创建状态图 ============
    # StateGraph 是 LangGraph 的核心：一个"状态机 + 节点 + 连线"的容器。
    # 泛型参数 EvaluatorOptimizerState（就是上面定义的 TypedDict）规定了整张图
    # 中所有节点共享的"状态"数据结构，每个节点读写的就是这个 dict。
    graph = StateGraph(EvaluatorOptimizerState)

    # ============ 二、注册节点（Node） ============
    # add_node(名字, 函数)：把一个普通 Python 函数挂到图上。
    # 函数的入参是当前状态 dict，返回值是"要更新到状态里的那部分字段"。
    # 名字是字符串，可以随便起，但最好与函数功能对应，方便理解和调试。
    # 注意：这里只是"登记"节点，此刻还不会执行任何代码。
    graph.add_node("generate", generate_node)   # 生成节点：LLM 根据需求写初稿
    graph.add_node("evaluate", evaluate_node)   # 评估节点：LLM 给初稿打分并给反馈
    graph.add_node("optimize", optimize_node)   # 优化节点：LLM 根据反馈改进内容

    # ============ 三、指定入口节点 ============
    # set_entry_point(名字)：告诉 LangGraph 从哪个节点开始执行。
    # 一张图必须有且仅有一个入口，就像程序要有 main 函数一样。
    graph.set_entry_point("generate")           # 整个流程从 generate 开始

    # ============ 四、连普通边（Edge） ============
    # add_edge(起点, 终点)：表示"起点执行完，无条件、自动走到终点"。
    # 这样连接后，generate 跑完必定进入 evaluate，形成固定顺序。
    graph.add_edge("generate", "evaluate")      # 生成完初稿 -> 立刻去评估

    # ============ 五、连条件边（Conditional Edge） ============
    # add_conditional_edges(起点, 判断函数, 映射字典)：
    #   1) 起点节点执行完后，会自动调用判断函数 should_continue(state)；
    #   2) 该函数返回一个"字符串关键字"（比如 "optimize" 或 END）；
    #   3) 映射字典用这个关键字去查表，得到真正要去往的下一个节点名。
    # 字典 {关键字: 目标节点} 的含义：
    #   - 返回 "optimize" -> 去 "optimize" 节点（继续改进）
    #   - 返回 END（langgraph 内置常量，代表"流程结束"）-> 终止整张图
    # 这样 evaluate 之后是"继续优化"还是"结束"，由运行时动态决定，
    # 从而形成循环：evaluate -> (合格?) -> optimize -> evaluate -> ...
    graph.add_conditional_edges(
        "evaluate",                                 # 判断的起点节点
        should_continue,                            # 判断函数，读 state 里的反馈
        {"optimize": "optimize", END: END},         # 关键字 -> 目标节点的映射表
    )

    # ============ 六、闭合循环 ============
    # 上一步的条件边只处理了 evaluate 的去向，还要把 optimize 接回 evaluate，
    # 这样优化完之后会重新回到评估节点，形成"评估-优化"的循环回路。
    graph.add_edge("optimize", "evaluate")      # 改进完内容 -> 回到评估再打分

    # ============ 七、编译生成可运行图 ============
    # compile() 会对整张图做校验（比如入口是否设置、节点是否都存在、有没有死路），
    # 校验通过后返回一个"编译好的图对象"。之后调用 graph.stream(...) 才能真正运行。
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
