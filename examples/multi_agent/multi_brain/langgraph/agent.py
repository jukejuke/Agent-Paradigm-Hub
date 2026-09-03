"""
Multi-Brain - LangGraph 实现
=============================

核心思想：让多个不同角色（技术专家 / 产品经理 / 财务顾问 / 风险评估）
对同一问题进行并行思考，再由一个"综合者"汇总各方观点，形成更全面、更平衡的最终决策建议。

使用 LangGraph 的 StateGraph 建模：START 分叉到 4 个并行角色节点，
各节点把观点写入独立状态键，最后汇聚到 synthesize 节点综合输出。
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
# 角色定义
# ==============================================================================

TECH_PROMPT = "你是资深技术专家，关注技术可行性、实现路径、性能瓶颈与架构选型。请以专业身份分析问题，给出清晰可落地的技术判断，控制在 200 字以内。"
PRODUCT_PROMPT = "你是资深产品经理，关注用户需求、市场价值、竞争态势与产品定位。请以专业身份分析问题，给出面向市场的产品判断，控制在 200 字以内。"
FINANCE_PROMPT = "你是资深财务顾问，关注成本预算、投资回报、现金流与财务风险。请以专业身份分析问题，给出可量化的财务判断，控制在 200 字以内。"
RISK_PROMPT = "你是资深风险评估专家，关注安全风险、合规问题、潜在隐患与不确定性。请以专业身份分析问题，给出审慎的风险判断，控制在 200 字以内。"


# ==============================================================================
# 状态定义
# ==============================================================================

class MultiBrainState(TypedDict):
    """Multi-Brain 状态"""
    question: str               # 用户问题
    perspectives: list[str]     # 各方观点（保留字段）
    final: str                  # 综合后的最终建议
    persp_tech: Optional[str]   # 技术专家观点
    persp_product: Optional[str]  # 产品经理观点
    persp_finance: Optional[str]  # 财务顾问观点
    persp_risk: Optional[str]   # 风险评估观点


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


def tech_node(state: MultiBrainState) -> dict:
    """技术专家节点：以技术专家身份回答 question。

    Args:
        state: 当前 Multi-Brain 状态

    Returns:
        写入 persp_tech 键的字典
    """
    llm = _get_llm()
    messages = [SystemMessage(content=TECH_PROMPT), HumanMessage(content=state["question"])]
    response = llm.invoke(messages)
    return {"persp_tech": response.content}


def product_node(state: MultiBrainState) -> dict:
    """产品经理节点：以产品经理身份回答 question。

    Args:
        state: 当前 Multi-Brain 状态

    Returns:
        写入 persp_product 键的字典
    """
    llm = _get_llm()
    messages = [SystemMessage(content=PRODUCT_PROMPT), HumanMessage(content=state["question"])]
    response = llm.invoke(messages)
    return {"persp_product": response.content}


def finance_node(state: MultiBrainState) -> dict:
    """财务顾问节点：以财务顾问身份回答 question。

    Args:
        state: 当前 Multi-Brain 状态

    Returns:
        写入 persp_finance 键的字典
    """
    llm = _get_llm()
    messages = [SystemMessage(content=FINANCE_PROMPT), HumanMessage(content=state["question"])]
    response = llm.invoke(messages)
    return {"persp_finance": response.content}


def risk_node(state: MultiBrainState) -> dict:
    """风险评估节点：以风险评估身份回答 question。

    Args:
        state: 当前 Multi-Brain 状态

    Returns:
        写入 persp_risk 键的字典
    """
    llm = _get_llm()
    messages = [SystemMessage(content=RISK_PROMPT), HumanMessage(content=state["question"])]
    response = llm.invoke(messages)
    return {"persp_risk": response.content}


def synthesize_node(state: MultiBrainState) -> dict:
    """综合者节点：汇总 4 个角色的观点并输出最终决策建议。

    Args:
        state: 当前 Multi-Brain 状态

    Returns:
        包含 final 最终建议的字典
    """
    llm = _get_llm()
    perspectives_text = (
        f"【技术专家】\n{state['persp_tech']}\n\n"
        f"【产品经理】\n{state['persp_product']}\n\n"
        f"【财务顾问】\n{state['persp_finance']}\n\n"
        f"【风险评估】\n{state['persp_risk']}"
    )
    messages = [
        SystemMessage(content="你是综合分析专家，请整合多个专业角色的观点，形成一个全面、平衡的最终决策建议。"),
        HumanMessage(content=f"问题: {state['question']}\n\n各角色观点:\n{perspectives_text}\n\n请综合给出最终决策建议:"),
    ]
    response = llm.invoke(messages)
    return {"final": response.content}


# ==============================================================================
# 图构建与入口
# ==============================================================================

def build_multi_brain_graph():
    """
    构建 Multi-Brain 状态图

    Returns:
        编译后的 LangGraph 图
    """
    graph = StateGraph(MultiBrainState)
    graph.add_node("tech", tech_node)
    graph.add_node("product", product_node)
    graph.add_node("finance", finance_node)
    graph.add_node("risk", risk_node)
    graph.add_node("synthesize", synthesize_node)
    # START 分叉到 4 个角色节点，实现并行思考
    graph.add_edge(START, "tech")
    graph.add_edge(START, "product")
    graph.add_edge(START, "finance")
    graph.add_edge(START, "risk")
    # 4 个节点汇聚到综合者节点
    graph.add_edge("tech", "synthesize")
    graph.add_edge("product", "synthesize")
    graph.add_edge("finance", "synthesize")
    graph.add_edge("risk", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def run(question: str) -> str:
    """
    运行 Multi-Brain 并返回最终答案

    Args:
        question: 用户问题

    Returns:
        综合后的最终决策建议文本
    """
    graph = build_multi_brain_graph()
    result = graph.invoke({"question": question})
    return result["final"]


def main():
    """演示入口：以固定问题运行 Multi-Brain 并打印最终建议"""
    question = "是否值得投入 100 万做 AI Agent 产品？"
    final = run(question)
    print(f"问题: {question}")
    print(f"最终建议:\n{final}")

if __name__ == "__main__":
    main()
