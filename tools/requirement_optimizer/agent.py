"""
需求工作量放大器 - 核心逻辑（Evaluator-Optimizer · LangGraph）
=============================================================

输入一段简短需求，输出一份「看起来工作量更大」的优化后需求，
且字数与原需求差别不大（±30%）。

实现方式：Evaluator-Optimizer 循环
  generate（生成放大初稿）-> evaluate（四维度打分）-> optimize（按反馈改进）
  -> evaluate ... 直到 Pass=yes / 分数达标 / 达到最大迭代次数。
作者: yaosh
日期: 2026-10-09
"""

import re
import sys
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from tools.requirement_optimizer.config import OptimizerConfig
from tools.requirement_optimizer.prompts import (
    EVALUATE_SYSTEM,
    GENERATE_SYSTEM,
    OPTIMIZE_SYSTEM,
)

# 定位仓库根目录并加载 .env（支持从任意目录直接运行本工具）
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# 字数容差上限（原需求的 130%），超过则硬性截断兜底
MAX_LENGTH_RATIO = 1.3


class OptimizerState(TypedDict):
    """需求放大图的节点状态"""

    raw_requirement: str  # 用户原始需求
    original_len: int  # 原需求字符数
    current: str  # 当前优化后需求（初稿或改进稿）
    feedback: str  # 评估反馈（含 Score / Feedback / Pass）
    score: int  # 评估分数（1-10）
    passed: bool  # 评估是否通过
    iteration: int  # 当前迭代轮次


def count_chars(text: str) -> int:
    """
    统计文本字符数（去除首尾空白后）

    Args:
        text: 待统计文本

    Returns:
        字符数
    """
    return len(text.strip())


def strip_code_fence(text: str) -> str:
    """
    去掉文本外层可能包裹的 Markdown 代码块围栏

    Args:
        text: 模型原始输出

    Returns:
        清理围栏后的纯文本
    """
    stripped = text.strip()
    if stripped.startswith("```") and "\n" in stripped:
        lines = stripped.split("\n")
        lines = lines[1:]  # 去掉开头的 ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # 去掉结尾的 ```
        return "\n".join(lines).strip()
    return stripped


def truncate_to_length(text: str, target: int) -> str:
    """
    将文本截断到目标字数以内，尽量在标点处断句以保持可读性

    Args:
        text: 待截断文本
        target: 目标字数上限

    Returns:
        截断后的文本
    """
    if count_chars(text) <= target:
        return text
    truncated = text[:target].strip()
    # 尽量在最后一个标点处断句，避免半句截断
    for punct in ("。", "！", "？", "；", ".", "!", "?", ";"):
        idx = truncated.rfind(punct)
        if idx >= target // 2:  # 标点至少在前半部分，避免截得过短
            return truncated[: idx + 1].strip()
    return truncated


def parse_evaluation(feedback: str) -> tuple[int, bool]:
    """
    从评估文本中解析分数与是否通过

    兼容「Score: X/10」与「Pass: yes/no」格式；解析失败时分数兜底为 0、
    不通过，保证循环仍可由最大迭代次数终止。

    Args:
        feedback: 评估节点输出的文本

    Returns:
        (分数, 是否通过)
    """
    score_match = re.search(r"Score:\s*(\d+)", feedback)
    score = int(score_match.group(1)) if score_match else 0
    pass_match = re.search(r"Pass:\s*(yes|no)", feedback, re.IGNORECASE)
    passed = bool(pass_match and pass_match.group(1).lower() == "yes")
    return score, passed


class RequirementOptimizerAgent:
    """基于 Evaluator-Optimizer 模式的需求工作量放大 Agent（LangGraph 实现）
    作者: yaosh
    日期: 2026-10-09
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        初始化 Agent：创建 LLM 与编译后的 Evaluator-Optimizer 图

        Args:
            config: 运行配置，缺省时从环境变量读取
        """
        config = config or OptimizerConfig()
        self.max_iterations = config.max_iterations
        self.pass_score = config.pass_score
        self.llm = ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
        )
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ 节点
    def _generate_node(self, state: OptimizerState) -> dict:
        """
        生成节点：LLM 以「需求分析师」身份产出工作量放大初稿

        Args:
            state: 当前图状态

        Returns:
            包含初稿 current 与初始迭代次数 iteration=1 的状态更新
        """
        messages = [
            SystemMessage(content=GENERATE_SYSTEM),
            HumanMessage(content=f"原始需求：\n{state['raw_requirement']}"),
        ]
        draft = "".join(chunk.content or "" for chunk in self.llm.stream(messages))
        return {"current": draft, "iteration": 1}

    def _evaluate_node(self, state: OptimizerState) -> dict:
        """
        评估节点：LLM 以「严格 QA」角色对 current 打分并给出反馈

        Args:
            state: 当前图状态

        Returns:
            包含评估文本 feedback、解析出的 score 与 passed 的状态更新
        """
        messages = [
            SystemMessage(content=EVALUATE_SYSTEM),
            HumanMessage(
                content=(
                    f"原需求（{state['original_len']} 字）：\n"
                    f"{state['raw_requirement']}\n\n"
                    f"当前优化后需求（{count_chars(state['current'])} 字）：\n"
                    f"{state['current']}\n\n"
                    f"请评估。"
                )
            ),
        ]
        feedback = "".join(chunk.content or "" for chunk in self.llm.stream(messages))
        score, passed = parse_evaluation(feedback)
        return {"feedback": feedback, "score": score, "passed": passed}

    def _optimize_node(self, state: OptimizerState) -> dict:
        """
        优化节点：LLM 根据评估反馈逐条改进 current，并递增迭代次数

        Args:
            state: 当前图状态

        Returns:
            包含改进后 current 与 iteration+1 的状态更新
        """
        messages = [
            SystemMessage(content=OPTIMIZE_SYSTEM),
            HumanMessage(
                content=(
                    f"原需求（{state['original_len']} 字）：\n"
                    f"{state['raw_requirement']}\n\n"
                    f"当前需求：\n{state['current']}\n\n"
                    f"评审意见：\n{state['feedback']}\n\n请改进。"
                )
            ),
        ]
        improved = "".join(chunk.content or "" for chunk in self.llm.stream(messages))
        return {"current": improved, "iteration": state["iteration"] + 1}

    # ------------------------------------------------------------------ 路由
    def _should_continue(self, state: OptimizerState):
        """
        判断是否继续优化循环

        Args:
            state: 当前图状态

        Returns:
            "optimize" 表示继续优化，END 表示结束循环
        """
        if state.get("passed"):
            return END
        if state.get("score", 0) >= self.pass_score:
            return END
        if state["iteration"] >= self.max_iterations:
            return END
        return "optimize"

    # ------------------------------------------------------------------ 图
    def _build_graph(self):
        """
        构建 Evaluator-Optimizer 图：generate -> evaluate -> (optimize -> evaluate)*

        Returns:
            编译后的 LangGraph 图
        """
        graph = StateGraph(OptimizerState)
        graph.add_node("generate", self._generate_node)
        graph.add_node("evaluate", self._evaluate_node)
        graph.add_node("optimize", self._optimize_node)

        graph.set_entry_point("generate")
        graph.add_edge("generate", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._should_continue,
            {"optimize": "optimize", END: END},
        )
        graph.add_edge("optimize", "evaluate")
        return graph.compile()

    # ------------------------------------------------------------------ 运行
    def optimize(self, raw_requirement: str, verbose: bool = True) -> str:
        """
        运行 Evaluator-Optimizer 需求放大流程，返回优化后的需求文本

        循环结束后对结果做代码围栏清理；若字数仍超过原需求的 130%，
        硬性截断到目标字数以内作为兜底。

        Args:
            raw_requirement: 用户原始需求
            verbose: 是否打印迭代过程，默认 True

        Returns:
            优化后的需求文本
        """
        original_len = count_chars(raw_requirement)
        if verbose:
            print(f"\n{'=' * 60}")
            preview = raw_requirement[:80] + ("..." if len(raw_requirement) > 80 else "")
            print(f"原始需求：{preview}")
            print(f"{'=' * 60}\n")

        final_state = None
        # updates 模式打印每个节点的状态更新，values 模式用于取最终状态
        for mode, payload in self.graph.stream(
            {"raw_requirement": raw_requirement, "original_len": original_len},
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                for node_name, update in payload.items():
                    if not verbose:
                        continue
                    print(f"\n[{node_name}]")
                    if "iteration" in update:
                        print(f"  迭代轮次: {update['iteration']}")
                    if node_name == "evaluate":
                        print(f"  分数: {update.get('score', '?')}/10  "
                              f"通过: {'是' if update.get('passed') else '否'}")
                        first_line = update.get("feedback", "").strip().split("\n")[0]
                        print(f"  反馈(首行): {first_line}")
            else:  # mode == "values"
                final_state = payload

        result = strip_code_fence(final_state["current"])
        # 硬性兜底：字数仍超限则截断
        if original_len > 0 and count_chars(result) > original_len * MAX_LENGTH_RATIO:
            result = truncate_to_length(result, int(original_len * MAX_LENGTH_RATIO))
        return result


def optimize(requirement: str) -> str:
    """
    便捷函数：使用默认配置放大需求的感知工作量

    Args:
        requirement: 用户原始需求

    Returns:
        优化后的需求文本
    """
    return RequirementOptimizerAgent().optimize(requirement)


def main():
    """
    命令行入口：python -m tools.requirement_optimizer [需求]

    未提供参数时，交互式读取用户需求；仍为空则使用演示需求。
    运行后打印原需求与优化后需求的字数对比。
    """
    requirement = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not requirement:
        requirement = input("请输入你的需求（回车结束）:\n").strip()
    if not requirement:
        requirement = "做一个用户登录功能。"
        print(f"（未输入需求，使用演示需求：{requirement}）")

    agent = RequirementOptimizerAgent()
    optimized = agent.optimize(requirement)

    original_len = count_chars(requirement)
    optimized_len = count_chars(optimized)
    ratio = optimized_len / original_len if original_len else 0

    print("\n" + "=" * 60)
    print("【原需求】")
    print(requirement)
    print(f"（字数：{original_len}）")
    print("-" * 60)
    print("【优化后需求】")
    print(optimized)
    print(f"（字数：{optimized_len}，原需求的 {ratio:.1%}）")
    print("=" * 60)


if __name__ == "__main__":
    main()
