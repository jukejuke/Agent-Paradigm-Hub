"""
Reflection - LangGraph 实现：编程输入提示词优化 Agent
=====================================================

核心思想：让 LLM 先把用户粗糙的开发需求打磨成一份「可直接交给编程智能体
（Trae / Claude Code / OpenCode 等）执行的提示词」初稿，再以「严厉批评者」
角色从编程提示词质量维度进行自我批评，最后根据批评意见改进提示词，
循环迭代直到满意或达到最大迭代次数。

流程：generate（生成提示词初稿）-> reflect（自我批评）-> refine（按批评改进）
      -> reflect -> refine -> ... 直到 should_continue 判定结束，
      输出可直接复制给编程智能体的最终提示词。
"""

import sys
from pathlib import Path
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from tools.prompt_optimizer.config import OptimizerConfig
from tools.prompt_optimizer.prompts import (
    GENERATE_SYSTEM,
    REFINE_SYSTEM,
    REFLECT_SYSTEM,
    STOP_KEYWORDS,
)

# 定位仓库根目录并加载 .env（支持从任意目录直接运行本工具）
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class OptimizerState(TypedDict):
    """提示词优化图的节点状态"""
    raw_prompt: str   # 用户原始需求
    draft: str        # 当前提示词（初稿或改进稿）
    critique: str     # 批评意见
    iteration: int    # 当前迭代轮次


class PromptOptimizerAgent:
    """基于 Reflection 模式的编程提示词优化 Agent（LangGraph 实现）
    作者: yaosh
    日期: 2026-09-04
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        初始化 Agent：创建 LLM 与编译后的 Reflection 图

        Args:
            config: 优化配置，缺省时从环境变量读取
        """
        config = config or OptimizerConfig()
        self.max_iterations = config.max_iterations
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
        生成节点：LLM 根据原始需求生成提示词初稿

        Args:
            state: 当前图状态

        Returns:
            包含初稿 draft 与初始迭代次数 iteration=1 的状态更新
        """
        messages = [
            SystemMessage(content=GENERATE_SYSTEM),
            HumanMessage(content=f"原始需求：\n{state['raw_prompt']}"),
        ]
        draft = "".join(chunk.content or "" for chunk in self.llm.stream(messages))
        return {"draft": draft, "iteration": 1}

    def _reflect_node(self, state: OptimizerState) -> dict:
        """
        反思节点：LLM 以「严厉批评者」角色审阅 draft，输出中文批评意见

        Args:
            state: 当前图状态

        Returns:
            包含批评意见 critique 的状态更新
        """
        messages = [
            SystemMessage(content=REFLECT_SYSTEM),
            HumanMessage(content=f"待评审的提示词：\n{state['draft']}"),
        ]
        critique = "".join(chunk.content or "" for chunk in self.llm.stream(messages))
        return {"critique": critique}

    def _refine_node(self, state: OptimizerState) -> dict:
        """
        改进节点：LLM 根据 critique 改进 draft，并递增迭代次数

        Args:
            state: 当前图状态

        Returns:
            包含改进后 draft 与 iteration+1 的状态更新
        """
        messages = [
            SystemMessage(content=REFINE_SYSTEM),
            HumanMessage(
                content=(
                    f"上一版提示词：\n{state['draft']}\n\n"
                    f"评审意见：\n{state['critique']}\n\n请改进。"
                )
            ),
        ]
        refined = "".join(chunk.content or "" for chunk in self.llm.stream(messages))
        return {"draft": refined, "iteration": state["iteration"] + 1}

    # ------------------------------------------------------------------ 路由
    def _should_continue(self, state: OptimizerState):
        """
        判断是否继续反思-改进循环

        Args:
            state: 当前图状态

        Returns:
            "refine" 表示继续改进，END 表示结束循环
        """
        if any(keyword in state["critique"] for keyword in STOP_KEYWORDS):
            return END
        if state["iteration"] >= self.max_iterations:
            return END
        return "refine"

    # ------------------------------------------------------------------ 图
    def _build_graph(self):
        """
        构建 Reflection 图：generate -> reflect -> (refine -> reflect)* 直到结束

        Returns:
            编译后的 LangGraph 图
        """
        graph = StateGraph(OptimizerState)
        graph.add_node("generate", self._generate_node)
        graph.add_node("reflect", self._reflect_node)
        graph.add_node("refine", self._refine_node)

        graph.set_entry_point("generate")
        graph.add_edge("generate", "reflect")
        graph.add_conditional_edges(
            "reflect",
            self._should_continue,
            {"refine": "refine", END: END},
        )
        graph.add_edge("refine", "reflect")
        return graph.compile()

    # ------------------------------------------------------------------ 运行
    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """
        去掉提示词外层可能包裹的 Markdown 代码块围栏，便于直接复制

        Args:
            text: 模型原始输出

        Returns:
            清理围栏后的纯提示词文本
        """
        stripped = text.strip()
        if stripped.startswith("```") and "\n" in stripped:
            lines = stripped.split("\n")
            lines = lines[1:]  # 去掉开头的 ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # 去掉结尾的 ```
            return "\n".join(lines).strip()
        return stripped

    def optimize(self, raw_prompt: str, verbose: bool = True) -> str:
        """
        运行 Reflection 提示词优化流程，返回优化后的最终提示词

        Args:
            raw_prompt: 用户原始需求（粗糙的提示词）
            verbose: 是否打印迭代过程，默认 True

        Returns:
            可直接交给编程智能体的优化后提示词
        """
        print(f"\n{'=' * 60}")
        preview = raw_prompt[:80] + ("..." if len(raw_prompt) > 80 else "")
        print(f"原始需求：{preview}")
        print(f"{'=' * 60}\n")

        final_state = None
        # updates 模式打印每个节点的状态更新，values 模式用于取最终状态
        for mode, payload in self.graph.stream(
            {"raw_prompt": raw_prompt}, stream_mode=["updates", "values"]
        ):
            if mode == "updates":
                for node_name, update in payload.items():
                    if not verbose:
                        continue
                    print(f"\n[{node_name}]")
                    if "iteration" in update:
                        print(f"  迭代轮次: {update['iteration']}")
                    if node_name == "reflect" and "critique" in update:
                        # 仅展示批评意见首行，避免刷屏
                        first_line = update["critique"].strip().split("\n")[0]
                        print(f"  批评(首行): {first_line}")
            else:  # mode == "values"
                final_state = payload

        return self._strip_code_fence(final_state["draft"])


def main():
    """
    命令行入口：python -m tools.prompt_optimizer [原始需求]

    未提供参数时，交互式读取用户的开发需求；仍为空则使用演示需求。
    """
    raw_prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not raw_prompt:
        raw_prompt = input("请输入你的编程开发需求（可直接粘贴粗糙的提示词，回车结束）:\n").strip()
    if not raw_prompt:
        raw_prompt = "请帮我写一个 Python 脚本，从 CSV 读取数据并生成统计图表。"
        print(f"（未输入需求，使用演示需求：{raw_prompt}）")

    agent = PromptOptimizerAgent()
    optimized = agent.optimize(raw_prompt)

    print("\n" + "=" * 60)
    print("最终优化后的提示词（可直接复制给 Trae / Claude Code / OpenCode 等编程智能体）:")
    print("=" * 60)
    print(optimized)
    print("=" * 60)


if __name__ == "__main__":
    main()
