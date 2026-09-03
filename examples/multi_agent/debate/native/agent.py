"""
Agent Debate - 原生实现
========================

核心思想：多个 Agent 持不同立场进行辩论，
最后由裁判（Judge）综合各方观点得出结论。

适合场景：有争议的问题讨论、方案评审、多角度决策。
"""

from typing import Optional

import os
import sys

# 将项目根目录加入 sys.path，使本脚本可在任意工作目录运行
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from utils.llm_client import LLMClient


# ==============================================================================
# Agent Debate 实现
# ==============================================================================

class DebateAgent:
    """辩论参与者 Agent"""

    def __init__(self, name: str, position: str, llm: Optional[LLMClient] = None):
        """
        初始化辩论 Agent

        Args:
            name: Agent 名称
            position: 立场描述（如"正方：应该引入AI"）
            llm: LLM 客户端
        """
        self.name = name
        self.position = position
        self.llm = llm or LLMClient()

    def speak(self, topic: str, context: str = "", round_num: int = 1) -> str:
        """
        发表辩论观点

        Args:
            topic: 辩论主题
            context: 之前轮次的辩论内容
            round_num: 当前轮次

        Returns:
            发言内容
        """
        system_prompt = f"""你是 {self.name}，你的立场是: {self.position}
当前是第 {round_num} 轮辩论。
请围绕主题有力地阐述你的观点，可以反驳对方的论点。
发言控制在 150 字以内。"""

        messages = [{
            "role": "user",
            "content": f"辩论主题: {topic}\n\n前序辩论:\n{context}\n\n请发表你的观点:"
        }]

        return self.llm.chat(messages, system_prompt=system_prompt)


class DebateOrchestrator:
    """辩论编排者 - 主持多 Agent 辩论"""

    def __init__(self, llm: Optional[LLMClient] = None):
        """
        初始化辩论编排者

        Args:
            llm: LLM 客户端（用于裁判角色）
        """
        self.llm = llm or LLMClient()
        self.debaters = [
            DebateAgent("正方-小张", "AI Agent 将取代大部分传统工作", self.llm),
            DebateAgent("反方-小李", "AI Agent 主要是辅助工具，不会大规模取代工作", self.llm),
        ]

    def conduct_debate(self, topic: str, rounds: int = 3) -> str:
        """
        执行多轮辩论

        Args:
            topic: 辩论主题
            rounds: 辩论轮数

        Returns:
            裁判的最终结论
        """
        print(f"\n{'='*50}")
        print(f"🎤 辩论主题: {topic}")
        print(f"参赛选手: {', '.join(d.name for d in self.debaters)}")
        print(f"{'='*50}")

        debate_history = ""

        for round_num in range(1, rounds + 1):
            print(f"\n--- 第 {round_num} 轮 ---")
            round_content = f"[第 {round_num} 轮]\n"

            for debater in self.debaters:
                statement = debater.speak(topic, debate_history, round_num)
                print(f"  {debater.name}: {statement}")
                round_content += f"{debater.name}: {statement}\n"

            debate_history += round_content

        # 裁判总结
        print(f"\n{'='*50}")
        print("👨‍⚖️ 裁判总结")
        print(f"{'='*50}")

        judge_prompt = f"""辩论主题: {topic}

完整辩论记录:
{debate_history}

请作为中立裁判:
1. 总结双方的核心论点
2. 评价双方论证的强弱
3. 给出你的综合判断（不选边，综合双方观点）"""

        messages = [{"role": "user", "content": judge_prompt}]
        verdict = self.llm.chat(messages, system_prompt="你是一个公正的辩论裁判。")
        return verdict


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 Agent Debate 的使用"""
    orchestrator = DebateOrchestrator()
    topic = "AI Agent 是否会在未来 10 年内大规模取代人类工作？"
    result = orchestrator.conduct_debate(topic, rounds=3)
    print(f"\n📝 裁判结论:\n{result}")


if __name__ == "__main__":
    main()
