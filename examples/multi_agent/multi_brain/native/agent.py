"""
Multi-Brain Agent - 原生实现
=============================

核心思想：让多个不同角色/视角的 Agent 同时对同一问题给出答案，
然后由一个"汇总者"综合各方观点，形成更全面、更可靠的最终答案。

与 Debate 的区别：Debate 是对抗式（一方说 A，另一方说 B），
Multi-Brain 是协作式（各方从不同角度思考，然后互补）。

类似"群体智慧"或"多专家会诊"。
"""

from typing import Optional

from utils.llm_client import LLMClient


# ==============================================================================
# Multi-Brain 实现
# ==============================================================================

class BrainAgent:
    """单个 Brain Agent - 从特定视角思考问题"""

    def __init__(self, perspective: str, description: str, llm: Optional[LLMClient] = None):
        """
        初始化 Brain

        Args:
            perspective: 视角名称（如 "经济学家"）
            description: 该视角的思考特点
            llm: LLM 客户端
        """
        self.perspective = perspective
        self.description = description
        self.llm = llm or LLMClient()

    def think(self, question: str) -> str:
        """
        从特定视角给出思考

        Args:
            question: 问题

        Returns:
            该视角下的答案
        """
        system_prompt = f"""请以 {self.perspective} 的身份思考问题。
你的思考特点: {self.description}

请直接给出你的专业分析和建议，200 字以内。"""

        messages = [{"role": "user", "content": question}]
        print(f"  🧠 [{self.perspective}] 思考中...")
        answer = self.llm.chat(messages, system_prompt=system_prompt)
        print(f"  ✅ [{self.perspective}] 完成思考")
        return answer


class MultiBrainAgent:
    """Multi-Brain 聚合器 - 协调多个 Brain 并汇总"""

    def __init__(self, llm: Optional[LLMClient] = None):
        """
        初始化 Multi-Brain

        Args:
            llm: LLM 客户端（用于汇总）
        """
        self.llm = llm or LLMClient()
        # 定义多个不同视角的 Brain
        self.brains = [
            BrainAgent("技术专家", "关注技术可行性、实现路径、性能瓶颈", self.llm),
            BrainAgent("产品经理", "关注用户需求、市场价值、竞争态势", self.llm),
            BrainAgent("财务顾问", "关注成本预算、投资回报、财务风险", self.llm),
            BrainAgent("风险评估员", "关注安全风险、合规问题、潜在隐患", self.llm),
        ]

    def gather_perspectives(self, question: str) -> dict[str, str]:
        """
        收集所有 Brain 的视角（串行调用，实际可并行优化）

        Args:
            question: 问题

        Returns:
            视角名称 -> 答案的映射
        """
        print(f"\n{'='*50}")
        print(f"🧠 多 Brain 思考中...")
        print(f"{'='*50}")

        perspectives = {}
        for brain in self.brains:
            answer = brain.think(question)
            perspectives[brain.perspective] = answer

        return perspectives

    def synthesize(self, question: str, perspectives: dict[str, str]) -> str:
        """
        综合多个视角形成最终答案

        Args:
            question: 原始问题
            perspectives: 各视角的答案

        Returns:
            综合后的最终答案
        """
        # 格式化各视角的输入
        perspective_text = "\n\n".join(
            f"【{name}】\n{answer}" for name, answer in perspectives.items()
        )

        system_prompt = """你是一个综合分析专家。请整合多个专业视角的意见，
形成一个全面、平衡的最终回答。不要简单罗列，要找到不同视角间的联系和共识。"""

        messages = [{
            "role": "user",
            "content": f"问题: {question}\n\n以下是多个专家的观点:\n\n{perspective_text}\n\n请综合以上观点给出最终答案:"
        }]

        print(f"\n🔗 综合分析中...")
        final = self.llm.chat(messages, system_prompt=system_prompt)
        return final

    def run(self, question: str) -> str:
        """
        完整运行 Multi-Brain 流程

        Args:
            question: 问题

        Returns:
            综合后的答案
        """
        print(f"\n🎯 问题: {question}")

        # 1. 收集各视角
        perspectives = self.gather_perspectives(question)

        # 2. 综合汇总
        print(f"\n{'='*50}")
        final = self.synthesize(question, perspectives)
        return final


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 Multi-Brain 的使用"""
    agent = MultiBrainAgent()
    question = "我们公司要不要投入 100 万预算做一个 AI Agent 产品？"
    result = agent.run(question)
    print(f"\n{'='*50}")
    print(f"综合结论:\n{result}")


if __name__ == "__main__":
    main()
