"""
Reflection Agent - 原生实现
============================

核心思想：让 LLM 生成初始答案，然后进行自我反思和批评，
再根据反思结果改进答案，循环迭代直到满意。

与 ReAct 的区别：ReAct 是用工具来获取外部信息，
Reflection 是 LLM 自己审视自己的输出，进行内部质量提升。

参考: Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023)
"""

from typing import Optional

import os
import sys

# 将项目根目录加入 sys.path，使本脚本可在任意工作目录运行
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from utils.llm_client import LLMClient


# ==============================================================================
# Reflection Agent 实现
# ==============================================================================

class ReflectionAgent:
    """Reflection 范式智能体 - 自我反思改进答案"""

    GENERATION_SYSTEM_PROMPT = """你是一个善于解决问题的助手。请直接给出你的答案。
不需要考虑语言的完美性，先快速给出一个初稿回答。"""

    REFLECTION_SYSTEM_PROMPT = """你是一个严厉的批评者。请审视下面的回答，找出其中的:
1. 逻辑错误
2. 事实错误
3. 遗漏的关键点
4. 可以改进的地方

请用中文逐条列出你的批评意见，每条不超过一句话。
如果回答已经很好，请说"答案已经很好，无需改进"。"""

    REFINEMENT_SYSTEM_PROMPT = """你是一个改进专家。根据下面的批评意见，重新给出一个更好的答案。
必须逐条回应批评意见，确保每个问题都被解决。"""

    def __init__(self, llm: Optional[LLMClient] = None, max_iterations: int = 3):
        """
        初始化 Reflection Agent

        Args:
            llm: LLM 客户端实例
            max_iterations: 最大反思-改进轮次
        """
        self.llm = llm or LLMClient()
        self.max_iterations = max_iterations

    def _generate(self, question: str) -> str:
        """生成初始答案"""
        messages = [{"role": "user", "content": question}]
        return self.llm.chat(messages, system_prompt=self.GENERATION_SYSTEM_PROMPT)

    def _reflect(self, question: str, answer: str) -> str:
        """对答案进行反思批评"""
        messages = [{
            "role": "user",
            "content": f"问题: {question}\n\n当前回答:\n{answer}\n\n请进行反思批评。"
        }]
        return self.llm.chat(messages, system_prompt=self.REFLECTION_SYSTEM_PROMPT)

    def _refine(self, question: str, answer: str, critique: str) -> str:
        """根据批评改进答案"""
        messages = [{
            "role": "user",
            "content": f"问题: {question}\n\n上一轮回答:\n{answer}\n\n批评意见:\n{critique}\n\n请改进答案。"
        }]
        return self.llm.chat(messages, system_prompt=self.REFINEMENT_SYSTEM_PROMPT)

    def run(self, question: str) -> str:
        """
        运行 Reflection Agent

        Args:
            question: 用户的问题

        Returns:
            经过反思改进后的最终答案
        """
        print(f"🎯 问题: {question}\n")

        # 1. 生成初始答案
        current_answer = self._generate(question)
        print(f"📝 初始答案:\n{current_answer}\n")

        for i in range(1, self.max_iterations + 1):
            print(f"\n--- 第 {i} 轮反思 ---")

            # 2. 反思批评
            critique = self._reflect(question, current_answer)
            print(f"🔍 批评意见:\n{critique}\n")

            # 检查是否可以停止
            if "已经很好" in critique or "无需改进" in critique:
                print("✅ LLM 认为答案已经很好，停止反思。")
                break

            # 3. 根据批评改进
            current_answer = self._refine(question, current_answer, critique)
            print(f"✨ 改进后答案:\n{current_answer}\n")

        return current_answer


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 Reflection Agent 的使用"""
    agent = ReflectionAgent(max_iterations=3)
    question = "解释一下什么是区块链，以及它和传统数据库有什么区别。"
    answer = agent.run(question)
    print(f"\n{'='*50}")
    print(f"最终答案:\n{answer}")


if __name__ == "__main__":
    main()
