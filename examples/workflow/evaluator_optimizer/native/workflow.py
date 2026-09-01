"""
Evaluator-Optimizer Workflow - 原生实现
=========================================

核心思想：生成器（Generator）产出内容，评估器（Evaluator）打分和给出改进建议，
然后优化器（Optimizer）根据反馈改进内容，循环迭代直到质量达标。

适合场景：代码生成、文案优化、设计稿评审等有明确评估标准的场景。
"""

import re
from typing import Optional

from utils.llm_client import LLMClient


# ==============================================================================
# Evaluator-Optimizer 实现
# ==============================================================================

class EvaluatorOptimizerWorkflow:
    """Evaluator-Optimizer 工作流 - 生成-评估-改进循环"""

    GENERATOR_PROMPT = """请根据用户需求生成一个初稿。
快速生成，不要追求完美。"""

    EVALUATOR_PROMPT = """你是一个严格的评估器。请从以下维度给内容打分（1-10）并给出改进建议：
1. 准确性 (Accuracy)
2. 完整性 (Completeness)
3. 清晰度 (Clarity)
4. 实用性 (Practicality)

请严格按照以下格式输出：
Score: X/10
Feedback: [具体的改进建议，分点列出]
Pass: yes/no (是否达到 7 分以上可通过)"""

    OPTIMIZER_PROMPT = """你是一个改进专家。请根据评估反馈改进内容。
必须逐条回应反馈中的每个问题。"""

    def __init__(self, llm: Optional[LLMClient] = None, max_iterations: int = 5, pass_score: int = 7):
        """
        初始化工作流

        Args:
            llm: LLM 客户端
            max_iterations: 最大迭代次数
            pass_score: 通过的最低分数
        """
        self.llm = llm or LLMClient()
        self.max_iterations = max_iterations
        self.pass_score = pass_score

    def generate(self, requirements: str) -> str:
        """生成初稿"""
        messages = [{"role": "user", "content": requirements}]
        return self.llm.chat(messages, system_prompt=self.GENERATOR_PROMPT)

    def evaluate(self, requirements: str, content: str) -> tuple[int, str, bool]:
        """
        评估内容质量

        Returns:
            (分数, 反馈, 是否通过)
        """
        messages = [{
            "role": "user",
            "content": f"需求: {requirements}\n\n当前内容:\n{content}\n\n请评估:"
        }]
        eval_result = self.llm.chat(messages, system_prompt=self.EVALUATOR_PROMPT)

        # 解析分数
        score_match = re.search(r"Score:\s*(\d+)", eval_result)
        score = int(score_match.group(1)) if score_match else 5

        # 解析是否通过
        pass_match = re.search(r"Pass:\s*(yes|no)", eval_result, re.IGNORECASE)
        passed = (pass_match and pass_match.group(1).lower() == "yes") or score >= self.pass_score

        return score, eval_result, passed

    def optimize(self, requirements: str, content: str, feedback: str) -> str:
        """根据反馈改进内容"""
        messages = [{
            "role": "user",
            "content": f"需求: {requirements}\n\n当前内容:\n{content}\n\n评估反馈:\n{feedback}\n\n请改进:"
        }]
        return self.llm.chat(messages, system_prompt=self.OPTIMIZER_PROMPT)

    def run(self, requirements: str) -> str:
        """
        运行完整工作流

        Args:
            requirements: 需求描述

        Returns:
            最终内容
        """
        print(f"\n{'='*50}")
        print(f"🎯 需求: {requirements}")
        print(f"{'='*50}")

        # 1. 生成初稿
        print("\n📝 阶段 1: 生成初稿")
        current = self.generate(requirements)
        print(f"初稿:\n{current[:100]}...")

        for i in range(1, self.max_iterations + 1):
            print(f"\n--- 第 {i} 轮评估/优化 ---")

            # 2. 评估
            score, feedback, passed = self.evaluate(requirements, current)
            print(f"📊 评分: {score}/10 {'✅ 通过' if passed else '❌ 未通过'}")
            print(f"反馈:\n{feedback[:150]}...")

            if passed:
                print("🎉 质量达标，停止迭代！")
                break

            # 3. 优化
            print("✨ 优化内容...")
            current = self.optimize(requirements, current, feedback)

        return current


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 Evaluator-Optimizer 的使用"""
    workflow = EvaluatorOptimizerWorkflow(max_iterations=3)
    requirements = "写一段 100 字左右的产品介绍，用于 AI Agent SDK 的首页 banner"
    result = workflow.run(requirements)
    print(f"\n{'='*50}")
    print(f"最终版本:\n{result}")


if __name__ == "__main__":
    main()
