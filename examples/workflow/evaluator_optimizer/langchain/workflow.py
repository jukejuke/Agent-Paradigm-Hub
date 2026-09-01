"""
Evaluator-Optimizer Workflow - LangChain 实现
===============================================

使用 LangChain Chain 组合实现 生成→评估→改进 循环。
"""

from typing import Optional, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==============================================================================
# LangChain 实现
# ==============================================================================

class LangChainEvaluatorOptimizer:
    """LangChain Evaluator-Optimizer"""

    def __init__(self, model_name: str = "gpt-4o-mini", max_iterations: int = 5, pass_score: int = 7):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.max_iterations = max_iterations
        self.pass_score = pass_score

        # 构建三条链
        self.gen_chain = self._build_generator()
        self.eval_chain = self._build_evaluator()
        self.opt_chain = self._build_optimizer()

    def _build_generator(self) -> Any:
        """构建生成链"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "快速生成初稿，不要追求完美。"),
            ("user", "{requirements}")
        ])
        return prompt | self.llm | StrOutputParser()

    def _build_evaluator(self) -> Any:
        """构建评估链"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "严格评估。输出 'Score: X/10' 和 'Pass: yes/no'。"),
            ("user", "需求: {req}\n\n内容:\n{content}")
        ])
        return prompt | self.llm | StrOutputParser()

    def _build_optimizer(self) -> Any:
        """构建优化链"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "根据反馈逐条改进内容。"),
            ("user", "需求: {req}\n\n当前:\n{content}\n\n反馈:\n{feedback}")
        ])
        return prompt | self.llm | StrOutputParser()

    def _parse_score(self, eval_text: str) -> int:
        """从评估文本中提取分数"""
        import re
        match = re.search(r"Score:\s*(\d+)", eval_text)
        return int(match.group(1)) if match else 5

    def _is_pass(self, eval_text: str) -> bool:
        """判断是否通过"""
        import re
        match = re.search(r"Pass:\s*(yes|no)", eval_text, re.IGNORECASE)
        if match:
            return match.group(1).lower() == "yes"
        return self._parse_score(eval_text) >= self.pass_score

    def run(self, requirements: str) -> str:
        """运行完整工作流"""
        print(f"\n🎯 需求: {requirements}\n{'='*50}")

        # 生成
        print("\n📝 生成初稿...")
        current = self.gen_chain.invoke({"requirements": requirements})
        print(f"初稿: {current[:80]}...")

        for i in range(1, self.max_iterations + 1):
            print(f"\n--- 第 {i} 轮 ---")

            # 评估
            eval_result = self.eval_chain.invoke({"req": requirements, "content": current})
            score = self._parse_score(eval_result)
            passed = self._is_pass(eval_result)
            print(f"📊 Score: {score} {'✅' if passed else '❌'}")

            if passed:
                print("🎉 达标，停止！")
                break

            # 优化
            current = self.opt_chain.invoke({
                "req": requirements,
                "content": current,
                "feedback": eval_result,
            })
            print(f"✨ 优化: {current[:80]}...")

        return current


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 LangChain Evaluator-Optimizer"""
    workflow = LangChainEvaluatorOptimizer(max_iterations=3)
    result = workflow.run("写一段 100 字左右的产品介绍，用于 AI Agent SDK 的首页 banner")
    print(f"\n{'='*50}")
    print(f"最终版本:\n{result}")


if __name__ == "__main__":
    main()
