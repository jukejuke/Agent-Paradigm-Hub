"""
Multi-Brain Agent - LangChain 实现
====================================

使用 LangChain 实现多视角思考和汇总。
"""

from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==============================================================================
# Multi-Brain 实现
# ==============================================================================

class LangChainBrain:
    """LangChain Brain - 单视角思考"""

    def __init__(self, perspective: str, description: str, llm: ChatOpenAI):
        self.perspective = perspective
        self.description = description
        self.llm = llm

    def think(self, question: str) -> str:
        """从特定视角思考"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是 {p}，思考特点: {d}。请以专业身份分析，200字以内。"),
            ("user", "{question}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"p": self.perspective, "d": self.description, "question": question})
        print(f"  ✅ [{self.perspective}] 完成")
        return result


class LangChainMultiBrain:
    """LangChain Multi-Brain 聚合器"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.brains = [
            LangChainBrain("技术专家", "关注技术可行性、实现路径、性能瓶颈", self.llm),
            LangChainBrain("产品经理", "关注用户需求、市场价值、竞争态势", self.llm),
            LangChainBrain("财务顾问", "关注成本预算、投资回报、财务风险", self.llm),
            LangChainBrain("风险评估员", "关注安全风险、合规问题、潜在隐患", self.llm),
        ]

    def run(self, question: str) -> str:
        """完整运行 Multi-Brain"""
        print(f"\n🧠 Multi-Brain 思考中...\n{'='*50}")

        # 收集各视角
        perspectives = {}
        for brain in self.brains:
            print(f"  🤔 [{brain.perspective}] 思考中...")
            perspectives[brain.perspective] = brain.think(question)

        # 汇总
        print(f"\n🔗 综合分析中...")
        perspective_text = "\n\n".join(f"【{n}】\n{a}" for n, a in perspectives.items())

        synth_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是综合分析专家，整合多视角形成全面平衡的回答。"),
            ("user", "问题: {q}\n\n专家观点:\n{p}\n\n请综合给出最终答案:")
        ])
        final = (synth_prompt | self.llm | StrOutputParser()).invoke({"q": question, "p": perspective_text})
        return final


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 LangChain Multi-Brain"""
    agent = LangChainMultiBrain()
    result = agent.run("我们公司要不要投入 100 万预算做一个 AI Agent 产品？")
    print(f"\n{'='*50}")
    print(f"综合结论:\n{result}")


if __name__ == "__main__":
    main()
