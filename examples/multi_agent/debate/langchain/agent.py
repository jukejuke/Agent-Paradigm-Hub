"""
Agent Debate - LangChain 实现
==============================

使用 LangChain 实现多 Agent 辩论，用 Runnable 链组装辩论流程。
"""

from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==============================================================================
# Debate 实现
# ==============================================================================

class LangChainDebater:
    """LangChain 辩论参与者"""

    def __init__(self, name: str, position: str, llm: ChatOpenAI):
        self.name = name
        self.position = position
        self.llm = llm

    def speak(self, topic: str, history: str, round_num: int) -> str:
        """发表观点"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是 {name}，立场: {position}。第 {round} 轮辩论。"
                       "请围绕主题有力阐述观点，可反驳对方。150字以内。"),
            ("user", "主题: {topic}\n\n前序辩论:\n{history}\n\n请发言:")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "name": self.name, "position": self.position,
            "round": round_num, "topic": topic, "history": history,
        })


class LangChainDebateOrchestrator:
    """LangChain 辩论编排者"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.debaters = [
            LangChainDebater("正方-小张", "AI Agent 将取代大部分传统工作", self.llm),
            LangChainDebater("反方-小李", "AI Agent 主要是辅助工具", self.llm),
        ]

    def conduct(self, topic: str, rounds: int = 3) -> str:
        """执行辩论"""
        print(f"\n🎤 辩论主题: {topic}\n{'='*50}")
        history = ""

        for r in range(1, rounds + 1):
            print(f"\n--- 第 {r} 轮 ---")
            round_content = f"[第{r}轮]\n"
            for d in self.debaters:
                statement = d.speak(topic, history, r)
                print(f"  {d.name}: {statement}")
                round_content += f"{d.name}: {statement}\n"
            history += round_content

        # 裁判
        print(f"\n👨‍⚖️ 裁判总结\n{'='*50}")
        judge_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是公正的辩论裁判。总结双方论点，评价强弱，给出综合判断。"),
            ("user", "主题: {topic}\n\n辩论记录:\n{history}")
        ])
        verdict = (judge_prompt | self.llm | StrOutputParser()).invoke({"topic": topic, "history": history})
        return verdict


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 LangChain Debate"""
    orchestrator = LangChainDebateOrchestrator()
    result = orchestrator.conduct("AI Agent 是否会在未来 10 年内大规模取代人类工作？", rounds=3)
    print(f"\n📝 裁判结论:\n{result}")


if __name__ == "__main__":
    main()
