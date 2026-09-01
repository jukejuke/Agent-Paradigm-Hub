"""
Prompt Chaining Workflow - LangChain 实现
=========================================

使用 LangChain RunnableSequence 实现链式调用。
"""

from typing import Optional, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ==============================================================================
# LangChain 实现
# ==============================================================================

class LangChainPromptChaining:
    """LangChain 链式提示"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)

    def run(self, initial_input: str, steps: list[dict]) -> str:
        """
        执行链式处理

        Args:
            initial_input: 初始输入
            steps: 步骤列表，每个包含 name, system_prompt, user_template

        Returns:
            最终输出
        """
        print(f"\n🔗 Prompt Chaining ({len(steps)} 步)\n{'='*50}")
        current = initial_input

        for i, step in enumerate(steps):
            prompt = ChatPromptTemplate.from_messages([
                ("system", step["system_prompt"]),
                ("user", step["user_template"])
            ])
            chain = prompt | self.llm | StrOutputParser()

            print(f"\n--- Step {i+1}: {step['name']} ---")
            print(f"  输入: {current[:50]}...")
            current = chain.invoke({"input": current})
            print(f"  输出: {current[:50]}...")

        return current


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 LangChain Prompt Chaining"""
    workflow = LangChainPromptChaining()
    final = workflow.run(
        "我想写一篇关于 AI Agent 发展历程的博客文章",
        [
            {
                "name": "结构化大纲",
                "system_prompt": "文章策划。将想法扩展为 5 点大纲，每点不超过 20 字。",
                "user_template": "为以下想法制定大纲:\n{input}",
            },
            {
                "name": "内容润色",
                "system_prompt": "资深编辑。将大纲扩展成文章段落，每段 50-100 字。",
                "user_template": "将大纲扩展成初稿:\n{input}",
            },
            {
                "name": "生成推文",
                "system_prompt": "社交媒体专家。提炼成 280 字内的中文推文，开头有钩子。",
                "user_template": "将文章提炼成推文:\n{input}",
            },
        ]
    )
    print(f"\n{'='*50}")
    print(f"📤 最终推文:\n{final}")


if __name__ == "__main__":
    main()
