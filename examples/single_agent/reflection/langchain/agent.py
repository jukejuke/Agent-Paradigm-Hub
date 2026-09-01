"""
Reflection Agent - LangChain 实现
==================================

使用 LangChain 框架实现的 Reflection Agent。
通过 Chain 组合实现: 生成 -> 反思 -> 改进 的循环。
"""

from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==============================================================================
# Reflection 实现
# ==============================================================================

def create_reflection_chains(llm: ChatOpenAI):
    """
    创建反思-改进链

    Args:
        llm: ChatOpenAI 实例

    Returns:
        (generate_chain, reflect_chain, refine_chain) 三个链式调用
    """
    # 生成链
    generate_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个善于解决问题的助手。快速给出一个答案初稿。"),
        ("user", "{question}")
    ])
    generate_chain = generate_prompt | llm | StrOutputParser()

    # 反思链
    reflect_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个严厉的批评者。找出回答中的逻辑错误、事实错误、遗漏点。"
                    "用中文逐条列出，每条一句话。如果很好就说'答案已经很好'。"),
        ("user", "问题: {question}\n\n回答:\n{answer}\n\n请批评。")
    ])
    reflect_chain = reflect_prompt | llm | StrOutputParser()

    # 改进链
    refine_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是改进专家。根据批评意见逐条改进答案。"),
        ("user", "问题: {question}\n\n上一轮回答:\n{answer}\n\n批评:\n{critique}\n\n请改进。")
    ])
    refine_chain = refine_prompt | llm | StrOutputParser()

    return generate_chain, reflect_chain, refine_chain


def run(
    question: str,
    model_name: str = "gpt-4o-mini",
    max_iterations: int = 3,
) -> str:
    """
    运行 Reflection Agent

    Args:
        question: 用户问题
        model_name: 模型名称
        max_iterations: 最大反思轮次

    Returns:
        最终答案
    """
    llm = ChatOpenAI(model=model_name, temperature=0.7)
    generate_chain, reflect_chain, refine_chain = create_reflection_chains(llm)

    print(f"🎯 问题: {question}\n")

    # 生成初始答案
    current_answer = generate_chain.invoke({"question": question})
    print(f"📝 初始答案:\n{current_answer}\n")

    for i in range(1, max_iterations + 1):
        print(f"\n--- 第 {i} 轮反思 ---")

        # 反思
        critique = reflect_chain.invoke({"question": question, "answer": current_answer})
        print(f"🔍 批评意见:\n{critique}\n")

        if "已经很好" in critique:
            print("✅ 停止反思")
            break

        # 改进
        current_answer = refine_chain.invoke({
            "question": question,
            "answer": current_answer,
            "critique": critique,
        })
        print(f"✨ 改进后答案:\n{current_answer}\n")

    return current_answer


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 LangChain Reflection 的使用"""
    answer = run("解释一下什么是区块链，以及它和传统数据库有什么区别。")
    print(f"\n{'='*50}")
    print(f"最终答案:\n{answer}")


if __name__ == "__main__":
    main()
