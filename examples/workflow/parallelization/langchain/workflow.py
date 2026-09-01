"""
Parallelization Workflow - LangChain 实现
==========================================

使用 LangChain 的 Runnable 并发功能实现并行处理。
"""

from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


# ==============================================================================
# LangChain 实现
# ==============================================================================

class LangChainParallelization:
    """LangChain 并行处理"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)

    def run(self, main_task: str, subtasks: list[str]) -> str:
        """
        运行并行工作流

        Args:
            main_task: 主任务
            subtasks: 子任务列表

        Returns:
            汇总结果
        """
        print(f"\n🎯 主任务: {main_task}\n📋 {len(subtasks)} 个并行子任务\n{'='*50}")

        # 为每个子任务创建一个链
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是高效处理者，快速准确完成子任务。200字以内。"),
            ("user", "{subtask}")
        ])
        chain = prompt | self.llm | StrOutputParser()

        # 使用 RunnableParallel 并发执行
        runnable_map = RunnableParallel({
            f"task_{i}": chain for i in range(len(subtasks))
        })

        print("\n⚡ 并行执行中...")
        results = runnable_map.invoke({f"task_{i}": {"subtask": st} for i, st in enumerate(subtasks)})

        # 输出各结果
        print("\n📦 子任务结果:")
        for i, st in enumerate(subtasks):
            print(f"  ✅ [{i+1}] {st[:20]}... -> {results[f'task_{i}'][:40]}...")

        # 汇总
        print("\n🔗 汇总中...")
        combined = "\n\n".join(f"[{st}]\n{results[f'task_{i}']}" for i, st in enumerate(subtasks))
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是内容整合专家。整合多来源结果形成连贯回答。"),
            ("user", "主任务: {mt}\n\n各子任务结果:\n{c}")
        ])
        final = (summary_prompt | self.llm | StrOutputParser()).invoke({"mt": main_task, "c": combined})
        return final


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 LangChain Parallelization"""
    workflow = LangChainParallelization()
    result = workflow.run(
        "撰写一份关于云服务选型的指南",
        [
            "分析 AWS 的主要优势和适用场景",
            "分析 Azure 的主要优势和适用场景",
            "分析 GCP 的主要优势和适用场景",
            "对比三家云服务的价格策略",
        ]
    )
    print(f"\n{'='*50}")
    print(f"最终汇总:\n{result}")


if __name__ == "__main__":
    main()
