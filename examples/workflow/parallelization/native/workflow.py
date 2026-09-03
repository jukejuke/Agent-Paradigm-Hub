"""
Parallelization Workflow - 原生实现
====================================

核心思想：将一个大任务拆分成多个相互独立的子任务，
并行执行（使用 asyncio），然后汇总结果。

适合场景：批量内容生成、多维度分析、并发检索等无依赖关系的任务。
"""

import asyncio
from typing import Optional

import os
import sys

# 将项目根目录加入 sys.path，使本脚本可在任意工作目录运行
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from utils.llm_client import LLMClient


# ==============================================================================
# Parallelization 实现
# ==============================================================================

class ParallelizationWorkflow:
    """并行处理工作流 - 并发执行独立子任务"""

    def __init__(self, llm: Optional[LLMClient] = None):
        """
        初始化并行工作流

        Args:
            llm: LLM 客户端
        """
        self.llm = llm or LLMClient()

    async def _process_single(self, subtask: str, index: int, total: int) -> tuple[int, str]:
        """
        处理单个子任务（需要异步包装以实现并发调度）

        Args:
            subtask: 子任务描述
            index: 当前索引
            total: 总数量

        Returns:
            (索引, 结果)
        """
        system_prompt = f"""你是一个高效的处理者。当前任务 {index}/{total}。
快速、准确地完成分配给你的子任务。"""

        messages = [{"role": "user", "content": subtask}]

        print(f"  ⚡ [任务 {index}/{total}] 开始处理: {subtask[:30]}...")
        # asyncio.to_thread 让同步的 LLM 调用以非阻塞方式运行
        result = await asyncio.to_thread(
            self.llm.chat, messages, system_prompt
        )
        print(f"  ✅ [任务 {index}/{total}] 完成")
        return index, result

    async def _parallel_process(self, subtasks: list[str]) -> list[str]:
        """
        并行处理所有子任务

        Args:
            subtasks: 子任务列表

        Returns:
            按顺序排列的结果列表
        """
        total = len(subtasks)
        tasks = [
            self._process_single(st, i + 1, total)
            for i, st in enumerate(subtasks)
        ]

        # asyncio.gather 并发执行所有任务
        results = await asyncio.gather(*tasks)

        # 按索引排序保证顺序一致
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def run(self, main_task: str, subtasks: list[str]) -> str:
        """
        运行完整并行工作流

        Args:
            main_task: 主任务描述（用于最终汇总）
            subtasks: 需要并行执行的子任务列表

        Returns:
            汇总后的最终结果
        """
        print(f"\n{'='*50}")
        print(f"🎯 主任务: {main_task}")
        print(f"📋 拆分为 {len(subtasks)} 个并行子任务")
        print(f"{'='*50}")

        # 1. 并行执行所有子任务
        print("\n⚙️ 阶段 1: 并行处理子任务")
        results = asyncio.run(self._parallel_process(subtasks))

        # 打印各部分结果
        print("\n📦 各子任务结果:")
        for i, (task, result) in enumerate(zip(subtasks, results)):
            print(f"\n--- 子任务 {i+1}: {task} ---")
            print(f"结果: {result[:100]}...")

        # 2. 汇总结果
        print(f"\n🔗 阶段 2: 汇总结果")
        combined = "\n\n".join(f"[子任务{i+1}]\n{r}" for i, r in enumerate(results))

        summary_prompt = f"""主任务: {main_task}

各并行子任务的执行结果:
{combined}

请整合以上所有结果，形成一个完整、连贯的最终回答。"""

        messages = [{"role": "user", "content": summary_prompt}]
        final = self.llm.chat(messages, system_prompt="你是一个优秀的内容整合专家。")

        return final


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 Parallelization 的使用"""
    workflow = ParallelizationWorkflow()

    main_task = "撰写一份关于云服务选型的指南"
    subtasks = [
        "分析 AWS 的主要优势和适用场景",
        "分析 Azure 的主要优势和适用场景",
        "分析 GCP 的主要优势和适用场景",
        "对比三家云服务的价格策略",
    ]

    result = workflow.run(main_task, subtasks)
    print(f"\n{'='*50}")
    print(f"最终汇总:\n{result}")


if __name__ == "__main__":
    main()
