"""
Orchestrator-Worker Agent - 原生实现
======================================

核心思想：一个 Orchestrator（编排者）负责理解用户意图、
分解任务、分配给多个 Worker（执行者），最后汇总结果。

适合场景：需要多个专业 Agent 协作完成的复杂任务，
如"做一个市场调研报告"——需要搜索、分析、写作等多个环节。
"""

import json
from typing import Optional

import os
import sys

# 将项目根目录加入 sys.path，使本脚本可在任意工作目录运行
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from utils.llm_client import LLMClient


# ==============================================================================
# Worker Agent 定义
# ==============================================================================

class WorkerAgent:
    """执行者 Agent - 被编排者调用完成具体任务"""

    def __init__(self, name: str, role: str, llm: Optional[LLMClient] = None):
        """
        初始化 Worker

        Args:
            name: Worker 名称（如 "搜索专家"）
            role: 角色描述，定义这个 Worker 的专长
            llm: LLM 客户端实例
        """
        self.name = name
        self.role = role
        self.llm = llm or LLMClient()

    def execute(self, task: str, context: str = "") -> str:
        """
        执行分配给它的具体任务

        Args:
            task: 分配的子任务描述
            context: 前序步骤的结果上下文

        Returns:
            执行结果
        """
        system_prompt = f"""你是 {self.name}，你的角色是: {self.role}
请专注完成分配给你的任务，输出专业、简洁的结果。"""

        messages = [{
            "role": "user",
            "content": f"上下文:\n{context}\n\n你的任务:\n{task}"
        }]

        print(f"  🤖 [{self.name}] 开始执行...")
        result = self.llm.chat(messages, system_prompt=system_prompt)
        print(f"  ✅ [{self.name}] 完成")
        return result


# ==============================================================================
# Orchestrator Agent 定义
# ==============================================================================

class OrchestratorAgent:
    """编排者 Agent - 分解任务并调度 Workers"""

    def __init__(self, llm: Optional[LLMClient] = None):
        """
        初始化 Orchestrator

        Args:
            llm: LLM 客户端实例
        """
        self.llm = llm or LLMClient()
        # 注册可用的 Worker
        self.workers = {
            "researcher": WorkerAgent("搜索研究员", "负责搜索和收集相关信息"),
            "analyst": WorkerAgent("数据分析师", "负责分析数据、发现规律"),
            "writer": WorkerAgent("报告撰写者", "负责将分析结果整合成通顺的报告"),
            "reviewer": WorkerAgent("质量审核员", "负责检查内容质量和准确性"),
        }

    def decompose_task(self, task: str) -> list[dict]:
        """
        将复杂任务分解为子任务列表

        Args:
            task: 用户的复杂任务

        Returns:
            子任务列表，每个包含 worker 和 task 字段
        """
        system_prompt = """你是任务编排专家。将用户的复杂任务分解为有序的子任务。
可用 Worker:
- researcher: 搜索收集信息
- analyst: 分析数据
- writer: 撰写报告
- reviewer: 审核质量

输出 JSON 格式:
{"steps": [{"worker": "worker名称", "task": "子任务描述", "depends_on": [依赖的步骤索引]}]}"""

        messages = [{"role": "user", "content": f"请分解以下任务:\n{task}"}]
        response = self.llm.chat(messages, system_prompt=system_prompt)

        # 解析 JSON
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end])
            return data.get("steps", [])
        except json.JSONDecodeError:
            print("Warning: JSON 解析失败，使用默认三步计划")
            return [
                {"worker": "researcher", "task": f"收集关于 {task} 的信息", "depends_on": []},
                {"worker": "analyst", "task": f"分析收集到的信息", "depends_on": [0]},
                {"worker": "writer", "task": f"撰写最终报告", "depends_on": [1]},
            ]

    def run(self, task: str) -> str:
        """
        完整运行 Orchestrator-Worker 流程

        Args:
            task: 用户的复杂任务

        Returns:
            最终汇总结果
        """
        print(f"\n🎯 Orchestrator 收到任务: {task}")
        print("=" * 50)

        # 1. 分解任务
        print("\n📋 阶段一: 任务分解")
        steps = self.decompose_task(task)
        print(f"分解为 {len(steps)} 个子任务")
        for i, step in enumerate(steps):
            print(f"  Step {i+1}: [{step.get('worker', 'unknown')}] {step.get('task', '')}")

        # 2. 按顺序执行（简化版，实际可并行无依赖的步骤）
        print("\n⚙️ 阶段二: 调度 Workers 执行")
        step_results = []
        context = ""

        for i, step in enumerate(steps):
            worker_name = step.get("worker", "")
            task_desc = step.get("task", "")

            if worker_name in self.workers:
                print(f"\n--- Step {i+1} ---")
                result = self.workers[worker_name].execute(task_desc, context)
                step_results.append(result)
                # 将前序结果作为后续步骤的上下文
                context += f"\n\n[Step {i+1} 结果]\n{result}"

        # 3. 汇总最终结果
        print("\n📊 阶段三: 汇总结果")
        summary_prompt = f"原始任务: {task}\n\n各 Worker 执行结果:\n{context}\n\n请整合所有结果，给出最终完整的回答。"
        messages = [{"role": "user", "content": summary_prompt}]
        final_summary = self.llm.chat(messages, system_prompt="你是一个优秀的内容整合者，请将多个来源的结果整合成通顺、完整的回答。")

        return final_summary


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 Orchestrator-Worker 的使用"""
    orchestrator = OrchestratorAgent()
    task = "帮我写一份关于 AI Agent 发展趋势的简短研究报告"
    result = orchestrator.run(task)
    print(f"\n{'='*50}")
    print(f"最终结果:\n{result}")


if __name__ == "__main__":
    main()
