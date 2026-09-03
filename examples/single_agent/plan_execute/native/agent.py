"""
Plan-and-Execute Agent - 原生实现
==================================

核心思想：将任务分为两个阶段——规划（Plan）和执行（Execute）。
第一阶段让 LLM 输出完成任务所需的步骤列表。
第二阶段逐步执行每个步骤，并根据前序步骤的结果动态调整。

与 ReAct 的区别：ReAct 是边想边做，Plan-and-Execute 是先想好了再做。
"""

import json

import os
import sys
from typing import Optional

# 将项目根目录加入 sys.path，使本脚本可在任意工作目录运行
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from utils.llm_client import LLMClient


# ==============================================================================
# 工具函数
# ==============================================================================

def search_database(query: str) -> str:
    """模拟数据库查询工具"""
    return f"[数据库结果] 查询 '{query}' 成功，找到 3 条相关记录。"


def send_email(params: str) -> str:
    """模拟发送邮件工具"""
    return f"[邮件结果] 已按参数发送邮件: {params[:30]}..."


def generate_report(data: str) -> str:
    """模拟报告生成工具"""
    return f"[报告结果] 基于数据生成了一份报告，共 {len(data)} 字符。"


AVAILABLE_TOOLS = {
    "search_database": search_database,
    "send_email": send_email,
    "generate_report": generate_report,
}


# ==============================================================================
# Plan-and-Execute Agent 实现
# ==============================================================================

class PlanAndExecuteAgent:
    """Plan-and-Execute 范式智能体 - 先规划后执行"""

    PLANNING_SYSTEM_PROMPT = """你是一个任务规划专家。请为用户的任务制定详细的执行计划。

可用工具:
- search_database(query): 从数据库搜索信息
- send_email(params): 发送邮件
- generate_report(data): 基于数据生成报告

请以 JSON 格式输出计划，格式如下:
```json
{{
  "plan": [
    {{"step": 1, "tool": "工具名", "params": "参数说明", "purpose": "该步骤的目的"}},
    {{"step": 2, ...}}
  ]
}}
```

确保每个步骤都明确且可执行。"""

    EXECUTION_SYSTEM_PROMPT = """你是一个任务执行专家。按照给定的计划逐步执行任务。

每完成一步，请输出:
Step N 完成: 简短描述
Status: success 或 failed
Next: 是否继续下一步"""

    def __init__(self, llm: Optional[LLMClient] = None, max_steps: int = 10):
        """
        初始化 Plan-and-Execute Agent

        Args:
            llm: LLM 客户端实例
            max_steps: 最大执行步数
        """
        self.llm = llm or LLMClient()
        self.max_steps = max_steps

    def plan(self, task: str) -> list[dict]:
        """
        第一阶段：规划任务步骤

        Args:
            task: 用户的任务描述

        Returns:
            计划步骤列表
        """
        messages = [{"role": "user", "content": f"请为以下任务制定执行计划:\n\n{task}"}]
        response = self.llm.chat(messages, system_prompt=self.PLANNING_SYSTEM_PROMPT)
        print(f"\n📋 生成的计划:\n{response}")

        # 解析 JSON 计划
        try:
            # 尝试提取 JSON 块
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]
            plan_data = json.loads(json_str)
            return plan_data.get("plan", [])
        except json.JSONDecodeError:
            # 如果 JSON 解析失败，返回原始响应作为单步计划
            print("Warning: JSON 解析失败，使用原始计划")
            return [{"step": 1, "tool": "search_database", "params": task, "purpose": task}]

    def execute(self, plan: list[dict], task: str) -> str:
        """
        第二阶段：逐步执行计划

        Args:
            plan: 计划步骤列表
            task: 原始任务（用于最终汇总）

        Returns:
            执行结果汇总
        """
        results = []

        for step_info in plan[:self.max_steps]:
            step_num = step_info.get("step", len(results) + 1)
            tool_name = step_info.get("tool", "")
            params = step_info.get("params", "")
            purpose = step_info.get("purpose", "")

            print(f"\n--- 执行 Step {step_num}: {purpose} ---")

            if tool_name in AVAILABLE_TOOLS:
                result = AVAILABLE_TOOLS[tool_name](params)
                print(f"工具返回: {result}")
                results.append({"step": step_num, "tool": tool_name, "result": result})
            else:
                result = f"[错误] 未知工具: {tool_name}"
                print(result)
                results.append({"step": step_num, "tool": tool_name, "result": result})

        # 让 LLM 汇总执行结果
        print(f"\n📝 汇总执行结果...")
        summary_prompt = f"""原始任务: {task}

执行记录:
{json.dumps(results, ensure_ascii=False, indent=2)}

请根据执行记录，给出任务的最终完成情况总结。"""

        messages = [{"role": "user", "content": summary_prompt}]
        final_summary = self.llm.chat(messages, system_prompt=self.EXECUTION_SYSTEM_PROMPT)
        return final_summary

    def run(self, task: str) -> str:
        """
        完整运行：先规划，后执行

        Args:
            task: 用户的任务描述

        Returns:
            最终结果汇总
        """
        print("=" * 50)
        print(f"🎯 任务: {task}")
        print("=" * 50)

        # 阶段一：规划
        plan = self.plan(task)
        if not plan:
            return "规划失败：无法生成执行计划"

        # 阶段二：执行
        result = self.execute(plan, task)
        return result


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 Plan-and-Execute Agent 的使用"""
    agent = PlanAndExecuteAgent()
    task = "查询本月销售数据，生成月度报告，并发送给老板"
    result = agent.run(task)
    print(f"\n{'='*50}")
    print(f"最终结果:\n{result}")


if __name__ == "__main__":
    main()
