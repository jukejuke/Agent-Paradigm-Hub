"""
Orchestrator-Worker Agent - LangChain 实现
============================================

使用 LangChain 的 Runnable 抽象来实现 Orchestrator-Worker 模式。
"""

from typing import Optional

from langchain_openai import ChatOpenAI
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==============================================================================
# Worker 定义
# ==============================================================================

class LangChainWorker:
    """LangChain Worker Agent"""

    def __init__(self, name: str, role: str, llm: ChatOpenAI):
        """
        初始化 Worker

        Args:
            name: Worker 名称
            role: 角色描述
            llm: ChatOpenAI 实例
        """
        self.name = name
        self.role = role
        self.llm = llm

    def execute(self, task: str, context: str = "") -> str:
        """执行具体任务"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是 {name}，角色: {role}"),
            ("user", "上下文:\n{context}\n\n任务:\n{task}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"name": self.name, "role": self.role, "context": context, "task": task})
        print(f"  🤖 [{self.name}] 完成")
        return result


# ==============================================================================
# Orchestrator 定义
# ==============================================================================

class LangChainOrchestrator:
    """LangChain Orchestrator Agent"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """初始化 Orchestrator"""
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.workers = {
            "researcher": LangChainWorker("搜索研究员", "负责搜索和收集信息", self.llm),
            "analyst": LangChainWorker("数据分析师", "负责分析数据", self.llm),
            "writer": LangChainWorker("报告撰写者", "负责撰写报告", self.llm),
        }

    def decompose(self, task: str) -> list[dict]:
        """分解任务"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """将复杂任务分解为子步骤。
可用 Workers: researcher(搜索), analyst(分析), writer(写作)
输出 JSON: {{"steps": [{{"worker": "...", "task": "..."}}]}}"""),
            ("user", "{task}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"task": task})

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end])
            return data.get("steps", [])
        except (json.JSONDecodeError, ValueError):
            print(f"Warning: JSON 解析失败，使用默认三步计划")
            return [
                {"worker": "researcher", "task": f"收集关于 {task} 的信息"},
                {"worker": "analyst", "task": f"分析信息"},
                {"worker": "writer", "task": f"撰写报告"},
            ]

    def run(self, task: str) -> str:
        """运行 Orchestrator-Worker"""
        print(f"\n🎯 任务: {task}")
        print("=" * 50)

        # 分解
        print("\n📋 分解任务...")
        steps = self.decompose(task)
        for i, s in enumerate(steps):
            print(f"  Step {i+1}: [{s['worker']}] {s['task']}")

        # 执行
        print("\n⚙️ 调度 Workers...")
        context = ""
        for i, step in enumerate(steps):
            worker_name = step["worker"]
            if worker_name in self.workers:
                print(f"\n--- Step {i+1} ---")
                result = self.workers[worker_name].execute(step["task"], context)
                context += f"\n\n[Step {i+1}]\n{result}"

        # 汇总
        print("\n📊 汇总结果...")
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是内容整合专家。"),
            ("user", "任务: {task}\n\n各步骤结果:\n{results}\n\n请给出最终报告。")
        ])
        final = (summary_prompt | self.llm | StrOutputParser()).invoke({"task": task, "results": context})
        return final


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 LangChain Orchestrator-Worker"""
    orchestrator = LangChainOrchestrator()
    result = orchestrator.run("帮我写一份关于 AI Agent 发展趋势的简短研究报告")
    print(f"\n{'='*50}")
    print(f"最终结果:\n{result}")


if __name__ == "__main__":
    main()
