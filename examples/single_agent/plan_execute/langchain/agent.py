"""
Plan-and-Execute Agent - LangChain 实现
=========================================

使用 LangChain 框架实现的 Plan-and-Execute Agent。
LangChain AgentExecutor 虽然主要是 ReAct 模式，但我们可以通过
Prompt 工程来实现两阶段规划-执行的效果。
"""

from typing import Optional

import json

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==============================================================================
# 定义工具
# ==============================================================================

@tool
def search_database(query: str) -> str:
    """从数据库搜索信息"""
    return f"[数据库结果] 查询 '{query}' 成功，找到 3 条相关记录。"


@tool
def send_email(to: str, content: str) -> str:
    """发送邮件"""
    return f"[邮件结果] 已向 {to} 发送邮件，内容: {content[:30]}..."


@tool
def generate_report(data: str) -> str:
    """基于数据生成报告"""
    return f"[报告结果] 基于数据生成了一份报告，共 {len(data)} 字符。"


TOOLS = [search_database, send_email, generate_report]


# ==============================================================================
# Plan-and-Execute 实现
# ==============================================================================

def plan_task(llm: ChatOpenAI, task: str) -> list[dict]:
    """
    规划阶段：让 LLM 输出执行计划

    Args:
        llm: ChatOpenAI 实例
        task: 要完成的任务

    Returns:
        计划步骤列表
    """
    plan_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个任务规划专家。为任务制定执行计划。
可用工具: search_database, send_email, generate_report
输出 JSON 格式: {{"plan": [{{"tool": "...", "params": "...", "purpose": "..."}}]}}"""),
        ("user", "任务: {task}")
    ])

    # 使用 StrOutputParser + 手动 JSON 解析（更可靠）
    chain = plan_prompt | llm | StrOutputParser()
    response = chain.invoke({"task": task})

    # 手动提取 JSON 并解析
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        data = json.loads(response[start:end])
        plan = data.get("plan", [])
    except (json.JSONDecodeError, ValueError):
        print(f"Warning: JSON 解析失败，使用默认三步计划")
        plan = [
            {"tool": "search_database", "params": f"收集关于 {task} 的信息", "purpose": "收集信息"},
            {"tool": "generate_report", "params": f"分析 {task} 的结果", "purpose": "分析数据"},
            {"tool": "send_email", "params": f"发送 {task} 报告", "purpose": "发送报告"},
        ]

    print(f"📋 生成的计划: {plan}")
    return plan


def execute_plan(llm: ChatOpenAI, plan: list[dict], task: str) -> str:
    """
    执行阶段：逐步执行计划

    Args:
        llm: ChatOpenAI 实例
        plan: 计划步骤
        task: 原始任务

    Returns:
        执行结果总结
    """
    results = []
    tool_map = {tool.name: tool for tool in TOOLS}

    # 定义每个工具对应的参数名映射
    tool_param_map = {
        "search_database": "query",
        "send_email": "to",  # send_email 需要 to 和 content，简化处理
        "generate_report": "data",
    }

    for i, step in enumerate(plan):
        tool_name = step.get("tool", "")
        params = step.get("params", "")
        purpose = step.get("purpose", "")

        print(f"\n--- 执行 Step {i+1}: {purpose} ---")

        if tool_name in tool_map:
            # 根据工具名获取正确的参数 key
            param_key = tool_param_map.get(tool_name, "data")

            # send_email 特殊处理：需要两个参数
            if tool_name == "send_email":
                result = tool_map[tool_name].invoke({"to": "boss@company.com", "content": params})
            else:
                result = tool_map[tool_name].invoke({param_key: params})

            print(f"工具返回: {result}")
            results.append({"step": i + 1, "tool": tool_name, "result": str(result)})
        else:
            print(f"[错误] 未知工具: {tool_name}")

    # 汇总结果
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个任务执行专家。根据执行记录给出任务完成情况总结。"),
        ("user", "原始任务: {task}\n\n执行记录: {results}")
    ])

    summary = summary_prompt | llm
    final = summary.invoke({"task": task, "results": str(results)})
    return final.content


def run(task: str, model_name: str = "gpt-4o-mini") -> str:
    """
    完整运行 Plan-and-Execute

    Args:
        task: 任务描述
        model_name: 模型名称

    Returns:
        最终结果
    """
    llm = ChatOpenAI(model=model_name, temperature=0.7)

    print(f"🎯 任务: {task}")
    plan = plan_task(llm, task)
    result = execute_plan(llm, plan, task)
    return result


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 LangChain Plan-and-Execute 的使用"""
    result = run("查询本月销售数据，生成月度报告，并发送给老板")
    print(f"\n{'='*50}")
    print(f"最终结果:\n{result}")


if __name__ == "__main__":
    main()
