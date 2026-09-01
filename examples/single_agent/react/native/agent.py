"""
ReAct (Reason + Act) Agent - 原生实现
======================================

核心思想：推理（Reasoning）和行动（Acting）交替进行。
Agent 在每一步先通过 LLM 思考接下来做什么，然后执行工具调用，
观察结果后继续推理，直到得出最终答案。

参考: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
"""

import re
from typing import Optional

import os
import sys
# 将项目根目录加入 sys.path，使本脚本可在任意工作目录运行
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from utils.llm_client import LLMClient


# ==============================================================================
# 简单的工具函数（模拟 Agent 可调用的外部工具）
# ==============================================================================

def web_search(query: str) -> str:
    """模拟网络搜索工具"""
    return f"[搜索结果] 关于 '{query}' 的信息：这是一个模拟的搜索结果。"


def calculator(expression: str) -> str:
    """安全计算数学表达式（使用 ast 解析，避免 eval 安全风险）"""
    import ast
    import operator

    # 安全的运算符映射
    _SAFE_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }

    def _safe_eval(node):
        """递归安全求值 AST 节点"""
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            return _SAFE_OPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = _safe_eval(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        else:
            raise ValueError(f"不支持的表达式类型: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return f"[计算结果] {expression} = {result}"
    except Exception as e:
        return f"[计算错误] {e}"


# 注册可用工具
AVAILABLE_TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
}


# ==============================================================================
# ReAct Agent 实现
# ==============================================================================

class ReActAgent:
    """ReAct 范式智能体 - 推理与行动交替执行"""

    SYSTEM_PROMPT = """你是一个智能助手，使用 ReAct 模式解决问题。

你可以使用以下工具：
- web_search(query): 搜索网络获取信息
- calculator(expression): 计算数学表达式

每一步请严格按照以下格式输出：
Thought: 你的思考过程
Action: 要使用的工具名称（必须是 web_search 或 calculator）
Action Input: 工具的输入参数

当你得出最终答案时，请输出：
Thought: 你已经收集到足够信息的思考
Final Answer: 你的最终答案"""

    def __init__(self, llm: Optional[LLMClient] = None, max_steps: int = 10):
        """
        初始化 ReAct Agent

        Args:
            llm: LLM 客户端实例，默认创建新实例
            max_steps: 最大迭代步数，防止无限循环
        """
        self.llm = llm or LLMClient()
        self.max_steps = max_steps

    def run(self, question: str) -> str:
        """
        运行 ReAct Agent 解决问题

        Args:
            question: 用户的问题

        Returns:
            最终答案字符串
        """
        messages = [{"role": "user", "content": question}]

        for step in range(1, self.max_steps + 1):
            # 1. 让 LLM 思考并决定下一步行动
            response = self.llm.chat(messages, system_prompt=self.SYSTEM_PROMPT)
            print(f"\n--- Step {step} ---")
            print(f"Agent: {response}")

            # 2. 检查是否已得出最终答案
            final_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
            if final_match:
                return final_match.group(1).strip()

            # 3. 解析 Action 和 Action Input
            action_match = re.search(r"Action:\s*(\w+)", response)
            input_match = re.search(r"Action Input:\s*(.+)", response)

            if not action_match or not input_match:
                print("Warning: 无法解析 Action，尝试让 Agent 重新思考")
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "请按正确格式输出 Thought, Action, Action Input"})
                continue

            action = action_match.group(1)
            action_input = input_match.group(1).strip()

            # 4. 执行工具
            if action in AVAILABLE_TOOLS:
                tool_result = AVAILABLE_TOOLS[action](action_input)
                print(f"Tool ({action}): {tool_result}")
            else:
                tool_result = f"[错误] 未知工具: {action}"
                print(tool_result)

            # 5. 将结果加入对话历史
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {tool_result}"})

        return f"达到最大步数限制 ({self.max_steps})，未得出最终答案"


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 ReAct Agent 的使用"""
    agent = ReActAgent()
    question = "计算 15 * 27 的结果，然后搜索了解一下这个数字的含义"
    answer = agent.run(question)
    print(f"\n{'='*50}")
    print(f"最终答案: {answer}")


if __name__ == "__main__":
    main()
