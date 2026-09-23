"""
Coding Agent（代码生成智能体）- LangGraph 实现
============================================
核心流程：生成 → 执行 → 评估 →（未通过则修复）→ 再执行，直到通过验收或达到最大迭代次数。
四大核心组件：
  1. 智能体状态管理模块：state.py（CodingAgentState + 迭代历史）
  2. 工具调用接口：tools.py（run_python 沙箱代码执行）
  3. 决策逻辑单元：evaluate 节点 + should_continue 条件边
  4. 循环执行机制：StateGraph 条件循环（generate → execute → evaluate ⇄ fix）
"""
import os
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from dotenv import load_dotenv
from pathlib import Path

# 相对导入优先；直接以脚本方式运行时回退到绝对导入
try:
    from .state import CodingAgentState
    from .tools import run_python
    from .prompts import GENERATE_PROMPT, EVALUATE_PROMPT, FIX_PROMPT, strip_code_fence
except ImportError:
    from examples.single_agent.coding_agent.langgraph.state import CodingAgentState
    from examples.single_agent.coding_agent.langgraph.tools import run_python
    from examples.single_agent.coding_agent.langgraph.prompts import (
        GENERATE_PROMPT, EVALUATE_PROMPT, FIX_PROMPT, strip_code_fence,
    )

# 定位项目根目录并加载 .env（支持从任意目录直接运行本文件）
load_dotenv(Path(__file__).resolve().parents[4] / ".env")

# LLM 实例采用懒加载：避免在未配置 API Key 的测试环境中导入模块即报错
_llm = None

def _get_llm() -> ChatOpenAI:
    """懒加载统一的 ChatOpenAI 实例（代码生成使用较低温度提升确定性）"""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.2,
        )
    return _llm


def _invoke_llm(system_prompt: str, user_content: str) -> str:
    """
    统一调用 LLM 并累积返回文本

    Args:
        system_prompt: 系统提示词
        user_content: 用户消息内容

    Returns:
        LLM 生成的文本（去除首尾空白）

    Raises:
        RuntimeError: LLM 调用失败（中文提示，便于用户排查 API 配置）
    """
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
    try:
        reply = ""
        for chunk in _get_llm().stream(messages):
            reply += chunk.content
        return reply.strip()
    except Exception as e:
        raise RuntimeError(f"LLM 调用失败，请检查 OPENAI_API_KEY / 网络配置：{e}") from e


def generate_node(state: CodingAgentState) -> dict:
    """
    生成节点：根据任务需求生成初始代码

    Args:
        state: 当前图状态

    Returns:
        更新状态字典 {"code": 生成的代码}
    """
    code = strip_code_fence(_invoke_llm(GENERATE_PROMPT, f"任务需求：\n{state['task']}"))
    if not code:
        raise RuntimeError("LLM 未返回有效代码，无法继续执行")
    print(f"── [生成] 已生成初始代码（{len(code)} 字符）")
    return {"code": code}


def execute_node(state: CodingAgentState) -> dict:
    """
    执行节点：在沙箱中执行当前代码并收集输出（工具调用接口）

    Args:
        state: 当前图状态

    Returns:
        更新状态字典 {"execution_output": ..., "execution_error": ...}
    """
    result = run_python(state["code"])
    error = result["error"]
    print(f"── [执行] returncode={result['returncode']}，stdout 预览: {result['stdout'][:120]!r}")
    if error:
        print(f"── [执行] 错误: {error[:300]}")
    return {"execution_output": result["stdout"], "execution_error": error}


def evaluate_node(state: CodingAgentState) -> dict:
    """
    评估节点：综合规则与 LLM 评审判断是否通过验收（决策逻辑单元）

    规则：执行无错误（execution_error 为空）为硬性前提；
    LLM 评审：判断代码是否满足任务需求。
    LLM 评审失败时回退为纯规则判定（不中断流程）。

    Args:
        state: 当前图状态

    Returns:
        更新状态字典 {"passed": bool, "feedback": str, "history": [记录]}
    """
    # 规则判定：执行是否报错
    exec_ok = not state.get("execution_error", "")

    content = (
        f"任务需求：{state['task']}\n\n"
        f"生成的代码：\n{state['code']}\n\n"
        f"执行结果（stdout）：\n{state.get('execution_output', '')}\n\n"
        f"执行错误：{state.get('execution_error', '') or '无'}"
    )
    llm_pass, feedback = None, ""
    try:
        review = _invoke_llm(EVALUATE_PROMPT, content)
        m = re.search(r"Pass\s*:\s*(yes|no)", review, re.IGNORECASE)
        llm_pass = m.group(1).lower() == "yes" if m else None
        fm = re.search(r"Feedback\s*:\s*(.+)", review, re.IGNORECASE | re.DOTALL)
        feedback = fm.group(1).strip() if fm else review.strip()
    except RuntimeError as e:
        print(f"── [评估] LLM 评审失败，回退为规则判定: {e}")
        feedback = f"（LLM 评审不可用，按执行结果规则判定）{e}"

    # 综合判定：LLM 不可用则只按执行规则；否则须同时满足执行无错 + LLM 通过
    if llm_pass is None:
        passed = exec_ok
    else:
        passed = exec_ok and llm_pass
    print(f"── [评估] passed={passed}，feedback: {feedback[:200]}")
    return {
        "passed": passed,
        "feedback": feedback,
        "history": [
            {
                "iteration": state.get("iteration", 0),
                "passed": passed,
                "feedback": feedback,
                "execution_output_head": state.get("execution_output", "")[:200],
            }
        ],
    }


def fix_node(state: CodingAgentState) -> dict:
    """
    修复节点：根据评审意见修复代码并进入下一轮迭代

    Args:
        state: 当前图状态

    Returns:
        更新状态字典 {"code": 修复后代码, "iteration": 轮次+1}
    """
    content = (
        f"任务需求：{state['task']}\n\n"
        f"当前代码：\n{state['code']}\n\n"
        f"执行结果（stdout）：\n{state.get('execution_output', '')}\n\n"
        f"执行错误：{state.get('execution_error', '') or '无'}\n\n"
        f"评审意见：{state.get('feedback', '')}"
    )
    new_code = strip_code_fence(_invoke_llm(FIX_PROMPT, content))
    if not new_code:
        raise RuntimeError("LLM 未返回修复后的代码，无法继续执行")
    next_iter = state.get("iteration", 0) + 1
    print(f"── [修复] 第 {next_iter} 轮修复完成（{len(new_code)} 字符）")
    return {"code": new_code, "iteration": next_iter}


def should_continue(state: CodingAgentState) -> str:
    """
    决策逻辑单元：条件边判断函数

    通过验收（passed=True）或达到最大迭代次数 → 结束（返回 END）；
    否则 → 继续修复（返回 "fix"）。

    Args:
        state: 当前图状态

    Returns:
        "fix" 或 END（"__end__"）
    """
    if state.get("passed") or state.get("iteration", 0) >= state.get("max_iterations", 3):
        return END
    return "fix"


def build_coding_agent_graph():
    """
    构建并编译 Coding Agent 状态图（循环执行机制）

    流程：generate → execute → evaluate ─(passed 或达到上限)→ END
                                        └─(未通过)→ fix → execute → ...
    """
    graph = StateGraph(CodingAgentState)
    graph.add_node("generate", generate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("fix", fix_node)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "execute")
    graph.add_edge("execute", "evaluate")
    graph.add_conditional_edges("evaluate", should_continue, {"fix": "fix", END: END})
    graph.add_edge("fix", "execute")
    return graph.compile()


def run(task: str, max_iterations: int = 3) -> str:
    """
    运行完整的 Coding Agent 流程（updates + values 双模式流式输出）

    Args:
        task: 用户的任务需求
        max_iterations: 最大修复轮次（默认 3）

    Returns:
        最终验收的代码字符串
    """
    graph = build_coding_agent_graph()
    initial_state = {
        "task": task,
        "code": "",
        "execution_output": "",
        "execution_error": "",
        "feedback": "",
        "passed": False,
        "iteration": 0,
        "max_iterations": max_iterations,
    }
    print(f"\n{'=' * 60}")
    print(f"任务需求: {task}")
    print(f"最大修复轮次: {max_iterations}")
    print(f"{'=' * 60}\n")

    final_state = None
    try:
        for mode, payload in graph.stream(initial_state, stream_mode=["updates", "values"]):
            if mode == "updates":
                for node_name in payload.keys():
                    print(f"\n[节点] {node_name}")   # 节点细节由节点内 print 输出
            else:  # mode == "values"：取最终完整状态
                final_state = payload
    except RuntimeError as e:
        print(f"\n[运行失败] {e}")
        raise

    print("\n" + "=" * 60)
    if final_state.get("passed"):
        print("验收结果: 通过 ✅")
    else:
        print("验收结果: 达到最大迭代次数（未通过）")
    print(f"最终代码:\n{final_state['code']}")
    return final_state["code"]


def main():
    """演示 Coding Agent 的代码生成与迭代修复流程"""
    task = (
        "编写一个函数 fibonacci(n)，返回第 n 个斐波那契数（n 从 1 开始计数，fibonacci(1)=0, fibonacci(2)=1）。\n"
        "要求：\n"
        "1. n 为负数或非整数时抛出 ValueError；\n"
        "2. 在 if __name__ == '__main__': 中打印 fibonacci(1) 到 fibonacci(10) 的结果。"
    )
    run(task, max_iterations=3)


if __name__ == "__main__":
    main()
