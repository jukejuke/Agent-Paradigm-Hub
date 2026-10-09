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
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from dotenv import load_dotenv
from pathlib import Path

# 相对导入优先；直接以脚本方式运行时回退到绝对导入
try:
    from .state import CodingAgentState
    from .tools import read_file, run_python, save_code
    from .prompts import (
        GENERATE_PROMPT, EVALUATE_PROMPT, FIX_PROMPT, MODIFY_PROMPT, strip_code_fence,
    )
except ImportError:
    from examples.single_agent.coding_agent.langgraph.state import CodingAgentState
    from examples.single_agent.coding_agent.langgraph.tools import read_file, run_python, save_code
    from examples.single_agent.coding_agent.langgraph.prompts import (
        GENERATE_PROMPT, EVALUATE_PROMPT, FIX_PROMPT, MODIFY_PROMPT, strip_code_fence,
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
    生成节点：根据任务需求生成或修改代码

    从零生成模式（target_file 为空）：LLM 根据任务需求编写初始代码；
    修改模式（target_file 非空）：LLM 基于现有代码按任务需求修改。

    Args:
        state: 当前图状态

    Returns:
        更新状态字典 {"code": 生成的代码}
    """
    if state.get("target_file"):
        # 修改模式：基于现有代码进行修改
        user_content = f"任务需求：\n{state['task']}\n\n现有代码：\n{state['code']}"
        code = strip_code_fence(_invoke_llm(MODIFY_PROMPT, user_content))
        print(f"── [生成] 已基于现有代码完成修改（{len(code)} 字符）")
    else:
        # 从零生成模式
        code = strip_code_fence(_invoke_llm(GENERATE_PROMPT, f"任务需求：\n{state['task']}"))
        print(f"── [生成] 已生成初始代码（{len(code)} 字符）")
    if not code:
        raise RuntimeError("LLM 未返回有效代码，无法继续执行")
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


def save_node(state: CodingAgentState) -> dict:
    """
    保存节点：将最终代码写入本地文件

    Args:
        state: 当前图状态（需含 output_path 与 code）

    Returns:
        更新状态字典 {"history": [保存记录]}

    Raises:
        RuntimeError: 保存失败（中文提示）
    """
    path = state.get("output_path", "")
    result = save_code(path, state["code"])
    if not result["ok"]:
        raise RuntimeError(result["error"])
    print(f"── [保存] 代码已保存到: {result['path']}")
    return {"history": [{"event": "saved", "path": result["path"]}]}


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

    流程：generate → execute → evaluate ─(passed 或达到上限)→ save → END
                                        └─(未通过)→ fix → execute → ...
    """
    graph = StateGraph(CodingAgentState)
    graph.add_node("generate", generate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("fix", fix_node)
    graph.add_node("save", save_node)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "execute")
    graph.add_edge("execute", "evaluate")
    graph.add_conditional_edges("evaluate", should_continue, {"fix": "fix", "save": "save"})
    graph.add_edge("fix", "execute")
    graph.add_edge("save", END)
    return graph.compile()


def run(
    task: str,
    max_iterations: int = 3,
    target_file: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    运行完整的 Coding Agent 流程（updates + values 双模式流式输出）

    Args:
        task: 用户的任务需求
        max_iterations: 最大修复轮次（默认 3）
        target_file: 指定要修改的目标文件路径（为空则从零生成）
        output_path: 最终代码保存路径（为空时：修改模式默认写回 target_file，
                     从零生成默认保存到 <项目根>/outputs/coding_agent_output.py）

    Returns:
        最终验收的代码字符串
    """
    graph = build_coding_agent_graph()
    root = Path(__file__).resolve().parents[4]
    # 解析最终保存路径：显式 output_path > target_file（就地写回）> 默认 outputs 目录
    resolved_output = output_path or (target_file or str(root / "outputs" / "coding_agent_output.py"))

    # 修改模式：读取目标文件内容作为初始代码
    initial_code = ""
    if target_file:
        read_result = read_file(target_file)
        if not read_result["ok"]:
            raise RuntimeError(f"读取目标文件失败: {read_result['error']}")
        initial_code = read_result["content"]
        print(f"已读取目标文件: {target_file}（{len(initial_code)} 字符）")

    initial_state = {
        "task": task,
        "code": initial_code,
        "execution_output": "",
        "execution_error": "",
        "feedback": "",
        "passed": False,
        "iteration": 0,
        "max_iterations": max_iterations,
        "target_file": target_file or "",
        "output_path": resolved_output,
    }
    print(f"\n{'=' * 60}")
    print(f"任务需求: {task}")
    print(f"最大修复轮次: {max_iterations}")
    print(f"目标文件: {target_file or '（无，从零生成）'}")
    print(f"保存路径: {resolved_output}")
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
    print(f"最终代码已保存到: {final_state['output_path']}")
    print(f"最终代码:\n{final_state['code']}")
    return final_state["code"]


def demo_modify_file():
    """演示修改指定文件：读取现有脚本，要求 LLM 修改后写回"""
    root = Path(__file__).resolve().parents[4]
    sample_path = root / "outputs" / "sample_calc.py"
    # 准备一个示例脚本作为待修改文件
    sample_code = (
        "def multiply(a: int, b: int) -> int:\n"
        "    \"\"\"返回两个整数的乘积\"\"\"\n"
        "    return a * b\n\n"
        "if __name__ == '__main__':\n"
        "    print(multiply(3, 4))\n"
    )
    save_code(str(sample_path), sample_code)
    print(f"示例文件已创建: {sample_path}")

    task = (
        "将 multiply 函数改为接收三个参数并返回三者乘积（multiply(a, b, c)），"
        "并在 __main__ 中打印 multiply(2, 3, 4) 的结果。"
    )
    run(task, max_iterations=2, target_file=str(sample_path))


def main():
    """演示 Coding Agent 的代码生成、迭代修复、保存到本地与修改指定文件"""
    print("=" * 60)
    print("场景一：从零生成代码并保存到本地")
    print("=" * 60)
    task = (
        "编写一个函数 fibonacci(n)，返回第 n 个斐波那契数（n 从 1 开始计数，fibonacci(1)=0, fibonacci(2)=1）。\n"
        "要求：\n"
        "1. n 为负数或非整数时抛出 ValueError；\n"
        "2. 在 if __name__ == '__main__': 中打印 fibonacci(1) 到 fibonacci(10) 的结果。"
    )
    run(task, max_iterations=3)

    print("\n" + "=" * 60)
    print("场景二：修改指定文件")
    print("=" * 60)
    demo_modify_file()


if __name__ == "__main__":
    main()
