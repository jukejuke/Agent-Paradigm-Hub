"""
工具调用接口：与代码执行环境交互
================================
提供 run_python 沙箱代码执行工具，供 Coding Agent 节点函数直接调用，
并通过 run_python_tool（@tool 封装）供 LLM 工具调用使用。
"""
import subprocess
import sys
import tempfile

from langchain_core.tools import tool


def run_python(code: str, timeout: int = 30) -> dict:
    """
    在沙箱子进程中执行 Python 代码并返回结构化结果

    Args:
        code: 待执行的 Python 代码
        timeout: 执行超时秒数（默认 30）

    Returns:
        结构化结果字典 {"stdout": str, "stderr": str, "returncode": int|None, "error": str}；
        正常（returncode==0）时 error 为空字符串，异常情况一律捕获并写入 error，绝不抛出
    """
    result = {"stdout": "", "stderr": "", "returncode": None, "error": ""}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["returncode"] = proc.returncode
        if proc.returncode != 0:
            # 非零返回码：优先取 stderr 内容，为空时给出退出码提示
            result["error"] = proc.stderr.strip() or f"进程退出码非零: {proc.returncode}"
    except subprocess.TimeoutExpired:
        result["error"] = f"执行超时（超过 {timeout} 秒）"
    except Exception as e:
        result["error"] = f"执行失败: {e}"
    return result


# @tool 封装，供 LLM 工具调用使用；节点函数直接调用 run_python 底层函数
run_python_tool = tool(run_python)
