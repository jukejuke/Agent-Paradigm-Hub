"""
Coding Agent 单元测试（无需 LLM / API Key）

运行方式（在项目根目录）：
    python -m pytest examples/single_agent/coding_agent/tests/ -q
或直接运行：
    python examples/single_agent/coding_agent/tests/test_coding_agent.py
"""
import pathlib
import sys
import tempfile
import unittest

from langgraph.graph import END

# 兼容从任意目录直接运行：先尝试绝对导入，失败则把项目根目录加入 sys.path
try:
    from examples.single_agent.coding_agent.langgraph import agent, prompts, tools
except ModuleNotFoundError:
    # parents[4] 为项目根目录（tests → coding_agent → single_agent → examples → 根）
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
    from examples.single_agent.coding_agent.langgraph import agent, prompts, tools


class TestRunPython(unittest.TestCase):
    """run_python 工具：正常执行 / 语法错误 / 运行时异常 / 超时"""

    def test_normal_execution(self):
        # 正常代码：stdout 应包含计算结果，error 为空
        result = tools.run_python("print(1 + 1)")
        self.assertIn("2", result["stdout"])
        self.assertEqual(result["error"], "")
        self.assertEqual(result["returncode"], 0)

    def test_syntax_error(self):
        # 语法错误：被捕获为 error 字段，不抛出异常
        result = tools.run_python("def foo(:")
        self.assertTrue(result["error"])
        self.assertNotEqual(result["returncode"], 0)

    def test_runtime_error(self):
        # 运行时异常：同样被捕获，不抛出异常
        result = tools.run_python("raise ValueError('boom')")
        self.assertTrue(result["error"])
        self.assertNotEqual(result["returncode"], 0)

    def test_timeout(self):
        # 死循环：超时保护生效，不无限挂起
        result = tools.run_python("while True: pass", timeout=1)
        self.assertIn("超时", result["error"])


class TestShouldContinue(unittest.TestCase):
    """决策逻辑：should_continue 三分支"""

    def _make_state(self, **overrides):
        """构造最小合法状态字典"""
        state = {
            "task": "demo",
            "code": "",
            "execution_output": "",
            "execution_error": "",
            "feedback": "",
            "passed": False,
            "iteration": 0,
            "max_iterations": 3,
            "history": [],
        }
        state.update(overrides)
        return state

    def test_passed_ends(self):
        # 通过验收 → 结束
        self.assertEqual(agent.should_continue(self._make_state(passed=True)), END)

    def test_not_passed_fixes(self):
        # 未通过且未达上限 → 继续修复
        self.assertEqual(agent.should_continue(self._make_state(passed=False, iteration=1)), "fix")

    def test_max_iterations_ends(self):
        # 未通过但达到最大迭代次数 → 结束
        self.assertEqual(agent.should_continue(self._make_state(passed=False, iteration=3)), END)


class TestStripCodeFence(unittest.TestCase):
    """Markdown 代码围栏剥离"""

    def test_with_fence(self):
        # 标准围栏剥离
        self.assertEqual(prompts.strip_code_fence("```python\nprint(1)\n```"), "print(1)")

    def test_without_fence(self):
        # 无围栏：去除首尾空白后原样返回
        self.assertEqual(prompts.strip_code_fence("  print(1)  "), "print(1)")

    def test_with_explanation(self):
        # 围栏前后有说明文字：只提取围栏内代码
        text = "说明文字\n```python\nprint(1)\n```\n结束语"
        self.assertEqual(prompts.strip_code_fence(text), "print(1)")


class TestReadFile(unittest.TestCase):
    """read_file 工具：读取已存在文件 / 不存在的文件"""

    def test_read_existing_file(self):
        # 读取已存在的文件：ok=True，内容一致
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "hello.txt"
            path.write_text("print('hello')", encoding="utf-8")
            result = tools.read_file(str(path))
            self.assertTrue(result["ok"])
            self.assertEqual(result["content"], "print('hello')")
            self.assertEqual(result["error"], "")

    def test_read_missing_file(self):
        # 读取不存在的文件：ok=False + error，不抛异常
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "no_such_file.py"
            result = tools.read_file(str(path))
            self.assertFalse(result["ok"])
            self.assertTrue(result["error"])


class TestSaveCode(unittest.TestCase):
    """save_code 工具：正常保存 / 自动创建父目录 / 非法路径"""

    def test_save_and_content(self):
        # 正常保存到临时目录：ok=True，回读内容一致
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "out.py"
            result = tools.save_code(str(path), "print(1)")
            self.assertTrue(result["ok"])
            self.assertEqual(result["path"], str(path))
            self.assertEqual(path.read_text(encoding="utf-8"), "print(1)")

    def test_save_creates_parent_dir(self):
        # 父目录不存在时自动创建
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "a" / "b" / "out.py"
            result = tools.save_code(str(path), "x = 1")
            self.assertTrue(result["ok"])
            self.assertTrue(path.exists())

    def test_save_invalid_path(self):
        # 非法路径（指向已存在的目录）：ok=False + error，不抛异常
        with tempfile.TemporaryDirectory() as tmp:
            result = tools.save_code(tmp, "print(1)")
            self.assertFalse(result["ok"])
            self.assertTrue(result["error"])


class TestBuildGraph(unittest.TestCase):
    """状态图可成功构建（不调用 LLM / 不需要 API Key）"""

    def test_graph_contains_expected_nodes(self):
        # 构建编译后的图，校验五个核心节点（含保存节点）
        compiled = agent.build_coding_agent_graph()
        names = {node.name for node in compiled.get_graph().nodes.values()}
        self.assertTrue({"generate", "execute", "evaluate", "fix", "save"} <= names)


if __name__ == "__main__":
    unittest.main()
