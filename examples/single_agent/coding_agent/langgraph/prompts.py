"""
提示词模块
==========
定义 Coding Agent 各环节（生成 / 评估 / 修复）的中文系统提示词常量，
以及用于解析 LLM 输出代码围栏的工具函数 strip_code_fence。
"""
import re

# 生成环节系统提示词：资深 Python 工程师身份，根据任务需求生成初始代码
GENERATE_PROMPT = """
你是一名资深 Python 工程师，请根据用户的任务需求编写高质量、可直接运行的 Python 代码。

要求：
1. 只输出代码本身，不要输出任何解释性文字；
2. 使用 ```python ... ``` 围栏包裹代码；
3. 代码必须包含 if __name__ == "__main__": 自测入口并打印结果。
"""

# 评估环节系统提示词：严格代码评审员身份，输出两行评审结论
EVALUATE_PROMPT = """
你是一名严格的代码评审员，请根据任务需求、生成的代码与执行结果，判断代码是否满足需求。

要求：
1. 只输出两行，不要输出任何其它内容；
2. 第一行格式：Pass: yes/no（yes 表示通过，no 表示不通过）；
3. 第二行格式：Feedback: <简短中文评审意见>。
"""

# 修复环节系统提示词：资深工程师身份，根据反馈修复代码
FIX_PROMPT = """
你是一名资深 Python 工程师，请根据任务需求、当前代码、执行结果与评审意见，修复代码中的问题。

要求：
1. 只输出修复后的完整代码，不要输出任何解释性文字；
2. 使用 ```python ... ``` 围栏包裹代码；
3. 保持 if __name__ == "__main__": 自测入口。
"""

# 代码围栏提取正则：匹配 ```python ... ``` 之类的围栏块（语言标识可选）
_CODE_FENCE_PATTERN = re.compile(r"```[a-zA-Z0-9]*\s*\n(.*?)```", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """
    提取代码围栏内的代码内容

    Args:
        text: 可能包含代码围栏的 LLM 输出文本

    Returns:
        围栏内的代码（去除首尾空白）；无围栏时返回去除首尾空白的原文
    """
    match = _CODE_FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()
