"""
需求工作量放大器 - 核心逻辑
==========================

输入一段简短需求，输出一份「看起来工作量更大」的优化后需求，
且字数与原需求差别不大。

实现方式：单次 LLM 调用 + 专门设计的「工作量放大」系统提示词。
作者: yaosh
日期: 2026-10-09
"""

import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from tools.requirement_optimizer.config import OptimizerConfig
from tools.requirement_optimizer.prompts import AMPLIFY_SYSTEM, COMPRESS_SYSTEM
from utils.llm_client import LLMClient

# 定位仓库根目录并加载 .env（支持从任意目录直接运行本工具）
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# 字数容差上限（原需求的 130%），超过则触发压缩
MAX_LENGTH_RATIO = 1.3
# 压缩最大重试次数，避免无限循环
MAX_COMPRESS_RETRIES = 2
# 限流重试最大次数
MAX_RATE_LIMIT_RETRIES = 3
# 限流重试基础等待秒数（指数退避）
RATE_LIMIT_BACKOFF_BASE = 2.0


def count_chars(text: str) -> int:
    """
    统计文本字符数（去除首尾空白后）

    Args:
        text: 待统计文本

    Returns:
        字符数
    """
    return len(text.strip())


def strip_code_fence(text: str) -> str:
    """
    去掉文本外层可能包裹的 Markdown 代码块围栏

    Args:
        text: 模型原始输出

    Returns:
        清理围栏后的纯文本
    """
    stripped = text.strip()
    if stripped.startswith("```") and "\n" in stripped:
        lines = stripped.split("\n")
        lines = lines[1:]  # 去掉开头的 ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # 去掉结尾的 ```
        return "\n".join(lines).strip()
    return stripped


def _is_rate_limit_error(exc: Exception) -> bool:
    """
    判断异常是否为 API 限流错误（429）

    通过异常类型名或错误信息中是否包含限流关键词来识别，
    兼容 OpenAI 与 Anthropic 等不同提供商。

    Args:
        exc: 捕获到的异常

    Returns:
        True 表示是限流错误
    """
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "rate" in name or "ratelimit" in name or "429" in msg or "rate limit" in msg


def _truncate_to_length(text: str, target: int) -> str:
    """
    将文本截断到目标字数以内，尽量在标点处断句以保持可读性

    Args:
        text: 待截断文本
        target: 目标字数上限

    Returns:
        截断后的文本
    """
    if count_chars(text) <= target:
        return text
    truncated = text[:target].strip()
    # 尽量在最后一个标点处断句，避免半句截断
    for punct in ("。", "！", "？", "；", ".", "!", "?", ";"):
        idx = truncated.rfind(punct)
        if idx >= target // 2:  # 标点至少在前半部分，避免截得过短
            return truncated[: idx + 1].strip()
    return truncated


class RequirementOptimizer:
    """需求工作量放大器

    基于 LLMClient 与「工作量放大」系统提示词，将简短需求改写为
    字数相近但显得更专业、更复杂的需求文本。
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        初始化放大器：创建 LLM 客户端

        Args:
            config: 运行配置，缺省时从环境变量读取
        """
        config = config or OptimizerConfig()
        self.llm = LLMClient(
            provider=config.provider,
            model=config.model,
            temperature=config.temperature,
        )

    def _chat_with_retry(self, messages: list[dict], system_prompt: str) -> str:
        """
        调用 LLM，遇限流（429）时指数退避重试

        Args:
            messages: 消息列表
            system_prompt: 系统提示词

        Returns:
            LLM 回复文本

        Raises:
            非限流异常直接抛出；限流超过最大重试次数后抛出最后一次异常
        """
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return self.llm.chat(messages, system_prompt=system_prompt)
            except Exception as exc:
                last_exc = exc
                if _is_rate_limit_error(exc) and attempt < MAX_RATE_LIMIT_RETRIES:
                    wait = RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
                    print(f"[限流] 触发 API 速率限制，{wait:.0f}s 后重试（第 {attempt + 1} 次）...")
                    time.sleep(wait)
                    continue
                raise
        # 理论上不会走到这里，仅为类型检查兜底
        raise last_exc  # type: ignore[misc]

    def optimize(self, requirement: str) -> str:
        """
        运行工作量放大流程，返回优化后的需求文本

        若放大后字数超过原需求的 130%，会自动触发压缩重试，
        压缩后仍超限则硬性截断，确保输出字数与原需求相近。

        Args:
            requirement: 用户原始需求

        Returns:
            优化后的需求文本（已清理代码围栏）
        """
        messages = [{"role": "user", "content": requirement}]
        result = self._chat_with_retry(messages, system_prompt=AMPLIFY_SYSTEM)
        result = strip_code_fence(result)

        original_len = count_chars(requirement)
        # 字数超限时触发压缩循环
        if original_len > 0 and count_chars(result) > original_len * MAX_LENGTH_RATIO:
            result = self._compress_to_limit(result, original_len)
        return result

    def _compress_to_limit(self, text: str, original_len: int) -> str:
        """
        将超长的优化结果压缩到原需求字数的 130% 以内

        若压缩重试后仍超限，则硬性截断到目标字数。

        Args:
            text: 超长的优化结果文本
            original_len: 原需求字符数

        Returns:
            压缩（或截断）后的文本，字数不超过原需求的 130%
        """
        target = int(original_len * MAX_LENGTH_RATIO)
        current = text
        for _ in range(MAX_COMPRESS_RETRIES):
            if count_chars(current) <= target:
                break
            user_msg = (
                f"目标字数上限：{target} 字\n"
                f"待压缩文本：\n{current}"
            )
            messages = [{"role": "user", "content": user_msg}]
            compressed = self._chat_with_retry(messages, system_prompt=COMPRESS_SYSTEM)
            current = strip_code_fence(compressed)
        # 硬性兜底：压缩后仍超限则截断
        if count_chars(current) > target:
            current = _truncate_to_length(current, target)
        return current


def optimize(requirement: str) -> str:
    """
    便捷函数：使用默认配置放大需求的感知工作量

    Args:
        requirement: 用户原始需求

    Returns:
        优化后的需求文本
    """
    return RequirementOptimizer().optimize(requirement)


def main():
    """
    命令行入口：python -m tools.requirement_optimizer [需求]

    未提供参数时，交互式读取用户需求；仍为空则使用演示需求。
    运行后打印原需求与优化后需求的字数对比。
    """
    requirement = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not requirement:
        requirement = input("请输入你的需求（回车结束）:\n").strip()
    if not requirement:
        requirement = "做一个用户登录功能。"
        print(f"（未输入需求，使用演示需求：{requirement}）")

    optimizer = RequirementOptimizer()
    optimized = optimizer.optimize(requirement)

    original_len = count_chars(requirement)
    optimized_len = count_chars(optimized)
    ratio = optimized_len / original_len if original_len else 0

    print("\n" + "=" * 60)
    print("【原需求】")
    print(requirement)
    print(f"（字数：{original_len}）")
    print("-" * 60)
    print("【优化后需求】")
    print(optimized)
    print(f"（字数：{optimized_len}，原需求的 {ratio:.1%}）")
    print("=" * 60)


if __name__ == "__main__":
    main()
