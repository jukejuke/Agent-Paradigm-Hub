"""
提示词优化 Agent 运行配置
========================

从环境变量读取模型、温度、最大迭代次数等配置。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 定位仓库根目录并加载 .env（支持从仓库任意位置运行本工具）
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class OptimizerConfig:
    """提示词优化 Agent 的配置对象
    作者: Sol
    日期: 2026-09-04

    支持通过环境变量覆盖默认值：
        - OPENAI_MODEL / OPENAI_API_KEY / OPENAI_BASE_URL
        - PROMPT_OPTIMIZER_TEMPERATURE
        - PROMPT_OPTIMIZER_MAX_ITERATIONS
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_iterations: int | None = None,
    ):
        """
        初始化配置

        Args:
            model: 使用的模型名，默认读取 OPENAI_MODEL（缺省 gpt-4o-mini）
            temperature: 温度参数，默认 0.7
            max_iterations: 反思-改进最大迭代次数，默认 3
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("PROMPT_OPTIMIZER_TEMPERATURE", "0.7"))
        )
        self.max_iterations = max_iterations or int(
            os.getenv("PROMPT_OPTIMIZER_MAX_ITERATIONS", "3")
        )
