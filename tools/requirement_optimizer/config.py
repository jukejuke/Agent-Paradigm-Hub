"""
需求工作量放大器 - 运行配置（Evaluator-Optimizer · LangGraph）
============================================================

从环境变量读取模型、温度、最大迭代次数、通过分数等配置，
并按 tools/ 约定加载仓库根目录 .env。
作者: yaosh
日期: 2026-10-09
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 定位仓库根目录并加载 .env（支持从仓库任意位置运行本工具）
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class OptimizerConfig:
    """需求工作量放大器的配置对象

    支持通过环境变量覆盖默认值：
        - OPENAI_MODEL / OPENAI_API_KEY / OPENAI_BASE_URL
        - REQUIREMENT_OPTIMIZER_TEMPERATURE
        - REQUIREMENT_OPTIMIZER_MAX_ITERATIONS
        - REQUIREMENT_OPTIMIZER_PASS_SCORE
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_iterations: int | None = None,
        pass_score: int | None = None,
    ):
        """
        初始化配置

        Args:
            model: 使用的模型名，默认读取 OPENAI_MODEL（缺省 gpt-4o-mini）
            temperature: 采样温度，默认 0.7
            max_iterations: 评估-优化最大迭代次数，默认 3
            pass_score: 评估通过的最低分数（1-10），默认 7
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("REQUIREMENT_OPTIMIZER_TEMPERATURE", "0.7"))
        )
        self.max_iterations = max_iterations or int(
            os.getenv("REQUIREMENT_OPTIMIZER_MAX_ITERATIONS", "3")
        )
        self.pass_score = pass_score or int(
            os.getenv("REQUIREMENT_OPTIMIZER_PASS_SCORE", "7")
        )
