"""
需求工作量放大器 - 运行配置
==========================

从环境变量读取模型、温度等配置，并按 tools/ 约定加载仓库根目录 .env。
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
        - DEFAULT_PROVIDER（openai / anthropic）
        - DEFAULT_TEMPERATURE
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ):
        """
        初始化配置

        Args:
            provider: LLM 提供商，默认读取 DEFAULT_PROVIDER（缺省 openai）
            model: 使用的模型名，默认读取 OPENAI_MODEL（缺省 gpt-4o-mini）
            temperature: 温度参数，默认 0.7
        """
        self.provider = provider or os.getenv("DEFAULT_PROVIDER", "openai")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
        )
