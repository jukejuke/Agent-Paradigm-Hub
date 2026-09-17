"""
Agnes Video 2.5 Flash 示例项目 - 配置目录
==========================================

集中导出模型配置与参数校验函数，供 src / utils / examples 复用。

作者: Sol
日期: 2026-09-17
"""

from config.settings import (
    MODEL_CONFIG,
    ModelConfig,
    get_model_config,
    validate_aspect_ratio,
    validate_mode,
    validate_seconds,
    validate_size,
)

__all__ = [
    "MODEL_CONFIG",
    "ModelConfig",
    "get_model_config",
    "validate_mode",
    "validate_seconds",
    "validate_size",
    "validate_aspect_ratio",
]
