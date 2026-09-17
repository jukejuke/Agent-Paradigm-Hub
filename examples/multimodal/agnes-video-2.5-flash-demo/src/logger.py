"""
Agnes 视频示例日志模块
======================

统一配置 Python logging：控制台输出 + 可选文件日志，
保证各模块与入口脚本使用一致的日志格式。

作者: yaosh
日期: 2026-09-17
"""

import logging
import os
import sys

# 日志格式：时间 | 级别 | 消息
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
# 时间戳格式
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "agnes-video",
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """
    初始化并返回全局日志器（重复调用不会重复添加处理器）

    Args:
        name: 日志器名称，默认 "agnes-video"
        level: 日志级别，默认 INFO
        log_file: 可选日志文件路径；为 None 时仅输出到控制台

    Returns:
        配置完成的 logging.Logger 实例
    """
    logger = logging.getLogger(name)
    # 已初始化则直接复用，避免重复添加 Handler 造成重复输出
    if getattr(logger, "_agnes_configured", False):
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    # 控制台输出（UTF-8 安全，适配中文日志）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 可选文件日志
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # 标记已配置，避免重复初始化
    logger._agnes_configured = True  # type: ignore[attr-defined]
    return logger
