"""
Agnes 模块异常体系与重试工具
============================

定义统一的异常层级（便于上层统一 try/except）与带指数退避的重试装饰器，
对限流（429）与临时服务端错误（5xx）自动重试，提升调用稳定性。

作者: Sol
日期: 2026-09-16
"""

import functools
import time

from openai import APIError, APIStatusError


class AgnesError(Exception):
    """Agnes 模块统一异常基类，上层只需捕获本类即可覆盖所有业务异常"""


class AgnesConfigError(AgnesError):
    """配置错误：API Key 缺失、分辨率 / 宽高比不合法等"""


class AgnesImageError(AgnesError):
    """图片处理错误：本地文件不存在、下载失败等"""


class AgnesAPIError(AgnesError):
    """API 调用错误：网络异常或服务端返回错误状态码"""


def _is_retryable_status(status_code: int) -> bool:
    """判断状态码是否可重试：429 限流与 5xx 服务端临时错误"""
    return status_code == 429 or 500 <= status_code < 600


def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    """
    指数退避重试装饰器：对 429 限流与 5xx 服务端错误自动重试

    说明：
    - 每次重试前等待 base_delay * 2 ** attempt 秒（指数退避）
    - 非可重试错误（4xx 业务错误、鉴权失败等）直接包装为 AgnesAPIError 抛出
    - 网络层错误（超时 / 连接失败）不重试，直接抛出，由调用方决定策略

    Args:
        max_retries: 最大重试次数（不含首次调用），默认 3
        base_delay: 首次退避基数（秒），默认 1.0

    Returns:
        装饰器
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except APIStatusError as e:
                    # 达到重试上限或错误不可重试时，包装为业务异常抛出
                    if attempt >= max_retries or not _is_retryable_status(e.status_code):
                        raise AgnesAPIError(
                            f"调用 Agnes API 失败（HTTP {e.status_code}）：{e}"
                        ) from e
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    attempt += 1
                except APIError as e:
                    raise AgnesAPIError(f"调用 Agnes API 发生网络错误：{e}") from e
        return wrapper
    return decorator
