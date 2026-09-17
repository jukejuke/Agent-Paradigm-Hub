"""
Agnes 视频模块异常体系与重试工具
================================

定义统一的异常层级（便于上层统一 try/except）与带指数退避的重试装饰器，
对限流（429）与临时服务端错误（5xx）自动重试，提升调用稳定性。

作者: yaosh
日期: 2026-09-17
"""

import functools
import time

import requests


class AgnesError(Exception):
    """Agnes 视频模块统一异常基类，上层只需捕获本类即可覆盖所有业务异常"""


class AgnesConfigError(AgnesError):
    """配置错误：API Key 缺失、帧数 / 帧率 / 分辨率档位非法等"""


class AgnesVideoError(AgnesError):
    """视频处理错误：本地图片不存在、任务失败、下载失败等"""


class AgnesAPIError(AgnesError):
    """API 调用错误：网络异常或服务端返回错误状态码"""


def _is_retryable_status(status_code: int) -> bool:
    """判断状态码是否可重试：429 限流与 5xx 服务端临时错误"""
    return status_code == 429 or 500 <= status_code < 600


def _parse_response(resp: requests.Response) -> dict:
    """
    将 HTTP 响应解析为 JSON 字典，解析失败时抛出中文 API 错误

    Args:
        resp: requests 响应对象

    Returns:
        JSON 字典

    Raises:
        AgnesAPIError: 响应体不是合法 JSON
    """
    try:
        return resp.json()
    except ValueError as e:
        raise AgnesAPIError(
            f"Agnes 视频 API 返回了无法解析的内容（HTTP {resp.status_code}）：{resp.text[:200]}"
        ) from e


def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    """
    指数退避重试装饰器：对 429 限流与 5xx 服务端错误自动重试

    说明：
    - 被装饰函数须返回 requests.Response，由本装饰器统一校验状态码
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
                    resp = func(*args, **kwargs)
                    # 2xx 视为成功，返回响应对象
                    if 200 <= resp.status_code < 300:
                        return resp
                    # 达到重试上限或错误不可重试时，包装为业务异常抛出
                    if attempt >= max_retries or not _is_retryable_status(resp.status_code):
                        raise AgnesAPIError(
                            f"调用 Agnes 视频 API 失败（HTTP {resp.status_code}）："
                            f"{resp.text[:300]}"
                        )
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                    attempt += 1
                except requests.RequestException as e:
                    raise AgnesAPIError(
                        f"调用 Agnes 视频 API 发生网络错误：{e}"
                    ) from e
        return wrapper
    return decorator
