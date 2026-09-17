"""
Agnes 视频 API 客户端
=====================

创建 OpenAI 兼容的 HTTP 会话，并提供视频异步任务 API 的封装：
创建任务（POST /videos）、查询结果（GET /agnesapi?video_id= 或
GET /videos/<task_id>）与轮询等待。

作者: Sol
日期: 2026-09-17
"""

import logging
import os
import time

import requests

from config.settings import MODEL_CONFIG, get_model_config
from utils.errors import (
    AgnesAPIError,
    AgnesConfigError,
    AgnesVideoError,
    _parse_response,
    with_retry,
)

logger = logging.getLogger("agnes-video")


def create_client(api_key: str | None = None, base_url: str | None = None) -> requests.Session:
    """
    创建带 Bearer 认证的 HTTP 会话

    Args:
        api_key: Agnes API Key；为 None 时从环境变量 AGNES_API_KEY 读取
        base_url: API 地址；为 None 时优先取环境变量 AGNES_BASE_URL，再缺省用默认值

    Returns:
        配置完成的 requests.Session 实例

    Raises:
        AgnesConfigError: 未配置 API Key 时
    """
    config = get_model_config()
    key = api_key or os.getenv(config.api_key_env)
    if not key:
        raise AgnesConfigError(
            f"未找到 {config.api_key_env} 环境变量，"
            "请将项目根目录 .env.example 复制为 .env，填入 Agnes API Key 后重试"
        )
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
    )
    return session


@with_retry()
def _post_videos(client: requests.Session, payload: dict) -> requests.Response:
    """发送创建视频任务的 HTTP 请求（POST /videos），由装饰器统一重试与校验状态码"""
    config = get_model_config()
    return client.post(f"{config.base_url}/videos", json=payload)


def create_video_task(client: requests.Session, payload: dict) -> dict:
    """
    创建视频生成任务（POST /videos）

    Args:
        client: create_client() 返回的会话
        payload: 请求体，含 model / prompt / mode / seconds / size /
                 aspect_ratio 等字段

    Returns:
        任务信息字典，含 task_id / video_id / status / progress

    Raises:
        AgnesAPIError: API 调用失败（429/5xx 已自动指数退避重试）
    """
    resp = _post_videos(client, payload)
    data = _parse_response(resp)
    task_id = data.get("task_id") or data.get("id")
    video_id = data.get("video_id")
    if not task_id:
        raise AgnesAPIError(
            f"创建视频任务失败：响应中未包含 task_id，返回：{list(data.keys())}"
        )
    logger.info("已创建视频任务：task_id=%s video_id=%s status=%s",
                task_id, video_id, data.get("status"))
    return data


@with_retry()
def _get_result_recommended(client: requests.Session, video_id: str) -> requests.Response:
    """按推荐方式查询任务结果：GET /agnesapi?video_id=<VIDEO_ID>&model_name=<模型ID>

    官方文档建议所有模式都携带 model_name；不带 model_name 仅对 text 模式有效。
    """
    config = get_model_config()
    return client.get(
        f"{config.base_url}/agnesapi",
        params={"video_id": video_id, "model_name": config.model_id},
    )


@with_retry()
def _get_result_legacy(client: requests.Session, task_id: str) -> requests.Response:
    """按兼容旧版方式查询任务结果：GET /videos/<TASK_ID>"""
    config = get_model_config()
    return client.get(f"{config.base_url}/videos/{task_id}")


def get_video_result(client: requests.Session, task_id: str, video_id: str | None = None) -> dict:
    """
    查询视频生成任务结果：优先推荐接口，失败时回退兼容旧版接口

    Args:
        client: create_client() 返回的会话
        task_id: 任务 ID
        video_id: 视频 ID（推荐接口需要）

    Returns:
        任务结果字典，含 status / progress 及完成后视频 URL

    Raises:
        AgnesAPIError: 两种查询方式均失败
    """
    if video_id:
        try:
            resp = _get_result_recommended(client, video_id)
            if 200 <= resp.status_code < 300:
                return _parse_response(resp)
        except AgnesAPIError as e:
            logger.warning("推荐接口查询失败，回退兼容接口：%s", e)
    resp = _get_result_legacy(client, task_id)
    return _parse_response(resp)


def wait_for_video(
    client: requests.Session,
    task_id: str,
    video_id: str | None = None,
    poll_interval: int = MODEL_CONFIG.poll_interval,
    timeout: int = MODEL_CONFIG.timeout,
    on_progress=None,
) -> dict:
    """
    轮询等待视频生成完成

    说明：
    - 任务状态流转：queued（排队）→ in_progress（生成中）→ completed（完成）
    - 状态为 failed / canceled 时抛出中文异常
    - 通过 on_progress 回调把 progress 字段透传给进度条
    - 支持 Ctrl+C 安全退出（KeyboardInterrupt 转为异常抛出）

    Args:
        client: create_client() 返回的会话
        task_id: 任务 ID
        video_id: 视频 ID
        poll_interval: 轮询间隔（秒），默认取配置 5 秒
        timeout: 超时上限（秒），默认 1800 秒
        on_progress: 可选回调，入参为当前进度数值（0-100）

    Returns:
        最终任务结果字典（status=completed）

    Raises:
        AgnesVideoError: 任务失败、超时或用户中断
    """
    elapsed = 0
    try:
        while elapsed < timeout:
            data = get_video_result(client, task_id, video_id)
            status = (data.get("status") or "queued").lower()
            progress = data.get("progress", 0)

            if status == "completed":
                if on_progress:
                    on_progress(100)
                logger.info("视频生成完成（耗时约 %d 秒）", elapsed)
                return data
            if status in ("failed", "canceled", "cancelled"):
                raise AgnesVideoError(f"视频生成任务已{status}：{data}")

            # 生成中 / 排队中：上报进度并等待下一轮
            if on_progress:
                on_progress(progress if isinstance(progress, (int, float)) else 0)
            time.sleep(poll_interval)
            elapsed += poll_interval

        raise AgnesVideoError(
            f"视频生成任务超时（超过 {timeout} 秒仍未完成），task_id={task_id}"
        )
    except KeyboardInterrupt:
        raise AgnesVideoError(f"用户中断视频生成任务，task_id={task_id}") from None
