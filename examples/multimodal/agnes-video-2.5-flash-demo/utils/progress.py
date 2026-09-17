"""
Agnes 视频生成进度显示工具
==========================

提供轻量文本进度条（仅依赖标准库），在轮询视频生成任务时按
API 返回的 progress 字段实时展示进度。

作者: Sol
日期: 2026-09-17
"""

import sys


class ProgressBar:
    """基于文本的进度条，支持原地重绘与自定义宽度"""

    def __init__(self, total: int = 100, width: int = 40):
        """
        初始化进度条

        Args:
            total: 进度总量，默认 100（对应 API 的 progress 百分比）
            width: 进度条图形宽度（字符数），默认 40
        """
        self.total = max(total, 1)
        self.width = max(width, 10)
        self._started = False

    def update(self, progress: float) -> None:
        """
        更新并重绘进度条（使用 \\r 原地刷新）

        Args:
            progress: 当前进度值（0 - total）
        """
        # 归一化到 0-100，防止 API 返回异常值导致图形溢出
        percent = max(0.0, min(100.0, float(progress) / self.total * 100.0))
        filled = int(self.width * percent / 100.0)
        bar = "#" * filled + "-" * (self.width - filled)
        # 追加空格避免残留字符；进度 100% 时换行收尾
        suffix = "\n" if percent >= 100.0 else ""
        sys.stdout.write(f"\r[{bar}] {percent:5.1f}%{suffix}")
        sys.stdout.flush()
        self._started = True

    def finish(self) -> None:
        """确保进度条以换行收尾，避免与后续日志粘连"""
        if self._started:
            sys.stdout.write("\n")
            sys.stdout.flush()
