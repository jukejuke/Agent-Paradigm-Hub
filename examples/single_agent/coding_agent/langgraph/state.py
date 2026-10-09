"""
Coding Agent 智能体状态管理模块
================================
定义图（Graph）共享的状态模型 CodingAgentState，用于跟踪代码生成过程中的上下文信息：
任务需求、当前代码、执行输出/错误、评审反馈、通过标记、迭代次数与历史记录。
"""
import operator
from typing import Annotated, TypedDict


class CodingAgentState(TypedDict):
    """Coding Agent 图的状态定义，跟踪代码生成过程的上下文信息

    作者: yaosh 创建时间: 2026-09-23
    """

    task: str                       # 用户的任务需求（原始描述）
    code: str                       # 当前生成/修复后的代码
    execution_output: str           # 最近一次沙箱执行的标准输出（stdout）
    execution_error: str            # 最近一次沙箱执行的错误信息（空字符串表示执行正常）
    feedback: str                   # 评估环节给出的评审意见
    passed: bool                    # 是否通过验收（规则 + LLM 评审综合判定）
    iteration: int                  # 当前已执行的修复轮次
    max_iterations: int             # 最大允许的修复轮次（防止无限循环）
    target_file: str                # 指定要修改的目标文件路径（空字符串表示从零生成）
    output_path: str                # 最终代码保存路径
    history: Annotated[list[dict], operator.add]  # 迭代历史记录（归约器追加，用于上下文跟踪）
