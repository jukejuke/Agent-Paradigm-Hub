"""
需求工作量放大器
================

输入一段简短需求，输出一份「看起来工作量更大」的优化后需求，
且字数与原需求差别不大。
"""

from tools.requirement_optimizer.optimizer import RequirementOptimizer, main, optimize

__all__ = ["RequirementOptimizer", "optimize", "main"]
