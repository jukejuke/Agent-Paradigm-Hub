"""
Prompt Chaining Workflow - 原生实现
=====================================

核心思想：将复杂任务分解成多个有顺序的步骤，
每一步的输出作为下一步的输入（上下文的一部分）。

适合场景：需要多步推理/转换的任务，如翻译→润色→发布、
提取→分析→总结、数据清洗→结构化→报告生成。
"""

from typing import Optional

import os
import sys

# 将项目根目录加入 sys.path，使本脚本可在任意工作目录运行
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from utils.llm_client import LLMClient


# ==============================================================================
# Prompt Chaining 实现
# ==============================================================================

class PromptChainingWorkflow:
    """链式提示工作流 - 多步骤顺序处理"""

    def __init__(self, llm: Optional[LLMClient] = None):
        """
        初始化链式提示工作流

        Args:
            llm: LLM 客户端
        """
        self.llm = llm or LLMClient()

    def run(self, initial_input: str, steps: list[dict]) -> str:
        """
        执行链式处理

        Args:
            initial_input: 初始输入
            steps: 步骤列表，每个步骤是 dict:
                {
                    "name": "步骤名称",
                    "system_prompt": "该步骤的系统提示词",
                    "user_template": "用户消息模板，可使用 {input} 引用上一步输出",
                }

        Returns:
            最后一个步骤的输出
        """
        print(f"\n{'='*50}")
        print(f"🔗 Prompt Chaining 开始")
        print(f"初始输入: {initial_input[:80]}...")
        print(f"共 {len(steps)} 个步骤")
        print(f"{'='*50}")

        current_output = initial_input

        for i, step in enumerate(steps):
            step_name = step.get("name", f"Step {i+1}")
            system_prompt = step.get("system_prompt", "你是一个助手。")
            user_template = step.get("user_template", "{input}")

            # 填充模板
            user_content = user_template.format(input=current_output)

            print(f"\n--- Step {i+1}: {step_name} ---")
            print(f"  输入: {current_output[:60]}...")

            messages = [{"role": "user", "content": user_content}]
            current_output = self.llm.chat(messages, system_prompt=system_prompt)

            print(f"  输出: {current_output[:60]}...")

        return current_output


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 Prompt Chaining 的使用"""
    workflow = PromptChainingWorkflow()

    # 示例：从原始想法 → 结构化大纲 → 精简摘要 → 推文
    initial_idea = "我想写一篇关于 AI Agent 发展历程的博客文章"

    steps = [
        {
            "name": "结构化大纲",
            "system_prompt": "你是一个文章策划。将用户想法扩展为 5 点文章大纲，每点不超过 20 字。",
            "user_template": "请为以下想法制定大纲:\n{input}",
        },
        {
            "name": "内容润色",
            "system_prompt": "你是一个资深编辑。将大纲扩展成详细的文章段落（每段 50-100 字），语言专业但易懂。",
            "user_template": "请将以下大纲扩展成文章初稿:\n{input}",
        },
        {
            "name": "生成推文",
            "system_prompt": "你是社交媒体专家。将文章核心观点提炼成一条 280 字以内的中文推文，开头要有吸引眼球的钩子。",
            "user_template": "请将以下文章提炼成推文:\n{input}",
        },
    ]

    final = workflow.run(initial_idea, steps)
    print(f"\n{'='*50}")
    print(f"📤 最终推文:\n{final}")


if __name__ == "__main__":
    main()
