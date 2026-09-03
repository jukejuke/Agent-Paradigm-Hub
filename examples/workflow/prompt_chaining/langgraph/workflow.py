"""
Prompt Chaining - LangGraph 实现
================================

核心思想：将复杂任务拆分成按顺序执行的多个步骤，前一步的输出作为下一步的输入，
依次完成「大纲 → 文章 → 推文」的链式处理。

流程：outline（生成大纲） → article（撰写文章） → tweet（提炼推文）。
"""

import os
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from pathlib import Path
from dotenv import load_dotenv

# 定位项目根目录并加载 .env（支持从任意目录直接运行本文件）
load_dotenv(Path(__file__).resolve().parents[4] / ".env")


class PromptChainingState(TypedDict):
    """Prompt Chaining 图的状态定义"""

    input: str     # 用户初始输入
    outline: str   # 生成的内容大纲
    article: str   # 根据大纲撰写的完整文章
    tweet: str     # 根据文章提炼的推文


# 统一模型：默认 gpt-4o-mini，可用环境变量 OPENAI_MODEL 覆盖
LLM = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0.7)


def outline_node(state: PromptChainingState) -> dict:
    """
    大纲生成节点：LLM 以「大纲生成」角色，根据用户输入生成内容大纲

    Args:
        state: 当前图状态，包含 input 字段

    Returns:
        字典 {"outline": 生成的大纲文本}
    """
    messages = [
        SystemMessage(
            content="你是内容策划专家。请根据用户提供的主题，生成一份结构清晰、"
                    "逻辑连贯的内容大纲，包含若干小节标题与要点。"
        ),
        HumanMessage(content=f"主题：{state['input']}\n\n请生成内容大纲。"),
    ]
    response = LLM.invoke(messages)
    return {"outline": str(response.content)}


def article_node(state: PromptChainingState) -> dict:
    """
    内容撰写节点：LLM 以「内容撰写」角色，根据大纲撰写完整文章

    Args:
        state: 当前图状态，包含 outline 字段

    Returns:
        字典 {"article": 撰写的完整文章文本}
    """
    messages = [
        SystemMessage(
            content="你是资深内容撰稿人。请根据提供的大纲撰写一篇完整、流畅、"
                    "有深度的文章，内容专业且通俗易懂。"
        ),
        HumanMessage(content=f"大纲：\n{state['outline']}\n\n请撰写完整文章。"),
    ]
    response = LLM.invoke(messages)
    return {"article": str(response.content)}


def tweet_node(state: PromptChainingState) -> dict:
    """
    推文提炼节点：LLM 以「推文提炼」角色，根据文章提炼一条推文

    Args:
        state: 当前图状态，包含 article 字段

    Returns:
        字典 {"tweet": 提炼出的推文文本}
    """
    messages = [
        SystemMessage(
            content="你是社交媒体运营专家。请将文章核心观点提炼成一条 280 字以内的"
                    "中文推文，开头要有吸引眼球的钩子。"
        ),
        HumanMessage(content=f"文章：\n{state['article']}\n\n请提炼成推文。"),
    ]
    response = LLM.invoke(messages)
    return {"tweet": str(response.content)}


def build_prompt_chaining_graph():
    """
    构建并编译 Prompt Chaining 图：outline → article → tweet

    Returns:
        编译后的 LangGraph 图对象
    """
    builder = StateGraph(PromptChainingState)

    builder.add_node("outline", outline_node)
    builder.add_node("article", article_node)
    builder.add_node("tweet", tweet_node)

    builder.set_entry_point("outline")
    builder.add_edge("outline", "article")
    builder.add_edge("article", "tweet")
    builder.add_edge("tweet", END)

    return builder.compile()


def run(initial_input: str) -> str:
    """
    运行完整的 Prompt Chaining 流程

    Args:
        initial_input: 用户初始输入主题

    Returns:
        最终提炼出的推文文本
    """
    graph = build_prompt_chaining_graph()
    result = graph.invoke({"input": initial_input})
    return result["tweet"]


def main():
    """演示 Prompt Chaining - LangGraph 实现"""
    final_tweet = run("介绍大语言模型 Agent 的三种协作模式")
    print("=" * 50)
    print(f"最终推文:\n{final_tweet}")


if __name__ == "__main__":
    main()
