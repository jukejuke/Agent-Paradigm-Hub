"""
Routing Workflow - LangChain 实现
==================================

使用 LangChain 实现路由判断和分发。
"""

from typing import Optional

from langchain_openai import ChatOpenAI
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==============================================================================
# LangChain 实现
# ==============================================================================

class LangChainRouting:
    """LangChain 智能路由"""

    ROUTES = {
        "general_qa": ("通用问答", "你是博学助手，通俗易懂。"),
        "code_helper": ("编程助手", "你是资深工程师，给完整代码和解释。"),
        "creative_writing": ("创意写作", "你是创意专家，文笔生动有趣。"),
        "data_analysis": ("数据分析", "你是数据分析师，注重逻辑和数据支撑。"),
    }

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)

    def route(self, request: str) -> str:
        """路由判断"""
        routes_info = "\n".join(f"- {k}: {v[0]}" for k, v in self.ROUTES.items())

        prompt = ChatPromptTemplate.from_messages([
            ("system", "判断请求类别。可选: {info}\n输出 JSON: {{\"route\": \"...\"}}"),
            ("user", "{req}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"info": routes_info, "req": request})

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end])
            return data.get("route", "general_qa")
        except (json.JSONDecodeError, ValueError):
            print(f"Warning: JSON 解析失败，默认 general_qa。原始: {response}")
            return "general_qa"

    def handle(self, route_key: str, request: str) -> str:
        """处理请求"""
        name, sys_prompt = self.ROUTES.get(route_key, self.ROUTES["general_qa"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("user", "{req}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"req": request})

    def run(self, request: str) -> str:
        """完整路由流程"""
        print(f"\n📥 请求: {request}\n{'='*50}")

        route_key = self.route(request)
        name = self.ROUTES.get(route_key, ("unknown",))[0]
        print(f"🔀 路由 → {name}")

        result = self.handle(route_key, request)
        return result


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 LangChain Routing"""
    workflow = LangChainRouting()
    for req in ["用 Python 写快速排序", "帮我想奶茶店 slogan"]:
        print(f"\n{'#'*50}")
        result = workflow.run(req)
        print(f"\n📤 回复:\n{result}")


if __name__ == "__main__":
    main()
