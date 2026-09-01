"""
Routing Workflow - 原生实现
============================

核心思想：先让 LLM 判断用户请求属于哪一类（路由），
然后将请求分发给对应的专业处理器处理。

适合场景：客服系统（问题分类）、多模型路由（简单问题用小模型，复杂问题用大模型）、
意图识别和分发等。
"""

import json
from typing import Optional

from utils.llm_client import LLMClient


# ==============================================================================
# Routing 实现
# ==============================================================================

class RoutingWorkflow:
    """智能路由工作流 - 根据请求类型分发处理"""

    # 定义可用的路由和对应的处理器
    ROUTES = {
        "general_qa": {
            "name": "通用问答",
            "description": "一般性的问答、解释、建议",
            "handler_prompt": "你是一个博学的助手，请用通俗易懂的方式回答问题。",
        },
        "code_helper": {
            "name": "编程助手",
            "description": "代码编写、调试、重构、技术问题",
            "handler_prompt": "你是一个资深工程师，擅长代码编写和调试。给出完整可运行的代码和解释。",
        },
        "creative_writing": {
            "name": "创意写作",
            "description": "文案创作、故事生成、营销内容",
            "handler_prompt": "你是一个创意写作专家，文笔生动有趣，善于打动读者。",
        },
        "data_analysis": {
            "name": "数据分析",
            "description": "数据分析、统计问题、数字处理",
            "handler_prompt": "你是一个数据分析师，注重逻辑和数据支撑，给出清晰的分析思路。",
        },
    }

    ROUTING_SYSTEM_PROMPT = """你是一个请求路由器。判断用户请求属于哪个类别。

可选类别:
{routes_info}

输出 JSON 格式:
{{"route": "类别key", "reason": "选择理由"}}

只输出 JSON，不要其他内容。"""

    def __init__(self, llm: Optional[LLMClient] = None):
        """
        初始化路由工作流

        Args:
            llm: LLM 客户端
        """
        self.llm = llm or LLMClient()

    def _build_routes_info(self) -> str:
        """构建路由信息文本"""
        return "\n".join(
            f"- {key}: {info['description']}"
            for key, info in self.ROUTES.items()
        )

    def route(self, request: str) -> tuple[str, str]:
        """
        判断请求应该路由到哪个处理器

        Args:
            request: 用户请求

        Returns:
            (路由key, 理由)
        """
        system_prompt = self.ROUTING_SYSTEM_PROMPT.format(
            routes_info=self._build_routes_info()
        )

        messages = [{"role": "user", "content": f"请路由以下请求:\n{request}"}]
        response = self.llm.chat(messages, system_prompt=system_prompt)

        # 解析 JSON
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end])
            return data.get("route", "general_qa"), data.get("reason", "")
        except json.JSONDecodeError:
            print(f"Warning: 路由解析失败，默认走 general_qa。原始回复: {response}")
            return "general_qa", "解析失败，使用默认路由"

    def handle(self, route_key: str, request: str) -> str:
        """
        使用指定处理器处理请求

        Args:
            route_key: 路由 key
            request: 用户请求

        Returns:
            处理结果
        """
        if route_key not in self.ROUTES:
            route_key = "general_qa"

        route_info = self.ROUTES[route_key]
        messages = [{"role": "user", "content": request}]

        return self.llm.chat(messages, system_prompt=route_info["handler_prompt"])

    def run(self, request: str) -> str:
        """
        运行完整路由工作流

        Args:
            request: 用户请求

        Returns:
            处理结果
        """
        print(f"\n{'='*50}")
        print(f"📥 收到请求: {request}")
        print(f"{'='*50}")

        # 1. 路由判断
        print("\n🔀 阶段 1: 路由判断")
        route_key, reason = self.route(request)
        route_info = self.ROUTES.get(route_key, {})
        print(f"  → 路由到: {route_info.get('name', route_key)}")
        print(f"  → 理由: {reason}")

        # 2. 分发处理
        print(f"\n⚙️ 阶段 2: 使用 {route_info.get('name')} 处理")
        result = self.handle(route_key, request)

        return result


# ==============================================================================
# 入口函数
# ==============================================================================

def main():
    """演示 Routing 的使用"""
    workflow = RoutingWorkflow()

    # 测试不同类型的请求
    test_requests = [
        "用 Python 写一个快速排序算法",
        "帮我想一个奶茶店的 slogan",
        "请解释一下什么是 Transformer 架构",
    ]

    for req in test_requests:
        print(f"\n{'#'*50}")
        result = workflow.run(req)
        print(f"\n📤 最终回复:\n{result}")


if __name__ == "__main__":
    main()
