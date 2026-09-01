"""
统一的 LLM 调用客户端
支持 OpenAI 和 Anthropic 两种提供商，提供一致的调用接口
"""

import os
from typing import Optional

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class LLMClient:
    """统一的 LLM 客户端封装类"""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        初始化 LLM 客户端

        Args:
            provider: LLM 提供商，支持 'openai' 和 'anthropic'，默认读取环境变量
            model: 使用的模型名称，默认读取环境变量
            temperature: 温度参数，默认 0.7
            max_tokens: 最大输出 token 数，默认 4096
        """
        self.provider = provider or os.getenv("DEFAULT_PROVIDER", "openai")
        self.temperature = temperature or float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
        self.max_tokens = max_tokens or int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))

        # 根据提供商设置模型和客户端
        if self.provider == "openai":
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self._init_openai()
        elif self.provider == "anthropic":
            self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
            self._init_anthropic()
        else:
            raise ValueError(f"不支持的 LLM 提供商: {self.provider}")

    def _init_openai(self) -> None:
        """初始化 OpenAI 客户端"""
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("未找到 OPENAI_API_KEY 环境变量")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _init_anthropic(self) -> None:
        """初始化 Anthropic 客户端"""
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("未找到 ANTHROPIC_API_KEY 环境变量")
        self._client = Anthropic(api_key=api_key)

    def chat(self, messages: list[dict], system_prompt: Optional[str] = None) -> str:
        """
        发送聊天请求并返回回复文本

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            system_prompt: 可选的系统提示词

        Returns:
            LLM 的文本回复
        """
        if self.provider == "openai":
            return self._chat_openai(messages, system_prompt)
        else:
            return self._chat_anthropic(messages, system_prompt)

    def _chat_openai(self, messages: list[dict], system_prompt: Optional[str]) -> str:
        """调用 OpenAI API"""
        # 构建消息列表
        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _chat_anthropic(self, messages: list[dict], system_prompt: Optional[str]) -> str:
        """调用 Anthropic API"""
        # Anthropic 的 system prompt 是独立参数
        response = self._client.messages.create(
            model=self.model,
            messages=messages,
            system=system_prompt or "",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.content[0].text or ""
