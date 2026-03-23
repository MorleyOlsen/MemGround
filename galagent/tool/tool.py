# tool.py
from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID, uuid4
from typing import AsyncIterator, Any, Dict, Optional, List
from pathlib import Path
import json
import yaml
from galagent.common.openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
)


class ToolNamespaceConfig:
    """工具命名空间配置类"""
    def __init__(self, name, description, tools):
        self.name = name
        self.description = description
        self.tools = tools


class Tool(ABC):
    """
    工具基类，定义了工具的基本接口
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        raise NotImplementedError

    @property
    @abstractmethod
    def instruction(self) -> str:
        """工具使用说明"""
        raise NotImplementedError

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        """工具配置"""
        return ToolNamespaceConfig(
            name=self.name,
            description=self.instruction,
            tools=[]
        )

    def make_response(
        self,
        content: Content,
        *, 
        metadata: Dict[str, Any] | None = None,
        author: Author | None = None,
        channel: str | None = None,
    ) -> Message:
        """创建工具响应消息，把响应结果交给assistant处理"""
        author = Author(role=Role.TOOL, name=self.name)

        message = Message(
            author=author,
            content=[content],
        ).with_recipient("assistant")

        if channel:
            message = message.with_channel(channel)

        return message

    @abstractmethod
    async def _process(self, message: Message) -> AsyncIterator[Message]:
        """处理消息的具体实现"""
        if False:
            yield
        raise NotImplementedError

    async def process(self, message: Message) -> AsyncIterator[Message]:
        """处理消息的公共接口"""
        async for m in self._process(message):
            yield m

    def sync_process(self, message: Message) -> List[Message]:
        """同步处理消息，返回消息列表"""
        import asyncio
        result = asyncio.run(self._sync_process(message))
        return result

    async def _sync_process(self, message: Message) -> List[Message]:
        """内部同步处理实现"""
        messages = []
        async for msg in self._process(message):
            messages.append(msg)
        return messages


class LLMConfigLoaderTool(Tool):
    """
    LLM配置加载工具，直接调用原有load_llm_config功能
    """

    def __init__(self, name: str = "LLMConfigLoader"):
        assert name == "LLMConfigLoader"
        self.name = name

    @classmethod
    def get_tool_name(cls) -> str:
        return "LLMConfigLoader"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @property
    def instruction(self) -> str:
        return """
        使用此工具加载LLM配置。支持从YAML文件加载配置。
        输入格式：{"config_path": "配置文件路径"}
        """

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        """直接调用原有函数加载LLM配置"""
        import json
        
        channel = message.channel
        raw_text = message.content[0].text
        
        try:
            args = json.loads(raw_text)
            config_path = args.get("config_path")
            
            if not config_path:
                yield self.make_response(TextContent(text="Error: Missing 'config_path' parameter"), channel=channel)
                return
            
            # 直接调用原有函数
            from galagent.agent.llm_policy import load_llm_config
            config = load_llm_config(Path(config_path))
            
            # 返回配置结果
            yield self.make_response(
                TextContent(text=json.dumps({
                    "provider": config.provider,
                    "api_key": config.api_key,
                    "base_url": config.base_url,
                    "model": config.model,
                    "temperature": config.temperature,
                    "max_output_tokens": config.max_output_tokens,
                    "goal_instruction": config.goal_instruction
                }, ensure_ascii=False, indent=2)),
                channel=channel
            )
            return
        except Exception as e:
            yield self.make_response(TextContent(text=f"Error: {e}"), channel=channel)
            return


class SceneLoaderTool(Tool):
    """
    场景加载工具，直接调用原有load_nodes功能
    """

    def __init__(self, name: str = "SceneLoader"):
        assert name == "SceneLoader"
        self.name = name

    @classmethod
    def get_tool_name(cls) -> str:
        return "SceneLoader"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def instruction(self) -> str:
        return """
        使用此工具加载场景数据。支持从JSON或YAML文件加载场景。
        输入格式：{"file_path": "场景文件路径"}
        """

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        """直接调用原有函数加载场景数据"""
        import json
        
        channel = message.channel
        raw_text = message.content[0].text
        
        try:
            args = json.loads(raw_text)
            file_path = args.get("file_path")
            
            if not file_path:
                yield self.make_response(TextContent(text="Error: Missing 'file_path' parameter"), channel=channel)
                return
            
            # 直接调用原有函数
            from galagent.env.dataset_loader import load_nodes
            nodes = load_nodes(Path(file_path))
            
            yield self.make_response(
                TextContent(text=json.dumps(nodes, default=lambda o: o.__dict__, ensure_ascii=False, indent=2)),
                channel=channel
            )
            return
        except Exception as e:
            yield self.make_response(TextContent(text=f"Error: {e}"), channel=channel)
            return


class EmbeddingGeneratorTool(Tool):
    """
    向量嵌入生成工具，直接调用原有get_qwen_embedding功能
    """

    def __init__(self, name: str = "EmbeddingGenerator"):
        assert name == "EmbeddingGenerator"
        self.name = name

    @classmethod
    def get_tool_name(cls) -> str:
        return "EmbeddingGenerator"

    @property
    def name(self) -> str:
        return self.get_tool_name()

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def instruction(self) -> str:
        return """
        使用此工具生成文本的向量嵌入。
        输入格式：{"text": "文本内容", "dim": "嵌入维度(可选)"}
        """

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        """直接调用原有函数生成向量嵌入"""
        import json
        
        channel = message.channel
        raw_text = message.content[0].text
        
        try:
            args = json.loads(raw_text)
            text = args.get("text")
            dim = args.get("dim", 1536)
            
            if not text:
                yield self.make_response(TextContent(text="Error: Missing 'text' parameter"), channel=channel)
                return
            
            # 直接调用原有函数
            from galagent.memory.store import get_qwen_embedding
            embedding = get_qwen_embedding(text, dim)
            
            yield self.make_response(
                TextContent(text=json.dumps({
                    "text": text,
                    "embedding": embedding,
                    "dim": len(embedding)
                }, ensure_ascii=False, indent=2)),
                channel=channel
            )
            return
        except Exception as e:
            yield self.make_response(TextContent(text=f"Error: {e}"), channel=channel)
            return

