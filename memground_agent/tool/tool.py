# tool.py
from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID, uuid4
from typing import AsyncIterator, Any, Dict, Optional, List
from pathlib import Path
import json
import yaml
from memground_agent.common.openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
)


class ToolNamespaceConfig:
    """Tool namespace configuration class"""
    def __init__(self, name, description, tools):
        self.name = name
        self.description = description
        self.tools = tools


class Tool(ABC):
    """
    Base class for tools, defines the basic tool interface
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        raise NotImplementedError

    @property
    @abstractmethod
    def instruction(self) -> str:
        """Tool usage instructions"""
        raise NotImplementedError

    @property
    def tool_config(self) -> ToolNamespaceConfig:
        """Tool configuration"""
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
        """Create a tool response message and hand the result to the assistant"""
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
        """Concrete implementation for processing messages"""
        if False:
            yield
        raise NotImplementedError

    async def process(self, message: Message) -> AsyncIterator[Message]:
        """Public interface for processing messages"""
        async for m in self._process(message):
            yield m

    def sync_process(self, message: Message) -> List[Message]:
        """Synchronously process a message, returning a list of messages"""
        import asyncio
        result = asyncio.run(self._sync_process(message))
        return result

    async def _sync_process(self, message: Message) -> List[Message]:
        """Internal synchronous processing implementation"""
        messages = []
        async for msg in self._process(message):
            messages.append(msg)
        return messages


class LLMConfigLoaderTool(Tool):
    """
    LLM configuration loader tool, directly calls the original load_llm_config functionality
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
        Use this tool to load LLM configuration. Supports loading configuration from a YAML file.
        Input format: {"config_path": "path/to/config/file"}
        """

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        """Directly call the original function to load LLM configuration"""
        import json

        channel = message.channel
        raw_text = message.content[0].text

        try:
            args = json.loads(raw_text)
            config_path = args.get("config_path")

            if not config_path:
                yield self.make_response(TextContent(text="Error: Missing 'config_path' parameter"), channel=channel)
                return

            # Directly call the original function
            from memground_agent.agent.llm_policy import load_llm_config
            config = load_llm_config(Path(config_path))

            # Return the configuration result
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
    Scene loader tool, directly calls the original load_nodes functionality
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
        Use this tool to load scene data. Supports loading scenes from JSON or YAML files.
        Input format: {"file_path": "path/to/scene/file"}
        """

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        """Directly call the original function to load scene data"""
        import json

        channel = message.channel
        raw_text = message.content[0].text

        try:
            args = json.loads(raw_text)
            file_path = args.get("file_path")

            if not file_path:
                yield self.make_response(TextContent(text="Error: Missing 'file_path' parameter"), channel=channel)
                return

            # Directly call the original function
            from memground_agent.env.dataset_loader import load_nodes
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
    Vector embedding generation tool, directly calls the original get_qwen_embedding functionality
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
        Use this tool to generate vector embeddings for text.
        Input format: {"text": "text content", "dim": "embedding dimension (optional)"}
        """

    async def _process(self, message: Message) -> AsyncIterator[Message]:
        """Directly call the original function to generate vector embeddings"""
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

            # Directly call the original function
            from memground_agent.memory.store import get_qwen_embedding
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

