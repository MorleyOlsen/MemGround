# galagent/memory/amem.py
"""A-mem 本地记忆管理Agent"""
from typing import Dict, List, Any, Optional
import os
from datetime import datetime

from .base_mem_agent import BaseMemAgent
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'A-mem'))
from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote


class AMemAgent(BaseMemAgent):
    """A-mem 记忆管理代理

    使用 A-mem 的 AgenticMemorySystem 进行记忆的存储、检索和管理
    """

    def __init__(
        self,
        game_name: str = "game_agent",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        verbose: bool = False,
        api_key: Optional[str] = None,
        evo_threshold: int = 100
    ):
        """初始化 A-mem Agent

        Args:
            game_name: 游戏名称，用作数据隔离标识
            embedding_model: 嵌入模型名称
            llm_backend: LLM后端 (openai 或 ollama)
            llm_model: LLM模型名称
            verbose: 是否输出详细日志
            api_key: LLM API key
            evo_threshold: 记忆演化阈值
        """
        super().__init__(game_name, verbose)

        self.embedding_model = embedding_model
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.api_key = api_key
        self.evo_threshold = evo_threshold

        if self.verbose:
            print(f"[A-mem] 初始化 AgenticMemorySystem，游戏: {game_name}")

        # 初始化 AgenticMemorySystem
        try:
            self.memory_system = AgenticMemorySystem(
                model_name=embedding_model,
                llm_backend=llm_backend,
                llm_model=llm_model,
                evo_threshold=evo_threshold,
                api_key=api_key
            )

            if self.verbose:
                print(f"[A-mem] AgenticMemorySystem 初始化成功")

        except ImportError as e:
            raise ImportError(f"请安装 A-mem 依赖: {e}")
        except Exception as e:
            raise RuntimeError(f"初始化 AgenticMemorySystem 失败: {e}")

    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """添加记忆到本地存储

        Args:
            content: 记忆内容
            metadata: 元数据（如 role, step, source, category, tags 等）

        Returns:
            包含 success 和 memory_id 的字典
        """
        try:
            # 准备参数
            kwargs = {}
            extra_tags = []  # 用于存储额外的 metadata 字段

            if metadata:
                # 映射 metadata 到 AgenticMemorySystem 的参数
                if "category" in metadata:
                    kwargs["category"] = metadata["category"]
                if "context" in metadata:
                    kwargs["context"] = metadata["context"]
                if "keywords" in metadata:
                    kwargs["keywords"] = metadata["keywords"]

                # 处理 tags：如果 metadata 中有 tags，保留它
                if "tags" in metadata:
                    if isinstance(metadata["tags"], list):
                        extra_tags.extend(metadata["tags"])
                    else:
                        extra_tags.append(str(metadata["tags"]))

                # 将其他字段（如 role, step, node_id, name 等）存储为 tags
                # 格式：["role:user", "step:1", "node_id:xxx"]
                reserved_fields = {"category", "tags", "context", "keywords"}
                for key, value in metadata.items():
                    if key not in reserved_fields:
                        extra_tags.append(f"{key}:{value}")

            # 设置 tags
            if extra_tags:
                kwargs["tags"] = extra_tags

            # 生成时间戳 (YYYYMMDDHHmm 格式)
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            kwargs["timestamp"] = timestamp

            # 调用 AgenticMemorySystem.add_note
            memory_id = self.memory_system.add_note(
                content=content,
                time=timestamp,
                **kwargs
            )

            if self.verbose:
                print(f"[A-mem] 添加记忆成功: {content[:50]}..., ID: {memory_id}")

            return {"success": True, "memory_id": memory_id}

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] 添加记忆失败: {e}")
            return {"success": False, "error": str(e)}

    def search_memories(
        self,
        query: str,
        top_k: int = 3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索记忆

        Args:
            query: 搜索查询
            top_k: 返回结果数量
            filters: 过滤条件（暂未使用，保留接口兼容性）

        Returns:
            记忆列表，每个记忆包含 id, text, score, metadata 等字段
        """
        try:
            # 使用 AgenticMemorySystem.search_agentic 进行搜索
            search_results = self.memory_system.search_agentic(query, k=top_k)

            if self.verbose:
                print(f"[A-mem] 搜索查询: {query}, 找到 {len(search_results)} 条结果")

            # 格式化结果，转换为统一的格式
            formatted_results = []
            for result in search_results:
                # 解析 tags 中的额外字段
                tags = result.get("tags", [])
                parsed_metadata = {}
                remaining_tags = []

                for tag in tags:
                    if isinstance(tag, str) and ":" in tag:
                        # 解析 "key:value" 格式的 tag
                        key, value = tag.split(":", 1)
                        parsed_metadata[key] = value
                    else:
                        # 保留非键值对格式的 tag
                        remaining_tags.append(tag)

                # 构建 metadata
                metadata = {
                    "context": result.get("context", ""),
                    "keywords": result.get("keywords", []),
                    "tags": remaining_tags,  # 只保留非键值对的 tags
                    "category": result.get("category", "Uncategorized"),
                    "timestamp": result.get("timestamp", ""),
                    "is_neighbor": result.get("is_neighbor", False),
                    **parsed_metadata  # 添加从 tags 解析出的额外字段（如 role, step 等）
                }

                formatted_results.append({
                    "id": result.get("id", ""),
                    "text": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "metadata": metadata,
                    "created_at": result.get("timestamp", ""),
                    "updated_at": result.get("timestamp", "")
                })

            return formatted_results

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] 搜索记忆失败: {e}")
            return []

    def get_all_memories(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """获取所有记忆

        Args:
            filters: 可选的过滤条件

        Returns:
            记忆列表
        """
        try:
            # 从 AgenticMemorySystem 获取所有记忆
            all_memories = list(self.memory_system.memories.values())

            # 格式化结果
            formatted_results = []
            for memory in all_memories:
                # 解析 tags 中的额外字段
                tags = memory.tags
                parsed_metadata = {}
                remaining_tags = []

                for tag in tags:
                    if isinstance(tag, str) and ":" in tag:
                        # 解析 "key:value" 格式的 tag
                        key, value = tag.split(":", 1)
                        parsed_metadata[key] = value
                    else:
                        # 保留非键值对格式的 tag
                        remaining_tags.append(tag)

                # 构建完整结果
                result = {
                    "id": memory.id,
                    "text": memory.content,
                    "metadata": {
                        "context": memory.context,
                        "keywords": memory.keywords,
                        "tags": remaining_tags,  # 只保留非键值对的 tags
                        "category": memory.category,
                        "timestamp": memory.timestamp,
                        "last_accessed": memory.last_accessed,
                        "retrieval_count": memory.retrieval_count,
                        "links": memory.links,
                        **parsed_metadata  # 添加从 tags 解析出的额外字段（如 role, step 等）
                    },
                    "created_at": memory.timestamp,
                    "updated_at": memory.last_accessed
                }

                # 应用过滤器
                if filters:
                    match = True
                    for key, value in filters.items():
                        if key not in result["metadata"] or result["metadata"][key] != value:
                            match = False
                            break
                    if match:
                        formatted_results.append(result)
                else:
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] 获取所有记忆失败: {e}")
            return []

    def update_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """更新记忆

        Args:
            memory_id: 记忆ID
            content: 新的内容
            metadata: 新的元数据

        Returns:
            包含 success 的字典
        """
        try:
            # 准备更新参数
            kwargs = {"content": content}
            extra_tags = []

            if metadata:
                # 映射 metadata 到 MemoryNote 的属性
                if "category" in metadata:
                    kwargs["category"] = metadata["category"]
                if "context" in metadata:
                    kwargs["context"] = metadata["context"]
                if "keywords" in metadata:
                    kwargs["keywords"] = metadata["keywords"]

                # 处理 tags：如果 metadata 中有 tags，保留它
                if "tags" in metadata:
                    if isinstance(metadata["tags"], list):
                        extra_tags.extend(metadata["tags"])
                    else:
                        extra_tags.append(str(metadata["tags"]))

                # 将其他字段（如 role, step, node_id, name 等）存储为 tags
                reserved_fields = {"category", "tags", "context", "keywords"}
                for key, value in metadata.items():
                    if key not in reserved_fields:
                        extra_tags.append(f"{key}:{value}")

            # 设置 tags
            if extra_tags:
                kwargs["tags"] = extra_tags

            # 调用 AgenticMemorySystem.update
            success = self.memory_system.update(memory_id, **kwargs)

            if success:
                if self.verbose:
                    print(f"[A-mem] 更新记忆成功: {memory_id}")
                return {"success": True}
            else:
                return {"success": False, "error": "Memory not found"}

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] 更新记忆失败: {e}")
            return {"success": False, "error": str(e)}

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """删除单个记忆

        Args:
            memory_id: 记忆ID

        Returns:
            包含 success 的字典
        """
        try:
            # 调用 AgenticMemorySystem.delete
            success = self.memory_system.delete(memory_id)

            if success:
                if self.verbose:
                    print(f"[A-mem] 删除记忆成功: {memory_id}")
                return {"success": True}
            else:
                return {"success": False, "error": "Memory not found"}

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] 删除记忆失败: {e}")
            return {"success": False, "error": str(e)}

    def delete_all_memories(self) -> Dict[str, Any]:
        """删除该游戏的所有记忆

        Returns:
            包含 success 的字典
        """
        try:
            # 获取所有记忆ID
            memory_ids = list(self.memory_system.memories.keys())

            # 删除所有记忆
            for memory_id in memory_ids:
                self.memory_system.delete(memory_id)

            if self.verbose:
                print(f"[A-mem] 删除所有记忆成功，共删除 {len(memory_ids)} 条")

            return {"success": True}

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] 删除所有记忆失败: {e}")
            return {"success": False, "error": str(e)}

    def get_state(self) -> Dict[str, Any]:
        """获取当前记忆状态用于持久化

        Returns:
            包含所有记忆数据的状态字典
        """
        try:
            # 序列化所有记忆
            memories_state = []
            for _, memory in self.memory_system.memories.items():
                memories_state.append({
                    "id": memory.id,
                    "content": memory.content,
                    "keywords": memory.keywords,
                    "links": memory.links,
                    "retrieval_count": memory.retrieval_count,
                    "timestamp": memory.timestamp,
                    "last_accessed": memory.last_accessed,
                    "context": memory.context,
                    "evolution_history": memory.evolution_history,
                    "category": memory.category,
                    "tags": memory.tags
                })

            return {
                "game_name": self.game_name,
                "embedding_model": self.embedding_model,
                "llm_backend": self.llm_backend,
                "llm_model": self.llm_model,
                "evo_threshold": self.evo_threshold,
                "evo_cnt": self.memory_system.evo_cnt,
                "memories": memories_state,
                "total_memories": len(memories_state)
            }

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] 获取状态失败: {e}")
            return {}

    def restore_state(self, state: Dict[str, Any]) -> None:
        """从持久化状态恢复记忆

        Args:
            state: 状态字典（来自 get_state）
        """
        try:
            # 恢复基本配置
            self.game_name = state.get("game_name", self.game_name)
            self.embedding_model = state.get("embedding_model", self.embedding_model)
            self.llm_backend = state.get("llm_backend", self.llm_backend)
            self.llm_model = state.get("llm_model", self.llm_model)
            self.evo_threshold = state.get("evo_threshold", self.evo_threshold)

            # 恢复演化计数
            self.memory_system.evo_cnt = state.get("evo_cnt", 0)

            # 恢复所有记忆
            memories_state = state.get("memories", [])
            for mem_data in memories_state:
                # 重建 MemoryNote 对象
                memory_note = MemoryNote(
                    content=mem_data["content"],
                    id=mem_data["id"],
                    keywords=mem_data.get("keywords", []),
                    links=mem_data.get("links", []),
                    retrieval_count=mem_data.get("retrieval_count", 0),
                    timestamp=mem_data.get("timestamp"),
                    last_accessed=mem_data.get("last_accessed"),
                    context=mem_data.get("context", "General"),
                    evolution_history=mem_data.get("evolution_history", []),
                    category=mem_data.get("category", "Uncategorized"),
                    tags=mem_data.get("tags", [])
                )

                # 添加到内存系统
                self.memory_system.memories[memory_note.id] = memory_note

                # 重建 ChromaDB 索引
                metadata = {
                    "id": memory_note.id,
                    "content": memory_note.content,
                    "keywords": memory_note.keywords,
                    "links": memory_note.links,
                    "retrieval_count": memory_note.retrieval_count,
                    "timestamp": memory_note.timestamp,
                    "last_accessed": memory_note.last_accessed,
                    "context": memory_note.context,
                    "evolution_history": memory_note.evolution_history,
                    "category": memory_note.category,
                    "tags": memory_note.tags
                }
                self.memory_system.retriever.add_document(
                    memory_note.content,
                    metadata,
                    memory_note.id
                )

            if self.verbose:
                print(f"[A-mem] 恢复状态成功，共恢复 {len(memories_state)} 条记忆")

        except Exception as e:
            if self.verbose:
                print(f"[A-mem] 恢复状态失败: {e}")
            raise
