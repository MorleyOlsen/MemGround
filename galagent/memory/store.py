# galagent/memory/store.py
# 可以改成记忆滑动窗口的维护函数
from __future__ import annotations

import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path
import os
from openai import OpenAI

from galagent.common.config import EmbeddingConfig, load_embedding_config
from galagent.memory.FaissManager import ThreadSafeFaissManager
from galagent.memory.base_mem_agent import BaseMemAgent


def get_qwen_embedding(text: str, config: EmbeddingConfig) -> List[float]:
    """调用qwen3-vl-embedding接口生成真实的embedding
    
    Args:
        text: 要生成embedding的文本
        config: Embedding配置
        
    Returns:
        生成的embedding向量
    """
    # 根据配置决定是否使用真实embedding服务
    if config.use_real and config.api_key:
        try:
            client = OpenAI(
                api_key=config.api_key,  
                base_url=config.base_url
            )

            completion = client.embeddings.create(
                model=config.model,
                input=text
            )
            embedding = completion.data[0].embedding
            print(f"Generated real Qwen embedding for text: {text[:20]}... (dim: {len(embedding)})")
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")


@dataclass
class MemoryItem:
    text: str
    meta: Dict[str, Any]
    embedding: Optional[List[float]] = None  # 添加embedding字段


class MemoryStore:
    def __init__(self, embedding_config: Optional[EmbeddingConfig] = None, use_faiss: bool = True, verbose: bool = False, memory_log_file: Optional[str] = None, mem_agent: Optional['BaseMemAgent'] = None, use_mem: bool = False):
        self._items: List[MemoryItem] = []
        self.use_faiss = use_faiss
        self.faiss_manager = None
        self._next_id = 0  # 用于分配唯一ID
        self.verbose = verbose  # 是否在控制台输出记忆
        self.memory_log_file = memory_log_file  # 记忆日志文件路径

        # Mem agent 集成
        self.mem_agent = mem_agent
        self.use_mem = use_mem and mem_agent is not None

        # 获取项目根目录
        root_path = Path(__file__).resolve().parent.parent.parent

        # 加载embedding配置
        if embedding_config is None:
            # 默认从项目根目录的config.yaml加载
            config_path = root_path / "config.yaml"
            embedding_config = load_embedding_config(config_path)

        self.embedding_config = embedding_config
        self.embedding_dim = embedding_config.dim

        # 如果启用Faiss，初始化FaissManager
        if self.use_faiss and not self.use_mem:
            try:
                # 创建faiss_data目录存储索引
                faiss_dir = root_path / "faiss_data"
                faiss_dir.mkdir(exist_ok=True)
                index_file = faiss_dir / "memory.index"

                # 使用内积（余弦相似度）作为度量
                self.faiss_manager = ThreadSafeFaissManager(
                    dim=self.embedding_dim,
                    index_file=str(index_file) if index_file.exists() else None,
                    metric='IP'  # Inner Product for cosine similarity
                )
                print(f"Faiss initialized successfully, vector dim: {self.embedding_dim}")
            except Exception as e:
                print(f"Faiss initialization failed: {e}")
                self.use_faiss = False

    def reset(self) -> None:
        """重置内存存储"""
        self._items.clear()
        self._next_id = 0
        # 重新初始化Faiss索引
        if self.use_faiss and self.faiss_manager:
            try:
                self.faiss_manager = ThreadSafeFaissManager(
                    dim=self.embedding_dim,
                    metric='IP'
                )
                print("Faiss index reset")
            except Exception as e:
                print(f"Faiss reset failed: {e}")

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None, embedding: Optional[List[float]] = None) -> None:
        """添加记忆项（仅负责添加，不管理窗口大小）

        注意：添加后需要调用 game_utils.manage_memory() 来管理记忆窗口
        """
        text = (text or "").strip()
        if not text:
            return

        # 如果没有提供embedding，自动生成
        if embedding is None:
            embedding = get_qwen_embedding(text, self.embedding_config)

        # 如果使用Faiss且embedding生成失败，返回
        if self.use_faiss and embedding is None:
            print("Warning: embedding generation failed, skipping...")
            return

        # 为embedding归一化（用于余弦相似度）
        if embedding and self.use_faiss:
            import numpy as np
            emb_array = np.array(embedding, dtype='float32')
            norm = np.linalg.norm(emb_array)
            if norm > 0:
                embedding = (emb_array / norm).tolist()

        # 创建新的记忆项
        item_id = self._next_id
        item = MemoryItem(text=text, meta=meta or {}, embedding=embedding)

        # 添加到内存列表
        self._items.append(item)

        # 添加到Faiss索引
        if self.use_faiss and self.faiss_manager and embedding:
            try:
                import numpy as np
                vec = np.array([embedding], dtype='float32')
                success = self.faiss_manager.add_vectors(vec, [item_id])
                if success and self.embedding_config.use_real:
                    print(f"Added memory to Faiss (ID: {item_id}): {text[:30]}...")
            except Exception as e:
                print(f"Failed to add vector to Faiss: {e}")

        self._next_id += 1

    def search_mem(self, query_embedding: List[float], top_k: int = 3, query_text: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """使用Faiss或Mem agent搜索相似向量

        Args:
            query_embedding: 查询向量（已归一化）
            top_k: 返回的相似向量数量
            query_text: 查询文本（用于Mem agent搜索）
            filters: 可选的过滤条件（用于Mem agent搜索，如 {"role": "user"}）

        Returns:
            搜索结果列表，每个结果包含text、meta和distance/score
        """
        # 如果使用Mem agent，使用Mem agent搜索
        if self.use_mem and self.mem_agent and query_text:
            results = self.mem_agent.search_memories(query_text, top_k=top_k, filters=filters)
            # 转换格式以匹配Faiss返回格式
            return [{
                "text": r["text"],
                "meta": r.get("metadata", {}),
                "distance": r.get("score", 0.0),
                "similarity": r.get("score", 0.0)
            } for r in results]

        # 否则使用Faiss
        if not self.use_faiss or not self.faiss_manager:
            return []

        try:
            import numpy as np

            # 归一化查询向量
            query_array = np.array(query_embedding, dtype='float32')
            norm = np.linalg.norm(query_array)
            if norm > 0:
                query_array = query_array / norm

            # 搜索
            distances, indices = self.faiss_manager.search(
                query_array.reshape(1, -1),
                k=min(top_k, len(self._items))
            )

            # 构建结果
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < len(self._items):
                    # 计算在当前items中的实际索引
                    # faiss中的ID是连续的，需要映射到当前的items
                    actual_idx = idx - (self._next_id - len(self._items))
                    if 0 <= actual_idx < len(self._items):
                        item = self._items[actual_idx]
                        results.append({
                            "text": item.text,
                            "meta": item.meta,
                            "distance": float(dist),
                            "similarity": float(dist)  # IP距离就是余弦相似度
                        })

            return results
        except Exception as e:
            print(f"Faiss search failed: {e}")
            return []

    def search_faiss(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """使用Faiss搜索相似向量（保留向后兼容性）

        Args:
            query_embedding: 查询向量（已归一化）
            top_k: 返回的相似向量数量

        Returns:
            搜索结果列表，每个结果包含text、meta和distance
        """
        return self.search_mem(query_embedding, top_k=top_k)

    def save_faiss_index(self) -> bool:
        """保存Faiss索引到磁盘"""
        if not self.use_faiss or not self.faiss_manager:
            return False

        try:
            root_path = Path(__file__).resolve().parent.parent.parent
            faiss_dir = root_path / "faiss_data"
            faiss_dir.mkdir(exist_ok=True)
            index_file = faiss_dir / "memory.index"

            self.faiss_manager.save(str(index_file))
            print(f"Faiss index saved to: {index_file}")
            return True
        except Exception as e:
            print(f"Failed to save Faiss index: {e}")
            return False

    def recent(self, k: int = 10) -> List[MemoryItem]:
        k = max(0, min(int(k), 500))
        return self._items[-k:]

    @property
    def items(self) -> List[MemoryItem]:
        return self._items

    @property
    def embeddings(self) -> List[List[float]]:
        """获取所有记忆项的embedding列表"""
        return [item.embedding for item in self._items if item.embedding is not None]

    def to_chat_messages(self, role_key: str = "role", include_system: bool = True, max_memories: int = 100) -> List[Dict[str, str]]:
        """将记忆转换为多轮对话格式的消息列表

        Args:
            role_key: 从meta中提取角色的键名，默认为"role"
            include_system: 是否包含system角色的消息，默认True
            max_memories: 最多返回的记忆数量，默认100（仅对Mem agent有效）

        Returns:
            格式为 [{"role": "user/assistant/system", "content": "..."}] 的消息列表
        """
        # 如果使用Mem agent，从Mem agent获取记忆
        if self.use_mem and self.mem_agent:
            all_memories = self.mem_agent.get_all_memories()

            # 按 created_at 排序（从旧到新）
            # created_at 是顶层字段，格式为 YYYYMMDDHHmm 或 ISO格式
            sorted_memories = sorted(
                all_memories,
                key=lambda m: m.get("created_at", "")
            )

            # 获取最新的 max_memories 条
            recent_memories = sorted_memories[-max_memories:] if len(sorted_memories) > max_memories else sorted_memories

            messages = []
            for mem in recent_memories:
                role = mem.get("metadata", {}).get(role_key, "user")

                # 根据参数决定是否跳过system消息
                if not include_system and role == "system":
                    continue

                messages.append({
                    "role": role,
                    "content": mem.get("text", "")
                })
            return messages

        # 否则使用传统方式
        messages = []
        for item in self._items:
            role = item.meta.get(role_key, "user")  # 默认为user角色

            # 根据参数决定是否跳过system消息
            if not include_system and role == "system":
                continue

            messages.append({
                "role": role,
                "content": item.text
            })

        return messages

    def add_message(self, content: str, role: str = "user", **extra_meta) -> None:
        """便捷方法：添加一条对话消息到记忆中

        Args:
            content: 消息内容
            role: 角色类型，可以是 "user", "assistant", "system"
            **extra_meta: 额外的元数据

        示例:
            store.add_message("请帮我分析这个问题", role="user", step=1)
            store.add_message("好的，让我来分析", role="assistant", step=1)
        """
        meta = {"role": role, **extra_meta}

        # 如果使用Mem agent，添加到Mem agent
        if self.use_mem and self.mem_agent:
            self.mem_agent.add_memory(content, metadata=meta)
        else:
            # 使用传统方式添加
            self.add(content, meta=meta)

        # 输出到控制台
        if self.verbose:
            step = extra_meta.get('step', '?')
            role_emoji = {"user": "📄", "assistant": "🤖", "system": "ℹ️"}.get(role, "💬")
            print(f"\n{role_emoji} [Step {step}] [{role.upper()}]")
            print(f"  {content[:200]}{'...' if len(content) > 200 else ''}")

        # 输出到文件
        if self.memory_log_file:
            try:
                import json
                from datetime import datetime

                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "role": role,
                    "content": content,
                    "meta": extra_meta
                }

                with open(self.memory_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Failed to write memory log: {e}")

    def delete_oldest(self, count: int = 1) -> int:
        """删除最早的若干条记忆（用于上下文长度管理）

        Args:
            count: 要删除的记忆数量，默认为1

        Returns:
            实际删除的记忆数量

        示例:
            # 当检测到上下文超长时
            deleted = store.delete_oldest(5)
            print(f"删除了 {deleted} 条最早的记忆")
        """
        if count <= 0:
            return 0

        # 计算实际可删除的数量
        actual_count = min(count, len(self._items))

        if actual_count == 0:
            return 0

        # 删除最早的记忆
        for _ in range(actual_count):
            oldest_item = self._items.pop(0)

            # 从Faiss中删除对应的向量
            if self.use_faiss and self.faiss_manager:
                try:
                    # 计算要删除的faiss ID
                    oldest_id = self._next_id - len(self._items) - 1
                    if oldest_id >= 0:
                        self.faiss_manager.remove_ids([oldest_id])
                        if self.embedding_config.use_real:
                            print(f"Deleted memory (ID: {oldest_id}): {oldest_item.text[:30]}...")
                except Exception as e:
                    print(f"Failed to remove vector from Faiss: {e}")

        return actual_count

    def delete_by_priority(self, count: int = 1) -> int:
        """按优先级删除记忆（优先删除assistant消息）

        删除策略：
        1. 优先删除role为assistant的消息（从最早的开始）
        2. 如果assistant消息不足，再删除其他消息（从最早的开始）

        Args:
            count: 要删除的记忆数量，默认为1

        Returns:
            实际删除的记忆数量

        示例:
            # 当检测到上下文超长时，优先删除assistant消息
            deleted = store.delete_by_priority(5)
            print(f"删除了 {deleted} 条记忆（优先assistant）")
        """
        if count <= 0:
            return 0

        deleted_count = 0

        # 第一阶段：删除assistant消息（从最早的开始）
        i = 0
        while i < len(self._items) and deleted_count < count:
            item = self._items[i]
            if item.meta.get("role") == "assistant":
                # 删除这条记忆
                deleted_item = self._items.pop(i)
                deleted_count += 1

                # 从Faiss中删除对应的向量
                if self.use_faiss and self.faiss_manager:
                    try:
                        # 计算要删除的faiss ID
                        item_id = self._next_id - len(self._items) - deleted_count
                        if item_id >= 0:
                            self.faiss_manager.remove_ids([item_id])
                            if self.embedding_config.use_real:
                                print(f"Deleted assistant memory (ID: {item_id}): {deleted_item.text[:30]}...")
                    except Exception as e:
                        print(f"Failed to remove vector from Faiss: {e}")
                # 不增加i，因为删除后当前位置是下一个元素
            else:
                i += 1

        # 第二阶段：如果还需要删除更多，调用delete_oldest删除最早的其他消息
        remaining = count - deleted_count
        if remaining > 0:
            deleted_count += self.delete_oldest(remaining)

        return deleted_count

    def get_total_tokens_estimate(self, chars_per_token: float = 2.5) -> int:
        """估算当前所有记忆的token数量

        Args:
            chars_per_token: 每个token的平均字符数，中文约2.5，英文约4

        Returns:
            估算的总token数

        示例:
            tokens = store.get_total_tokens_estimate()
            if tokens > 4000:
                store.delete_oldest(5)
        """
        total_chars = sum(len(item.text) for item in self._items)
        return int(total_chars / chars_per_token)


    def get_memory_context(self) -> Dict[str, Any]:
        """获取当前存储的记忆作为游戏上下文

        Returns:
            包含对话历史和记忆统计信息的字典
            {
                "conversation_history": List[Dict[str, str]],  # 对话历史
                "total_items": int,  # 总记忆条数
                "total_tokens": int,  # 估算的总token数
            }
        """
        return {
            "conversation_history": self.to_chat_messages(),
            "total_items": len(self._items),
            "total_tokens": self.get_total_tokens_estimate(),
        }

    def get_state(self) -> Dict[str, Any]:
        """获取记忆存储状态用于checkpoint

        Returns:
            包含所有记忆项的状态字典
        """
        state = {
            "items": [
                {
                    "text": item.text,
                    "meta": item.meta,
                    "embedding": item.embedding
                }
                for item in self._items
            ],
            "total_items": len(self._items),
            "total_tokens": self.get_total_tokens_estimate(),
            "next_id": self._next_id,
            "use_mem": self.use_mem
        }

        # 如果使用Mem agent，保存Mem agent状态
        if self.use_mem and self.mem_agent:
            state["mem_agent_state"] = self.mem_agent.get_state()

        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        """从checkpoint恢复记忆存储状态

        Args:
            state: 记忆状态字典
        """
        from galagent.memory.store import MemoryItem

        self._items.clear()

        for item_data in state["items"]:
            item = MemoryItem(
                text=item_data["text"],
                meta=item_data.get("meta", {}),
                embedding=item_data.get("embedding")
            )
            self._items.append(item)

        # 恢复next_id
        self._next_id = state.get("next_id", len(self._items))

        # 恢复use_mem标志
        self.use_mem = state.get("use_mem", False)

        # 如果使用Mem agent，恢复Mem agent状态
        if self.use_mem and self.mem_agent and "mem_agent_state" in state:
            self.mem_agent.restore_state(state["mem_agent_state"])

        print(f"[Memory] State restored: {len(self._items)} memories, "
              f"~{self.get_total_tokens_estimate()} tokens"
              f"{', using mem agent' if self.use_mem else ''}")
