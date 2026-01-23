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
    def __init__(self, embedding_config: Optional[EmbeddingConfig] = None, max_memory: int = 40, use_faiss: bool = True):
        self._items: List[MemoryItem] = []
        self.max_memory = max_memory  # 滑动窗口大小
        self.use_faiss = use_faiss
        self.faiss_manager = None
        self._next_id = 0  # 用于分配唯一ID

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
        if self.use_faiss:
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
                print(f"Faiss初始化成功，向量维度: {self.embedding_dim}")
            except Exception as e:
                print(f"Faiss初始化失败: {e}")
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
                print("Faiss索引已重置")
            except Exception as e:
                print(f"Faiss重置失败: {e}")

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None, embedding: Optional[List[float]] = None) -> None:
        """添加记忆项，实现滑动窗口逻辑"""
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

        # 滑动窗口逻辑：如果超过最大容量，删除最旧的
        if len(self._items) >= self.max_memory:
            # 删除最旧的项（索引0）
            oldest_item = self._items.pop(0)

            # 从Faiss中删除对应的向量
            if self.use_faiss and self.faiss_manager:
                try:
                    # 我们需要记录每个item的faiss_id
                    # 假设faiss_id就是添加时的顺序ID
                    oldest_id = item_id - len(self._items) - 1
                    if oldest_id >= 0:
                        self.faiss_manager.remove_ids([oldest_id])
                        if self.embedding_config.use_real:
                            print(f"从Faiss删除最旧记忆 (ID: {oldest_id}): {oldest_item.text[:30]}...")
                except Exception as e:
                    print(f"从Faiss删除向量失败: {e}")

        # 添加到内存列表
        self._items.append(item)

        # 添加到Faiss索引
        if self.use_faiss and self.faiss_manager and embedding:
            try:
                import numpy as np
                vec = np.array([embedding], dtype='float32')
                success = self.faiss_manager.add_vectors(vec, [item_id])
                if success and self.embedding_config.use_real:
                    print(f"添加记忆到Faiss (ID: {item_id}): {text[:30]}...")
            except Exception as e:
                print(f"添加向量到Faiss失败: {e}")

        self._next_id += 1

    def search_faiss(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """使用Faiss搜索相似向量

        Args:
            query_embedding: 查询向量（已归一化）
            top_k: 返回的相似向量数量

        Returns:
            搜索结果列表，每个结果包含text、meta和distance
        """
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
            print(f"Faiss搜索失败: {e}")
            return []

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
            print(f"Faiss索引已保存到: {index_file}")
            return True
        except Exception as e:
            print(f"保存Faiss索引失败: {e}")
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
