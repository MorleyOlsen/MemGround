# galagent/memory/retriever.py
# Implements two retrieval methods: keyword-based retrieval and vector embedding-based retrieval
from __future__ import annotations

import re
from typing import List, Optional, Dict, Any, AsyncIterator
from galagent.tool import Tool
from galagent.memory.store import MemoryStore, get_qwen_embedding

from galagent.common.openai_harmony import (
    Author,
    Content,
    Message,
    Role,
    TextContent,
)

def _tokenize(s: str) -> List[str]:
    s = (s or "").lower().strip()
    parts = re.split(r"[^\w\u4e00-\u9fff]+", s)
    return [p for p in parts if p]

# Keyword retrieval, selects top 3
class KeywordRetrieverTool(Tool):
    def __init__(self, store: MemoryStore, name: str = "KeywordRetrieverTool"):
        self.store = store
        assert name == "KeywordRetrieverTool"

    @property
    def name(self) -> str:
        return self.get_tool_name()
    
    @classmethod
    def get_tool_name(cls) -> str:
        return "KeywordRetrieverTool"
    
    @property
    def instruction(self) -> str:
        return """
        Use this tool to execute search actions in your chain of thought. The search actions will not be shown to the user. When the agent needs to do some action based on former information, the query will be searched within a corpus, and the top 3 results of the search will be returned to agent.        
        """.strip()
    
    async def _process(self, message: Message) -> AsyncIterator[Message]:
        """Directly call the existing class to perform vector retrieval"""
        import json
        
        channel = message.channel
        raw_text = message.content[0].text
        
        try:
            args = json.loads(raw_text)
            query = args.get("query")
            top_k = args.get("top_k", 3)
            
            if not query:
                yield self.make_response(TextContent(text="Error: Missing 'query' parameter"), channel=channel)
                return
            
            # Call the existing search method
            results = self._search(query, top_k=top_k)
            
            yield self.make_response(
                TextContent(text=json.dumps({
                    "query": query,
                    "top_k": top_k,
                    "results": [{"text": r} for r in results],
                    "total": len(results)
                }, ensure_ascii=False, indent=2)),
                channel=channel
            )
            return
        except Exception as e:
            yield self.make_response(TextContent(text=f"Error: {e}"), channel=channel)
            return
        
    def _search(self, query: str, top_k: int = 3) -> List[str]:
        top_k = int(top_k)
        if top_k <= 0:
            return []

        q_tokens = _tokenize(query)[:12]
        if not q_tokens:
            return []

        scored = []
        for idx, item in enumerate(self.store.items):
            text = item.text.lower()
            score = sum(1 for t in q_tokens if t and t in text)
            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        hits = [self.store.items[idx].text for _, idx in scored[:top_k]]
        return hits

# Embedding-based vector retriever
class VectorRetriever:
    def __init__(self, store: MemoryStore, embedding_dim: int = 1536):
        self.store = store
        self.embedding_dim = embedding_dim
        # Cache generated embeddings to avoid redundant computation
        self._embedding_cache: Dict[int, List[float]] = {}

    def _get_embedding(self, text: str) -> List[float]:
        """Get the embedding for a text, preferring from cache"""
        text_hash = hash(text)
        if text_hash not in self._embedding_cache:
            # Get the embedding configuration from the store
            self._embedding_cache[text_hash] = get_qwen_embedding(text, self.store.embedding_config)
        return self._embedding_cache[text_hash]

    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        if not emb1 or not emb2:
            return 0.0
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        # Calculate magnitudes
        norm1 = sum(a ** 2 for a in emb1) ** 0.5
        norm2 = sum(b ** 2 for b in emb2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        # Calculate cosine similarity
        return dot_product / (norm1 * norm2)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve the most similar memories using the embedding of the query text

        Prefers Faiss for vector retrieval; falls back to local computation if Faiss is unavailable
        """
        top_k = int(top_k)
        if top_k <= 0:
            return []

        # Get the embedding for the query
        query_embedding = self._get_embedding(query)

        # Prefer Faiss for retrieval
        if hasattr(self.store, 'use_faiss') and self.store.use_faiss and self.store.faiss_manager:
            try:
                faiss_results = self.store.search_faiss(query_embedding, top_k)
                if faiss_results:
                    if self.store.embedding_config.use_real:
                        print(f"[Faiss retrieval] Query: {query[:30]}... | Found {len(faiss_results)} results")
                    return [result['text'] for result in faiss_results]
            except Exception as e:
                print(f"Faiss retrieval failed, falling back to local computation: {e}")

        # If Faiss is unavailable or retrieval failed, fall back to local computation
        # Calculate similarity between all memory items and the query
        scored = []
        for idx, item in enumerate(self.store.items):
            # Use the embedding already stored in MemoryItem if available, otherwise generate it
            item_embedding = item.embedding if item.embedding is not None else self._get_embedding(item.text)
            similarity = self._calculate_similarity(query_embedding, item_embedding)
            scored.append((similarity, idx))

        # Sort by similarity in descending order, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [self.store.items[idx].text for _, idx in scored[:top_k]]

        if self.store.embedding_config.use_real:
            print(f"[Local retrieval] Query: {query[:30]}... | Found {len(hits)} results")

        return hits
