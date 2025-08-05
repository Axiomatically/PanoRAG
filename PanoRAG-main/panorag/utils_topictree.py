import json
import os
import asyncio
from nano_vectordb import NanoVectorDB
from .utils import compute_mdhash_id
import numpy as np

class TopicTreeVectorDB:
    def __init__(self, storage_path: str, embedding_func, embedding_dim: int):
        self.vdb = NanoVectorDB(embedding_dim, storage_file=storage_path)
        self.embedding_func = embedding_func
        self.node_map = {v['__id__']: v for v in self.vdb._NanoVectorDB__storage["data"]}

    async def match(self, keyword: str, top_k: int = 5, threshold: float = 0.35, ids: list[str] = None):
        embedding = await self.embedding_func([keyword])
        
        result = self.vdb.query(
            query=embedding[0],
            top_k=top_k,
            better_than_threshold=threshold,
            ids = ids
        )      
        return result

    def get_node_by_id(self, node_id):
        return self.node_map.get(node_id, None)

    def expand_up(self, node):
        if not node or not node.get("parent"):
            return []
        return [self.get_node_by_id(node["parent"])]

    def expand_down(self, node, limit: int = 5):
        node_id = node["__id__"]
        children = [n for n in self.node_map.values() if n.get("parent") == node_id]
        return children[:limit]

    def multi_hop_expand(self, node, hops=2):
        results = []
        def _recurse(n, level):
            children = self.expand_down(n)
            results.extend(children)
            if level > 1:
                for c in children:
                    _recurse(c, level - 1)
        _recurse(node, hops)
        return results

