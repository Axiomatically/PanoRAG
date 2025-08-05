import asyncio
import json
import re
from tqdm.asyncio import tqdm as tqdm_async
from typing import Union
from collections import Counter, defaultdict
import warnings
from panorag.utils import compute_mdhash_id
from panorag.llm.openai import gpt_4o_mini_complete, openai_embed
from nano_vectordb import NanoVectorDB

async def build_topic_tree_vdb(topic_tree_json, storage_path, embedding_func, embedding_dim,):
    from tqdm import tqdm

    def flatten_tree(node, parent=None, depth=0):
        cur_id = compute_mdhash_id(node["topic"], prefix="tp-")
        cur = {
            "__id__": cur_id,
            "topic": node["topic"],
            "explanation": node.get("explanation", ""),
            "parent": parent,
            "depth": depth,
            "content": f"{node['topic']}: {node.get('explanation', '')}"
        }
        all_nodes = [cur]
        for child in node.get("children", []):
            all_nodes.extend(flatten_tree(child, parent=cur_id, depth=depth + 1))
        return all_nodes

    with open(topic_tree_json, "r", encoding="utf-8") as f:
        tree = json.load(f)

    all_nodes = flatten_tree(tree)
    contents = [n["content"] for n in all_nodes]
    embeddings = []
    batch_size = 100
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i+batch_size]
        emb = await embedding_func(batch)
        embeddings.extend(emb)

    for i, emb in enumerate(embeddings):
        all_nodes[i]["__vector__"] = emb

    vdb = NanoVectorDB(embedding_dim, storage_file=storage_path)
    vdb.upsert(all_nodes)
    vdb.save()
    print(f"Saved {len(all_nodes)} topic nodes to NanoVectorDB: {storage_path}")

if __name__ == "__main__":
    import asyncio
    cls = "agriculture"
    asyncio.run(build_topic_tree_vdb(
        topic_tree_json=f"../datasets/trees/{cls}_topic_tree.json",     
        storage_path=f"../{cls}/vdb_topic_tree.json",         
        embedding_func=openai_embed,                     
        embedding_dim=1536                                 
    ))
