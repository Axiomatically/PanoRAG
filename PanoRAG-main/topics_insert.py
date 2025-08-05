import os
import json
import asyncio
from panorag import PanoRAG
from panorag.llm.openai import gpt_4o_mini_complete, openai_embed
from panorag.kg.shared_storage import initialize_pipeline_status

cls = os.environ.get("cls", "agriculture")
WORKING_DIR = f"./{cls}"
TOPIC_TREE_PATH = f"./datasets/trees/{cls}_topic_tree.json"

# 1. 提取叶子节点 topic: explanation
def extract_leaf_texts(topic_tree_path: str) -> str:
    with open(topic_tree_path, "r", encoding="utf-8") as f:
        tree = json.load(f)

    def traverse(node):
        if not node.get("children"):
            return [f"{node['topic']}: {node.get('explanation', '').strip()}"]
        results = []
        for child in node["children"]:
            results.extend(traverse(child))
        return results

    leaf_lines = traverse(tree)
    return "\n".join(leaf_lines)

# 2. 初始化 RAG 实例
async def initialize_rag():
    rag = PanoRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

# 3. 主入口：提取 + 插入
def main():
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR, exist_ok=True)

    content = extract_leaf_texts(TOPIC_TREE_PATH)
    rag = asyncio.run(initialize_rag())
    rag.insert(content)
    print(f"✅ Inserted {len(content.splitlines())} leaf nodes into RAG storage.")

if __name__ == "__main__":
    main()
