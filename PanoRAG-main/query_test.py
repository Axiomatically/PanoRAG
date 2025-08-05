import os
import asyncio
from panorag import PanoRAG, QueryParam
from panorag.llm.openai import gpt_4o_mini_complete, openai_embed
from panorag.kg.shared_storage import initialize_pipeline_status

cls = os.environ.get("cls", "default")
WORKING_DIR = f"../{cls}"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = PanoRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

def main():
    rag = asyncio.run(initialize_rag())
    print(rag.query("How many episodes were in the original run of the hbo show that kristin davis played the character charlotte york goldenblatt?", param=QueryParam(mode="hybrid")))

if __name__ == "__main__":
    main()
