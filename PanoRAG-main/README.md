The code for the paper **"PanoRAG: Enabling Consistent Global Topic Awareness in Graph-Based RAG"**.
## Install
```bash
cd PanoRAG-main
pip install -r requirements.txt
```
## LLM Setting
```bash
export OPENAI_API_KEY="sk-..."
export API_BASE=""
```
## Indexing
* You can directly run the script below to automatically complete all the preparation work.
```bash
cd reproduce
./run_index.sh
#python context_insert.py
#python construct_topic_tree.py #Generating the topic-tree
#python build_topic_tree_vdb.py #Embedding for dense search
#python topics_insert.py
```
* Context Insert
```bash
python context_insert.py
```
```python
import os
import asyncio
from panorag import PanoRAG, QueryParam
from panorag.llm.openai import gpt_4o_mini_complete, openai_embed
from panorag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./agriculture"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = PanoRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        # llm_model_func=gpt_4o_complete
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

def main():

    rag = asyncio.run(initialize_rag())

    with open("./agriculture_contexts.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

if __name__ == "__main__":
    main()
```
* Topic-tree Generation
```bash
python construct_topic_tree.py #Generating the topic-tree
python build_topic_tree_vdb.py #Embedding for dense search
```
* Topic Insert
```bash
python topics_insert.py
```
## Querying
```bash
python query.py
```

## Evaluation
```bash
python eval_single_file.py
or
python eval_multi_files.py
```


