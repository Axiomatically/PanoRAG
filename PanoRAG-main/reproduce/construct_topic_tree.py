import json
import time
import re
from typing import List, Dict, Any
from openai import OpenAI
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

client = OpenAI(base_url=" ")

cls = "agriculture"
def parse_topic_explanation(output: str) -> (str, str):
    match = re.search(r"(?i)^Topic\s*:\s*(.*?)\n(?:Explanation\s*:\s*)?(.*)", output.strip(), re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    if ":" in output:
        topic, explanation = output.split(":", 1)
        return topic.strip(), explanation.strip()
    return output.strip(), ""

with open(f"./{cls}/kv_store_text_chunks.json", "r") as f:
    chunk_data = json.load(f)
chunk_ids = list(chunk_data.keys())
chunks = [chunk_data[cid]["content"] for cid in chunk_ids]

print(" Summarizing each chunk...")
def summarize_chunk(content: str) -> Dict[str, str]:
    prompt = (
        "You are given a paragraph from a knowledge document.\n"
        "Your task is to extract fine-grained semantic facts, such as:\n"
        "- Specific people, locations, organizations, actions, dates, or relationships.\n"
        "- Each topic should represent one clear, indivisible fact or concept from the text.\n"
        "- Do not generalize or abstract. Use phrases directly from the text.\n\n"
        "For each extracted topic, write a brief explanation using only the wording and ideas in the original paragraph.\n"
        "Strictly follow this format:\n"
        "Topic: <topic>\nExplanation: <explanation>\n---\n(repeat for each topic)\n\n"
        f"Text:\n{content}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content
    topic_dict = {}
    for block in raw.split("---"):
        topic, expl = parse_topic_explanation(block)
        if topic:
            topic_dict[topic] = expl
    return topic_dict


def process_chunk(idx):
    chunk = chunks[idx]
    summary = summarize_chunk(chunk)
    return [{"topic": t, "explanation": e, "chunk_id": chunk_ids[idx]} for t, e in summary.items()]

with ThreadPoolExecutor(max_workers=10) as executor:
    all_results = list(tqdm(executor.map(process_chunk, range(len(chunks))), total=len(chunks)))
topic_nodes = [item for sublist in all_results for item in sublist]
print(f"Completed summarization in {time.time() - start_time:.2f}s.\n")


print("Getting topic embeddings...")
def get_embeddings(texts: List[str]) -> np.ndarray:
    results = []
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        res = client.embeddings.create(input=batch, model="text-embedding-3-small")
        results.extend([e.embedding for e in res.data])
    return np.array(results)

topics = [node["topic"] for node in topic_nodes]
embeddings = get_embeddings(topics)

print("Clustering topics (Layer 1)...")
cluster1 = AgglomerativeClustering(n_clusters=150).fit(embeddings)
layer1 = {}
for label, node in zip(cluster1.labels_, topic_nodes):
    layer1.setdefault(label, []).append(node)

print("Summarizing each topic cluster...")

def summarize_cluster(cluster_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    text = "\n".join([f"{n['topic']}: {n['explanation']}" for n in cluster_nodes])
    prompt = (
        "You are given a list of fine-grained topics and their explanations.\n"
        "Please identify the shared theme behind these facts and generate a mid-level summary topic.\n"
        "- Do not introduce abstract ideas, generalizations, or external knowledge.\n"
        "- Reuse only the words and ideas already in the topic names or explanations.\n\n"
        "Strict format:\n"
        "Topic: <merged topic>\nExplanation: <merged explanation>\n\n"
        f"{text}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    output = response.choices[0].message.content
    topic, expl = parse_topic_explanation(output)
    if topic and expl:
        return {"topic": topic, "explanation": expl, "children": cluster_nodes}
    return None

cluster_items = list(layer1.items())
with ThreadPoolExecutor(max_workers=10) as executor:
    intermediate_nodes = list(tqdm(executor.map(lambda pair: summarize_cluster(pair[1]), cluster_items), total=len(cluster_items)))
intermediate_nodes = [n for n in intermediate_nodes if n]

print("Clustering intermediate nodes...")
inter_topics = [n["topic"] for n in intermediate_nodes]
inter_embeds = get_embeddings(inter_topics)
cluster2 = AgglomerativeClustering(n_clusters=50).fit(inter_embeds)
layer2 = {}
for label, node in zip(cluster2.labels_, intermediate_nodes):
    layer2.setdefault(label, []).append(node)

topic_tree = {"topic": "Root", "children": []}

for label, group in tqdm(layer2.items(), desc="Final summarization"):
    combined = "\n".join([f"{n['topic']}: {n['explanation']}" for n in group])
    prompt = (
        "You are given a group of mid-level topics and their explanations.\n"
        "Please write one top-level topic and explanation that reflect their shared focus.\n"
        "- Use only the concepts and wording from the input.\n"
        "- Do not invent abstract categories or add external information.\n\n"
        "Strict format:\n"
        "Topic: <top-level topic>\nExplanation: <combined explanation>\n\n"
        f"{combined}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    output = response.choices[0].message.content
    topic, expl = parse_topic_explanation(output)
    if topic and expl:
        topic_tree["children"].append({"topic": topic, "explanation": expl, "children": group})

with open(f"{cls}_topic_tree.json", "w") as f:
    json.dump(topic_tree, f, indent=2)
print(f"Topic hierarchy saved to {cls}_topic_tree.json")
