import re
import json
import jsonlines
from openai import OpenAI
from statistics import mean
import csv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import get_close_matches

def normalize_query(query: str) -> str:
    return query.strip().lower()

def match_query(key, std_dict):
    if key in std_dict:
        return key
    close_matches = get_close_matches(key, std_dict.keys(), n=1, cutoff=0.85)
    return close_matches[0] if close_matches else None

client = OpenAI(base_url="")

def evaluate_query(query, std_dict, all_model_outputs, sys_prompt, result_files):
    key = normalize_query(query)
    matched_key = match_query(key, std_dict)

    if not matched_key:
        return (query, "missing in standard answer", None, None, None)

    standard = std_dict[matched_key]
    answers_list = []
    valid_model_indices = []

    for model_idx, model_dict in enumerate(all_model_outputs):
        if matched_key in model_dict:
            answers_list.append(model_dict[matched_key])
            valid_model_indices.append(model_idx)

    if not answers_list:
        return (query, "missing in all model outputs", None, None, None)

    shuffled_indices = list(range(len(answers_list)))
    random.shuffle(shuffled_indices)
    shuffled_answers = [answers_list[i] for i in shuffled_indices]
    shuffled_model_indices = [valid_model_indices[i] for i in shuffled_indices]

    prompt = f"""
### Question:
{query}

### Gold Standard Answer:
{standard}

### Candidate Answers:
"""
    for idx, ans in enumerate(shuffled_answers, 1):
        prompt += f"Answer {idx}:\n{ans}\n\n"

    prompt += """
Please return your evaluation in this format:
{
  "Answer 1": {"Score": 0.0-10.0, "Reason": "... (brief)"},
  ...
  "Ranking": {
      "Answer 1": 1-n,
      ...
  }
}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)

        scores = []
        for idx_in_prompt, model_idx in enumerate(shuffled_model_indices):
            score = parsed.get(f"Answer {idx_in_prompt+1}", {}).get("Score")
            if isinstance(score, (int, float)):
                scores.append((model_idx, score))

        scored_answers = [
            {"model_index": model_idx, "model_file": result_files[model_idx], "score": score}
            for model_idx, score in scores
        ]

        ranked = sorted(scores, key=lambda x: -x[1])
        ranks = [(model_idx, rank) for rank, (model_idx, _) in enumerate(ranked, start=1)]

        return (query, None, scored_answers, ranks, content)
    except Exception as e:
        return (query, f"parse error: {str(e)}", None, None, None)

def single_eval(query_file, standard_answer_file, result_files, output_file_path, error_file=None):
    with open(query_file, "r", encoding="utf-8") as f:
        data = f.read()
    queries = re.findall(r"- Question \d+: (.+)", data)

    with open(standard_answer_file, "r", encoding="utf-8") as f:
        std_data = json.load(f)
    std_dict = {normalize_query(item["query"]): item["result"] for item in std_data}

    error_queries = set()
    if error_file:
        with open(error_file, "r", encoding="utf-8") as f:
            error_data = json.load(f)
            error_queries = {normalize_query(item["query"]) for item in error_data}

    all_model_outputs = []
    for file in result_files:
        with open(file, "r", encoding="utf-8") as f:
            results = json.load(f)
            result_dict = {normalize_query(item["query"]): item["result"] for item in results}
            all_model_outputs.append(result_dict)

    file_scores = [[] for _ in range(len(result_files))]
    file_ranks = [[] for _ in range(len(result_files))]
    skipped_queries = []
    evaluated_data = []

    sys_prompt = """
You are a precise evaluator reviewing multiple answers to a factual question. Each answer should be graded with high priority on factual correctness and alignment with the gold standard.

Use the reference answer as the ground truth. It is intentionally concise and accurate. Prioritize:
- **Correctness**: Does the answer accurately and directly respond to the question?
- **Alignment**: Is the key fact or idea consistent with the gold standard answer?

Secondarily, consider:
- **Clarity**: Is the answer understandable and well phrased?
- **Conciseness**: Is the explanation unnecessarily verbose or repetitive?
- **Comprehensiveness** – Does the answer sufficiently cover all key aspects of the question?
- **Empowerment** – Does the answer help the reader understand the topic and make informed judgments?

Scoring instructions:
- Assign a score from 0.0 to 10.0 with one decimal place.
- Use fractional scores (e.g., 7.3, 9.6) to reflect subtle quality differences.
- Be fair: do not reward verbosity or style if the content is factually incorrect.

Provide a very brief reason (1–2 sentences) for each score. Output must be a JSON object with scores and ranking.
"""

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(evaluate_query, query, std_dict, all_model_outputs, sys_prompt, result_files): query
            for query in queries if normalize_query(query) not in error_queries
        }
        for future in as_completed(futures):
            query = futures[future]
            result = future.result()
            if len(result) == 5:
                query, skip_reason, scores, ranks, content = result
            else:
                query, skip_reason, scores, ranks = result
                content = None

            if skip_reason:
                skipped_queries.append({"query": query, "reason": skip_reason})
            else:
                for item in scores:
                    file_scores[item["model_index"]].append(item["score"])
                for model_idx, rank in ranks:
                    file_ranks[model_idx].append(rank)
                evaluated_data.append({
                    "question": query,
                    "model_scores": scores,
                    "evaluation": content
                })
                print(f"Evaluated: {query[:60]}...")

    with jsonlines.open(output_file_path, mode="w") as writer:
        for item in evaluated_data:
            writer.write(item)

    for idx, (scores, ranks) in enumerate(zip(file_scores, file_ranks)):
        avg_score = mean(scores) if scores else 0
        avg_rank = mean(ranks) if ranks else 0
        print(f"result_{idx+1}.json -> Avg Score: {avg_score:.2f}, Avg Rank: {avg_rank:.2f}")

    with open(output_file_path.replace(".jsonl", "_eval_summary.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Result File", "Average Score", "Average Rank"])
        for idx, (scores, ranks) in enumerate(zip(file_scores, file_ranks)):
            avg_score = mean(scores) if scores else 0
            avg_rank = mean(ranks) if ranks else 0
            writer.writerow([result_files[idx], f"{avg_score:.2f}", f"{avg_rank:.2f}"])


if __name__ == "__main__":
    cls = ""
    query_file = f"../datasets/questions/{cls}_questions.txt"
    standard_answer_file = f"../datasets/results/{cls}/merged_{cls}_standard_answer.json"
    result_files = ["file_path_1.json",
                    "file_path_2.json",
                    "file_path_3.json",
                    "file_path_4.json",

    ]
    error_file = f"../datasets/results/{cls}/{cls}_error.json"
    output_file_path = f"../datasets/results/eval_result_{cls}_multi_files.jsonl"

    single_eval(query_file, standard_answer_file, result_files, output_file_path, error_file=error_file)