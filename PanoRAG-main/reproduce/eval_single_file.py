import re
import json
import jsonlines
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(base_url="")

def evaluate_query_single_model(query: str,
                                std_dict: dict,
                                model_dict: dict,
                                sys_prompt: str):

    key = query.lower().strip()

    if key not in std_dict:
        return (query, "missing in standard answer", None, None)

    if key not in model_dict:
        return (query, "missing in model output", None, None)

    standard = std_dict[key]
    answer = model_dict[key]

    prompt = f"""
### Question:
{query}

### Gold Standard Answer:
{standard}

### Candidate Answer:
{answer}

Please evaluate the answer with a score between 0.0 and 10.0 and a brief reason.

Respond in the following format:
{{
  "Score": float (0.0 - 10.0),
  "Reason": "..."
}}
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
        score = parsed.get("Score", None)
        reason = parsed.get("Reason", "")
        return (query, None, score, reason, content)
    except Exception as e:
        return (query, f"parse error: {str(e)}", None, None, None)


def single_eval(query_file,
                standard_answer_file,
                result_files,
                output_file_path,
                error_file=None):

    with open(query_file, "r", encoding="utf-8") as f:
        data = f.read()
    pattern = re.compile(r"- Question \d+:\s*(.+)")
    queries = [q.strip() for q in pattern.findall(data)]

    with open(standard_answer_file, "r", encoding="utf-8") as f:
        std_data = json.load(f)
    std_dict = {item["query"].strip().lower(): item["result"] for item in std_data}

    model_file = result_files[0]         
    with open(model_file, "r", encoding="utf-8") as f:
        model_data = json.load(f)
    model_dict = {item["query"].strip().lower(): item["result"] for item in model_data}

    error_queries = set()
    if error_file:
        with open(error_file, "r", encoding="utf-8") as f:
            error_data = json.load(f)
            error_queries = {item["query"].strip().lower() for item in error_data}

    sys_prompt = """
You are a precise evaluator reviewing multiple answers to a factual question. Your primary scoring standard is alignment with the gold standard answer provided.

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
"""

    evaluated_data = []
    skipped_queries = []
    score_sum = 0.0
    valid_count = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                evaluate_query_single_model,
                query,
                std_dict,
                model_dict,
                sys_prompt
            ): query
            for query in queries
            if query.lower().strip() not in error_queries
        }

        for future in as_completed(futures):
            query = futures[future]
            result = future.result()

            if len(result) == 5:
                query, skip_reason, score, reason, content = result
            else:
                query, skip_reason, score, reason = result
                content = None

            if skip_reason:
                skipped_queries.append({"query": query, "reason": skip_reason})
            else:
                if isinstance(score, (int, float)):
                    score_sum += score
                    valid_count += 1
                evaluated_data.append({
                    "question": query,
                    "score": score,
                    "reason": reason,
                    "raw": content,
                })
                print(f"Evaluated: {query[:60]}... Score: {score}")
    with jsonlines.open(output_file_path, mode="w") as writer:
        for item in evaluated_data:
            writer.write(item)

    if skipped_queries:
        skipped_path = output_file_path.replace(".jsonl", "_skipped.json")
        with open(skipped_path, "w", encoding="utf-8") as f:
            json.dump(skipped_queries, f, ensure_ascii=False, indent=2)
        print(f"Skipped {len(skipped_queries)} queries saved to: {skipped_path}")

    if valid_count > 0:
        avg_score = score_sum / valid_count
        print(f"\nAverage Score: {avg_score:.2f} ({score_sum:.1f} / {valid_count})")
    else:
        print("\nNo valid scores to calculate average.")

if __name__ == "__main__":

    cls ="agriculture"
    query_file = f"../datasets/questions/{cls}_questions.txt"
    standard_answer_file = f"../datasets/results/{cls}/merged_{cls}_standard_answer.json"
    result_files = [
        f"../datasets/results/{cls}/{cls}_PanoRAG_result.json",
    ]
    error_file = f"../datasets/results/{cls}/{cls}_error.json"
    output_file_path = f"../datasets/results/{cls}/eval_result/{cls}_PanoRAG.jsonl"

    single_eval(query_file,
                standard_answer_file,
                result_files,
                output_file_path,
                error_file=error_file)
