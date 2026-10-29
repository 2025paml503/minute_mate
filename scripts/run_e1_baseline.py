import time, pandas as pd
from minutes.engine import load_model, generate_text

model, tokenizer = load_model("Qwen/Qwen2.5-1.5B-Instruct")
eval = pd.read_csv("data/eval_samples.csv")
rows = []
for i, row in eval.iterrows():
    prompt = f"Summarize the following conversation:\n{row['dialogue']}\n\nSummary:"
    start = time.time()
    answer = generate_text(model, tokenizer, user_text=prompt, do_sample=False)
    end = time.time()
    rows.append({"id": row["id"], "bucket": row["bucket"], "prompt_id": "p0_baseline", "decoding": "greedy", "output": answer, "gold": row["summary"], "seconds": end-start})
    if i % 10 == 0:
        print(i, "done")
        pd.DataFrame(rows).to_csv("experiments/e1_baseline.csv", index=False)



df = pd.DataFrame(rows).to_csv("experiments/e1_baseline.csv", index=False)
print("saved to experiments/e1_baseline.csv")