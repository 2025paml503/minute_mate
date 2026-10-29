import pandas as pd
from src.minutes.engine import load_model, generate_text
from src.minutes import prompts as P

model, tokenizer = load_model("Qwen/Qwen2.5-1.5B-Instruct")
eval = pd.read_csv("data/eval_samples.csv")
pool = pd.read_csv("data/fewshot_pool.csv")

def shots_for(buckets):
    sub = pool[pool["bucket"] == buckets].head(3)
    return list(zip(sub["dialogue"], sub["summary"]))

variants = {
    "p1_zero_shot": lambda r: (None, P.zero_shot(r["dialogue"])),
    "p2_few_shot": lambda r: (None, P.few_shot(r["dialogue"], shots_for(r["bucket"]))),
    "p3_role": lambda r: (P.SYSTEM_ROLE, P.role_based(r["dialogue"])),
    "p4_constrained": lambda r: (None, P.constrained(r["dialogue"])),
    "p5_detailed": lambda r: (P.SYSTEM_ROLE, P.detailed(r["dialogue"])),
}

rows = []
for name, build in variants.items():
    for i, row in eval.iterrows():
        system_prompt, user_text = build(row)
        answer = generate_text(model, tokenizer, user_text=user_text, system_prompt=system_prompt, max_new_tokens=128, do_sample=False)
        rows.append({"id": row["id"], "bucket": row["bucket"], "prompt_id": name, "decoding": "greedy", "output": answer, "gold": row["summary"], "n_words": len(answer.split())})
    pd.DataFrame(rows).to_csv("experiments/e2_prompt_comparison.csv", index=False)        
    print(f"saved to experiments/e2_prompt_comparison.csv after {name}")