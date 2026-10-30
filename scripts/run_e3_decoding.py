import pandas as pd, torch
from src.minutes.engine import load_model, generate_text
from src.minutes import prompts as P

DECODING = {
    "greedy":           dict(do_sample=False),
    "t0.3":             dict(do_sample=True, temperature=0.3, top_p=1.0, top_k=0),
    "t0.7":             dict(do_sample=True, temperature=0.7, top_p=1.0, top_k=0),
    "t1.0":             dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0),
    "top0.7_topp0.9":   dict(do_sample=True, temperature=0.7, top_p=0.9),
    "top0.7_topk50":    dict(do_sample=True, temperature=0.7, top_k=50),
}
model, tokenizer = load_model("Qwen/Qwen2.5-1.5B-Instruct")
sample = pd.read_csv("data/eval_samples.csv").groupby("bucket", observed=True).head(14).head(40)  ## 40 conversations
rows = []
for name, cfg in DECODING.items():
    repeats = 1 if name == "greedy" else 3
    for rep in range(repeats):
        torch.manual_seed(42 + rep)
        for _, r in sample.iterrows():
            out = generate_text(model, tokenizer, P.constrained(r["dialogue"]), max_new_tokens=128, **cfg)
            rows.append({"id": r["id"], "decoding": name , "repeat": rep, "output": out, "gold": r["summary"]})
        pd.DataFrame(rows).to_csv("experiments/e3_decoding_comparison.csv", index=False)        
    print(f"saved to experiments/e3_decoding_comparison.csv after {name}")