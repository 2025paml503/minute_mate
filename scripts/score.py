import re, pandas as pd, numpy as np
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def speakers_of(dialogue):
    return sorted({line.split(":")[0].strip() for line in dialogue.split("\n") if ":" in line})

def hallucinated_names(output, dialogue):
    """ Return capitalized words in the output that are not in the dialogue. """
    words = re.findall(r"\b([A-Z][a-z]+)\b", output)
    return sorted(w for w in set(words) if w.lower() not in dialogue.lower())

def score_file(path, eval_csv="data/eval_samples.csv"):
    df = pd.read_csv(path).merge(pd.read_csv(eval_csv)[["id", "dialogue"]], on="id")
    for k in ["rouge1", "rouge2", "rougeL"]:
        df[k] = df.apply(lambda r: scorer.score(r["gold"], r["output"])[k].fmeasure, axis=1)
    df["hallucinated_names"] = df.apply(lambda r: ", ".join(hallucinated_names(r["output"], r["dialogue"])), axis=1)
    df["first_speaker"] = df["dialogue"].apply(lambda d: speakers_of(d)[0] if speakers_of(d) else None)
    df["n_words"] = df["output"].str.split().str.len()
    return df 

if __name__ == "__main__":
    df = score_file("experiments/e2_prompt_comparison.csv")
    print(df.groupby("prompt_id")[["rouge1", "rouge2", "rougeL", "n_words"]].mean())