import pandas as pd
from datasets import load_dataset

ds = load_dataset("knkarthick/samsum")

def add_bucket(df):
    df = df.copy()
    df["n_turns"] = df["dialogue"].str.count("\n") + 1
    df["bucket"] = pd.cut(df["n_turns"], bins=[0, 8, 15, 10_000], labels=["short", "medium", "long"])
    return df

test = add_bucket(ds["test"].to_pandas())
val = add_bucket(ds["validation"].to_pandas())
train = add_bucket(ds["train"].to_pandas())

# 40 per bucket from TEST = 120 evaluation conversations
eval_samples = test.groupby("bucket", observed=True).sample(n=40, random_state=42)
eval_samples.to_csv("data/eval_samples.csv", index=False)

# 10 per bucket from VAL = 30 conversations for trying prompts
dev_samples = val.groupby("bucket", observed=True).sample(n=10, random_state=42)
dev_samples.to_csv("data/dev_samples.csv", index=False)

# 100 per bucket from TRAIN = pool for few-shot examples
fewshot_pool    = train.groupby("bucket", observed=True).sample(n=100, random_state=42)
fewshot_pool.to_csv("data/fewshot_pool.csv", index=False)

print(len(eval_samples), len(dev_samples), len(fewshot_pool))

