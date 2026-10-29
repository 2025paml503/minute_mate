from datasets import load_dataset

ds = load_dataset("knkarthick/samsum")
print(ds)

ex = ds["train"][0]
print("DAILOGUE:\n", ex["dialogue"])
print("SUMMARY:\n", ex["summary"])


### Some simple stats
import pandas as pd

df = ds["train"].to_pandas()
df["n_turns"] = df["dailogue"].str.count("\n") + 1
df["n_words"] = df["dailogue"].str.split().str.len()
df["sum_words"] = df["summary"].str.split().str.len()
df["speakers"]  = df["dailogue"].apply(lambda d: sorted({line.split(":")[0].strip() for line in d.split("\n") if ":" in line}))
df["n_speakers"] = df["speakers"].str.len()

print(df[["n_turns", "n_words", "sum_words", "speakers", "n_speakers"]].describe())
print(df["n_speakers"].value_counts().head())