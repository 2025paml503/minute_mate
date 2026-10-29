from datasets import load_dataset

ds = load_dataset("knkarthick/samsum")
print(ds)

ex = ds["train"][0]
print("DAILOGUE:\n", ex["dialogue"])
print("SUMMARY:\n", ex["summary"])