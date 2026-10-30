"""Compatibility wrapper for scoring an individual experiment CSV."""

import pandas as pd

from src.minutes.score import hallucinated_names, score_record, speakers_of
from .common import read_csv


def score_file(path, eval_csv="data/eval_samples.csv"):
    """Join source conversations and add ROUGE, length, and grounding indicators."""
    outputs = read_csv(path, required_columns=("id", "output", "gold"))
    conversations = read_csv(eval_csv, required_columns=("id", "dialogue"))
    frame = outputs.merge(
        conversations[["id", "dialogue"]],
        on="id",
        how="left",
        validate="many_to_one",
    )
    if frame["dialogue"].isna().any():
        missing_ids = frame.loc[frame["dialogue"].isna(), "id"].tolist()
        raise ValueError(f"Evaluation conversations missing for IDs: {missing_ids[:10]}")

    metrics = [
        score_record(str(row.gold), str(row.output), str(row.dialogue))
        for row in frame.itertuples(index=False)
    ]
    for metric in ("rouge1", "rouge2", "rougeL", "n_words", "n_hallucinated_names"):
        frame[metric] = [result[metric] for result in metrics]
    frame["hallucinated_names"] = [
        ", ".join(result["hallucinated_names"]) for result in metrics
    ]
    frame["first_speaker"] = frame["dialogue"].apply(
        lambda dialogue: speakers_of(dialogue)[0] if speakers_of(dialogue) else None
    )
    return frame

if __name__ == "__main__":
    df = score_file("experiments/e2_prompt_comparison.csv")
    print(df.groupby("prompt_id")[["rouge1", "rouge2", "rougeL", "n_words"]].mean())