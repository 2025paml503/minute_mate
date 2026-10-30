"""Score experiment outputs and export aggregate metrics and failure cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.minutes.score import score_record
from .common import DATA_DIR, EXPERIMENTS_DIR, ensure_project_dirs, read_csv, write_csv

EXPERIMENT_FILES = {
    "e1_baseline": ("e1_baseline.csv",),
    "e2_prompts": ("e2_prompt_comparison.csv", "e2_prompt_comparision.csv"),
    "e3_decoding": ("e3_decoding_comparison.csv", "e3_decoding.csv"),
    "e4_structured": ("e4_structured.csv",),
}


def _text_from_output(value: Any) -> str:
    """Read plain summary text or extract the summary field from generated JSON."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, dict):
        summary = parsed.get("summary")
        return str(summary) if summary is not None else text
    return text


def _existing_experiment(filenames: tuple[str, ...]) -> Path | None:
    for filename in filenames:
        candidate = EXPERIMENTS_DIR / filename
        if candidate.is_file():
            return candidate
    return None


def score_experiment(path: str | Path, experiment_name: str) -> pd.DataFrame:
    """Score one experiment CSV against its reference summaries and source chats."""
    experiment = read_csv(path, required_columns=("id", "output", "gold"))
    conversations = read_csv(DATA_DIR / "eval_samples.csv", required_columns=("id", "dialogue"))
    frame = experiment.merge(
        conversations[["id", "dialogue"]],
        on="id",
        how="left",
        validate="many_to_one",
    )
    if frame["dialogue"].isna().any():
        missing_ids = frame.loc[frame["dialogue"].isna(), "id"].tolist()
        raise ValueError(f"Evaluation conversations missing for IDs: {missing_ids[:10]}")

    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        prediction = _text_from_output(row["output"])
        metrics = score_record(str(row["gold"]), prediction, str(row["dialogue"]))
        records.append(
            {
                **row,
                "experiment": experiment_name,
                "output_text": prediction,
                **metrics,
            }
        )
    return pd.DataFrame.from_records(records)


def evaluate_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every available experiment and save summary/failure CSVs."""
    ensure_project_dirs()
    scored: list[pd.DataFrame] = []
    for experiment_name, candidates in EXPERIMENT_FILES.items():
        path = _existing_experiment(candidates)
        if path is not None:
            scored.append(score_experiment(path, experiment_name))
    if not scored:
        expected = ", ".join(filename for names in EXPERIMENT_FILES.values() for filename in names)
        raise FileNotFoundError(
            f"No experiment output CSVs found in {EXPERIMENTS_DIR}. Expected one of: {expected}"
        )

    details = pd.concat(scored, ignore_index=True)
    grouping = ["experiment"]
    for optional in ("prompt_id", "decoding"):
        if optional in details.columns:
            grouping.append(optional)
    metrics = ["rouge1", "rouge2", "rougeL", "n_words", "n_hallucinated_names"]
    summary = (
        details.groupby(grouping, dropna=False)[metrics]
        .mean()
        .reset_index()
        .sort_values(grouping, kind="stable")
    )
    summary_path = write_csv(summary, "experiments/eval_summary.csv")

    failures = details.sort_values(["rougeL", "n_hallucinated_names"], ascending=[True, False])
    failures = failures.groupby("experiment", group_keys=False, sort=False).head(3).copy()
    failures["failure_type"] = failures.apply(
        lambda row: (
            "possible unsupported name"
            if row["n_hallucinated_names"] > 0
            else "low lexical overlap; review for omission or paraphrase"
        ),
        axis=1,
    )
    failure_columns = [
        column
        for column in (
            "experiment",
            "id",
            "prompt_id",
            "decoding",
            "rougeL",
            "n_hallucinated_names",
            "hallucinated_names",
            "failure_type",
            "dialogue",
            "gold",
            "output_text",
        )
        if column in failures.columns
    ]
    failure_path = write_csv(failures[failure_columns], "experiments/e6_failure_cases.csv")
    print(f"Saved evaluation summary to {summary_path}")
    print(f"Saved failure analysis to {failure_path}")
    print(summary.to_string(index=False))
    return summary, failures[failure_columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    evaluate_all()


if __name__ == "__main__":
    main()