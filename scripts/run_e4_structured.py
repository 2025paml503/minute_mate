"""Run E4: generate and validate JSON meeting minutes for evaluation samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.minutes.engine import load_model
from src.minutes.structured import get_minutes
from .common import DATA_DIR, DEFAULT_MODEL, ensure_project_dirs, read_csv, write_csv


def run_experiment(
    model_name: str = DEFAULT_MODEL,
    input_path: str | Path = DATA_DIR / "eval_samples.csv",
    output_path: str | Path = "experiments/e4_structured.csv",
    limit: int | None = 30,
) -> pd.DataFrame:
    """Generate structured minutes, saving the validated JSON or raw failures."""
    ensure_project_dirs()
    samples = read_csv(input_path, required_columns=("id", "dialogue", "summary"))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be greater than zero or None")
        samples = samples.head(limit)
    if samples.empty:
        raise ValueError("No conversations found for structured-output evaluation")
    model, tokenizer = load_model(model_name)

    rows = []
    for _, row in samples.iterrows():
        minutes, log = get_minutes(model, tokenizer, str(row["dialogue"]))
        structured_output = minutes.model_dump(mode="json") if minutes is not None else None
        rows.append(
            {
                "id": row["id"],
                "bucket": row.get("bucket", ""),
                "output": json.dumps(structured_output, ensure_ascii=False) if structured_output else "",
                "gold": row["summary"],
                "valid": minutes is not None,
                "attempts": log["attempts"],
                "errors": json.dumps(log["errors"], ensure_ascii=False),
                "raw_output": log["raw"][-1] if log["raw"] else "",
            }
        )
        if len(rows) % 10 == 0:
            write_csv(pd.DataFrame(rows), output_path)
            print(f"Processed {len(rows)}/{len(samples)} structured examples")
    result = pd.DataFrame(rows)
    destination = write_csv(result, output_path)
    print(f"Saved structured outputs to {destination}; valid={int(result['valid'].sum())}/{len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "eval_samples.csv")
    parser.add_argument("--output", type=Path, default=Path("experiments/e4_structured.csv"))
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    run_experiment(args.model, args.input, args.output, args.limit)


if __name__ == "__main__":
    main()