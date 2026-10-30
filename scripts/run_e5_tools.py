"""Run E5: exercise the validated-minute tools using local in-memory simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.minutes.engine import load_model
from src.minutes.router import execute_proposed_calls, propose_calls
from src.minutes.structured import get_minutes
from src.minutes.tools import TOOL_SPECS
from .common import DATA_DIR, DEFAULT_MODEL, EXPERIMENTS_DIR, ensure_project_dirs, read_csv, write_csv


def _tool_schema(spec: dict[str, Any]) -> dict[str, Any]:
    model = spec["args_model"]
    return {
        "description": spec["description"],
        "input_schema": model.model_json_schema(),
    }


def run_experiment(
    model_name: str = DEFAULT_MODEL,
    input_path: str | Path = DATA_DIR / "dev_samples.csv",
    output_path: str | Path = "experiments/e5_tool_calls.csv",
    limit: int = 5,
) -> pd.DataFrame:
    """Extract minutes, propose calls, and demonstrate only local tool execution."""
    if limit < 1:
        raise ValueError("limit must be greater than zero")
    ensure_project_dirs()
    samples = read_csv(input_path, required_columns=("id", "dialogue"))
    samples = samples.head(limit)
    if samples.empty:
        raise ValueError("No conversations found for the tool demonstration")
    model, tokenizer = load_model(model_name)

    specs_path = EXPERIMENTS_DIR / "tool_specs.json"
    specs_path.write_text(
        json.dumps({name: _tool_schema(spec) for name, spec in TOOL_SPECS.items()}, indent=2),
        encoding="utf-8",
    )

    rows = []
    for _, row in samples.iterrows():
        minutes, validation = get_minutes(model, tokenizer, str(row["dialogue"]))
        calls = propose_calls(minutes) if minutes is not None else []
        results = execute_proposed_calls(calls) if calls else []
        rows.append(
            {
                "id": row["id"],
                "bucket": row.get("bucket", ""),
                "valid_minutes": minutes is not None,
                "minutes": json.dumps(
                    minutes.model_dump(mode="json") if minutes is not None else None,
                    ensure_ascii=False,
                ),
                "tool_calls": json.dumps(calls, ensure_ascii=False),
                "tool_results": json.dumps(results, ensure_ascii=False),
                "execution_mode": "local_in_memory_simulation",
                "validation_errors": json.dumps(validation["errors"], ensure_ascii=False),
            }
        )
    result = pd.DataFrame(rows)
    destination = write_csv(result, output_path)
    print(f"Saved tool demonstration to {destination}")
    print(f"Saved input/output schemas to {specs_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "dev_samples.csv")
    parser.add_argument("--output", type=Path, default=Path("experiments/e5_tool_calls.csv"))
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    run_experiment(args.model, args.input, args.output, args.limit)


if __name__ == "__main__":
    main()