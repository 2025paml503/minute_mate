"""Build an executable Jupyter notebook for the complete MinuteMate workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import PROJECT_ROOT


def _markdown(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook(output_path: str | Path | None = None) -> Path:
    """Write the reproducible project notebook and return its path."""
    destination = Path(output_path) if output_path else PROJECT_ROOT / "notebooks" / "MinuteMate.ipynb"
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    cells = [
        _markdown(
            "# MinuteMate: structured meeting summarization\n\n"
            "This notebook demonstrates the SAMSum workflow required by the project: "
            "dataset sampling, prompt and decoding comparisons, validated JSON, local "
            "tool simulation, and quantitative evaluation. Model loading downloads "
            "the configured Hugging Face model on first use."
        ),
        _code(
            "from pathlib import Path\n"
            "import sys\n"
            "ROOT = Path.cwd()\n"
            "if not (ROOT / 'src').is_dir():\n"
            "    ROOT = ROOT.parent\n"
            "sys.path.insert(0, str(ROOT))\n"
            "import json\n"
            "import pandas as pd\n"
            "import torch\n"
            "from datasets import load_dataset\n"
            "from src.minutes.engine import load_model, generate_text\n"
            "from src.minutes.prompts import SYSTEM_ROLE, zero_shot, few_shot, role_based, constrained, detailed\n"
            "from src.minutes.structured import get_minutes\n"
            "from src.minutes.router import propose_calls, execute_proposed_calls\n"
            "from src.minutes.score import score_record\n"
        ),
        _markdown("## Dataset sampling\n\nThe evaluation sample comes from the test split; few-shot examples come only from train."),
        _code(
            'dataset = load_dataset("knkarthick/samsum")\n'
            "def add_bucket(frame):\n"
            "    frame = frame.copy()\n"
            '    frame["n_turns"] = frame["dialogue"].str.count("\\\\n") + 1\n'
            '    frame["bucket"] = pd.cut(frame["n_turns"], bins=[0, 8, 15, 10_000], labels=["short", "medium", "long"])\n'
            "    return frame\n"
            'eval_frame = add_bucket(dataset["test"].to_pandas()).groupby("bucket", observed=True).sample(n=40, random_state=42).reset_index(drop=True)\n'
            'train_frame = add_bucket(dataset["train"].to_pandas())\n'
            'fewshot_pool = train_frame.groupby("bucket", observed=True).sample(n=100, random_state=42)\n'
            'eval_frame[["id", "bucket", "dialogue", "summary"]].head()\n'
        ),
        _markdown("## Baseline and prompt comparison\n\nUse the same held-out conversation to compare concise and detailed instructions."),
        _code(
            'MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"\n'
            "model, tokenizer = load_model(MODEL_NAME)\n"
            "sample = eval_frame.iloc[0]\n"
            'baseline = generate_text(model, tokenizer, f"Summarize the following conversation:\\n{sample[\'dialogue\']}\\n\\nSummary:", do_sample=False)\n'
            'examples = fewshot_pool[fewshot_pool["bucket"] == sample["bucket"]].head(3)[["dialogue", "summary"]].to_dict("records")\n'
            "comparisons = {\n"
            '    "p1_zero_shot": (None, zero_shot(sample["dialogue"])),\n'
            '    "p2_few_shot": (None, few_shot(sample["dialogue"], examples)),\n'
            '    "p3_role": (SYSTEM_ROLE, role_based(sample["dialogue"])),\n'
            '    "p4_constrained": (None, constrained(sample["dialogue"])),\n'
            '    "p5_detailed": (SYSTEM_ROLE, detailed(sample["dialogue"])),\n'
            "}\n"
            "outputs = {name: generate_text(model, tokenizer, prompt, system_prompt=system, do_sample=False) for name, (system, prompt) in comparisons.items()}\n"
            "print('Baseline:', baseline)\n"
            "outputs\n"
        ),
        _markdown("## Decoding controls\n\nGreedy decoding is deterministic. Sampling exposes temperature and top-p controls; inspect variability across seeds."),
        _code(
            "decoding_outputs = {}\n"
            "for label, options in {\n"
            '    "greedy": {"do_sample": False},\n'
            '    "temperature_0_7_top_p_0_9": {"do_sample": True, "temperature": 0.7, "top_p": 0.9},\n'
            "}.items():\n"
            '    decoding_outputs[label] = generate_text(model, tokenizer, constrained(sample["dialogue"]), max_new_tokens=128, **options)\n'
            "decoding_outputs\n"
        ),
        _markdown("## Structured output and tool simulation\n\nPydantic validates participants against transcript speakers. Tool calls remain local simulations; require human confirmation before external actions."),
        _code(
            'minutes, validation_log = get_minutes(model, tokenizer, sample["dialogue"])\n'
            "print(validation_log)\n"
            "if minutes is not None:\n"
            "    proposed_calls = propose_calls(minutes)\n"
            "    simulated_results = execute_proposed_calls(proposed_calls)\n"
            "    print(json.dumps(minutes.model_dump(mode='json'), indent=2))\n"
            "    print(json.dumps({'proposed_calls': proposed_calls, 'simulated_results': simulated_results}, indent=2))\n"
            "else:\n"
            "    print('No validated output; do not execute tools.')\n"
        ),
        _markdown("## Evaluation and error review\n\nROUGE measures lexical overlap, not factuality. Review unsupported names and low-scoring examples manually."),
        _code(
            "metrics = score_record(sample['summary'], baseline, sample['dialogue'])\n"
            "metrics\n"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Notebook output path")
    args = parser.parse_args()
    print(f"Created {build_notebook(args.output)}")


if __name__ == "__main__":
    main()