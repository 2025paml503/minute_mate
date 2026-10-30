"""Text-quality metrics used by the experiment evaluation scripts."""

from __future__ import annotations

import re
from typing import Any

from rouge_score import rouge_scorer

_ROUGE = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def speakers_of(dialogue: str) -> list[str]:
    """Return unique speaker labels found before the first colon on each line."""
    return sorted(
        {
            line.split(":", 1)[0].strip()
            for line in dialogue.splitlines()
            if ":" in line and line.split(":", 1)[0].strip()
        }
    )


def hallucinated_names(output: str, dialogue: str) -> list[str]:
    """Heuristically flag capitalized output words absent from the source dialogue."""
    candidates = set(re.findall(r"\b[A-Z][a-z]+\b", output))
    source = dialogue.casefold()
    return sorted(word for word in candidates if word.casefold() not in source)


def score_summary(reference: str, prediction: str) -> dict[str, float]:
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L F-measures."""
    scores = _ROUGE.score(reference or "", prediction or "")
    return {name: score.fmeasure for name, score in scores.items()}


def score_record(
    reference: str,
    prediction: str,
    dialogue: str,
) -> dict[str, Any]:
    """Calculate text metrics and lightweight grounding indicators for one output."""
    metrics: dict[str, Any] = score_summary(reference, prediction)
    metrics["n_words"] = len((prediction or "").split())
    metrics["hallucinated_names"] = hallucinated_names(prediction or "", dialogue or "")
    metrics["n_hallucinated_names"] = len(metrics["hallucinated_names"])
    return metrics