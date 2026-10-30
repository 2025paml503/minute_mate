"""Reusable end-to-end meeting-minutes workflow."""

from __future__ import annotations

from typing import Any

from .router import execute_proposed_calls, propose_calls
from .structured import get_minutes


def run_pipeline(
    dialogue: str,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_retries: int = 1,
    execute_tools: bool = False,
) -> dict[str, Any]:
    """Generate validated minutes and optionally simulate the proposed tool calls.

    Tool execution is opt-in. The included tools are in-memory demonstrations and
    do not create external tasks or calendar events.
    """
    if not isinstance(dialogue, str) or not dialogue.strip():
        raise ValueError("dialogue must be a non-empty string")
    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater")
    if (model is None) != (tokenizer is None):
        raise ValueError("model and tokenizer must either both be provided or both omitted")
    if model is None:
        from .engine import load_model

        model, tokenizer = load_model(model_name)

    minutes, validation = get_minutes(
        model,
        tokenizer,
        dialogue,
        max_retries=max_retries,
    )
    if minutes is None:
        return {
            "status": "invalid_output",
            "minutes": None,
            "validation": validation,
            "tool_calls": [],
            "tool_results": [],
        }

    calls = propose_calls(minutes)
    results = execute_proposed_calls(calls) if execute_tools else []
    return {
        "status": "ok",
        "minutes": minutes.model_dump(mode="json"),
        "validation": validation,
        "tool_calls": calls,
        "tool_results": results,
    }