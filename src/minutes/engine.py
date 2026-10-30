"""Loading and autoregressive generation helpers for Hugging Face causal LMs."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    """Choose the best available accelerator, falling back to CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> tuple[Any, Any]:
    """Load a pretrained causal language model and its tokenizer."""
    device = pick_device()
    print(f"Loading model {model_name} on device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"Loaded model {model_name} on device {device}")
    return model, tokenizer


def generate_text(
    model: Any,
    tokenizer: Any,
    user_text: str,
    system_prompt: str | None = None,
    max_new_tokens: int = 120,
    **decoding: Any,
) -> str:
    """Apply the tokenizer's chat template and return only newly generated text."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be greater than zero")
    if not user_text:
        raise ValueError("user_text must not be empty")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            **decoding,
        )
    new_tokens = outputs[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()