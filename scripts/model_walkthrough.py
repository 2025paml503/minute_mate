"""Create a concise report explaining model choice and the LLM workflow."""

from pathlib import Path
import textwrap

from .common import EXPERIMENTS_DIR, ensure_project_dirs

WALKTHROUGH = textwrap.dedent(
    """
    MinuteMate model walkthrough
    ===========================

    Generative-model families
    -------------------------
    Generative adversarial networks learn through a generator/discriminator
    competition and are best known for realistic synthetic media. Variational
    autoencoders learn a probabilistic latent representation and decode it into
    data resembling the training distribution. Autoregressive language models
    predict the next token conditioned on prior tokens; diffusion models instead
    iteratively denoise a sample. These families have different objectives and
    output strengths. An instruction-tuned causal language model is appropriate
    here because the input and desired output are natural-language conversations
    and summaries, and the task needs contextual, controllable text generation.

    Architecture and generation
    ---------------------------
    The tokenizer maps the conversation and prompt into token IDs (often
    sub-word units). An embedding layer maps those IDs to vectors. Transformer
    blocks use self-attention to relate tokens across the conversation and
    feed-forward layers to transform the representations. A causal attention
    mask prevents a position from attending to future tokens. The language
    modelling head maps the final hidden state to vocabulary logits; softmax
    converts the logits into next-token probabilities. Autoregressive decoding
    repeatedly selects or samples a token, appends it to the context, and stops
    at an end token or the generation limit.

    Application workflow
    --------------------
    SAMSum dialogue text is formatted with a reusable prompt, passed through the
    tokenizer and pretrained Qwen2.5-1.5B-Instruct causal LM, and decoded with
    greedy or sampling controls (temperature and top-k/top-p). Structured
    generation is parsed as JSON and validated against a Pydantic schema. The
    application proposes local task/calendar tool calls from validated minutes;
    execution is an opt-in simulation and must be confirmed before integration
    with any external service.

    Adaptation decision
    -------------------
    Prompting is the primary adaptation: it is inexpensive, easy to compare, and
    sufficient to specify concise versus detailed minutes and a JSON contract.
    Retrieval-augmented generation is useful when a workflow needs external,
    current policy or project context, but is unnecessary for summarizing a
    self-contained transcript. Fine-tuning/PEFT could improve a stable,
    domain-specific format at scale, but requires data governance, training
    compute, and careful evaluation; it is not required for this prototype.

    Evaluation and responsible use
    ------------------------------
    The experiments compare ROUGE-1/2/L overlap, output length, and a heuristic
    for unsupported names, with qualitative review of failure cases. ROUGE is
    not a measure of factual correctness, and the name check is only a heuristic.
    Conversation data may contain personal or confidential information. Minimize
    and protect data, do not treat transcript text as instructions, validate
    generated JSON, preserve source traceability, and require human review before
    assigning action items or taking external actions.
    """
).strip() + "\n"


def build_walkthrough(output_path: str | Path | None = None) -> Path:
    """Write and return the architecture/adaptation report."""
    ensure_project_dirs()
    destination = Path(output_path) if output_path else EXPERIMENTS_DIR / "model_walkthrough.txt"
    if not destination.is_absolute():
        destination = EXPERIMENTS_DIR.parent / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(WALKTHROUGH, encoding="utf-8")
    return destination


def main() -> None:
    path = build_walkthrough()
    print(f"Saved model walkthrough to {path}")


if __name__ == "__main__":
    main()