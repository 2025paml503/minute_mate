# MinuteMate: Structured Meeting and Conversation Summarization

## Technical Project Report

**Course:** Applied Generative AI  
**Project dataset:** SAMSum (D1)  
**Group name:** [Insert group name]  
**Group members:** [Insert names and IDs]  
**Instructor:** [Insert instructor name]  
**Date:** [Insert submission date]

> **Evidence note.** This report describes the implementation and the sampled data currently in the project workspace. At report preparation time, the experiment result CSVs in `experiments/` were empty (0 bytes). Consequently, no model-generated scores or comparative outcomes are claimed below. Experiment design and commands are documented, and result tables identify what remains to be run. Replace the pending-results sections with measured outputs before presenting this as a completed empirical evaluation.

## Abstract

MinuteMate is a prototype workflow for turning short, informal conversations into useful meeting minutes. Given a dialogue, it generates a concise summary, participants, key points, decisions, action items, open questions, and a coarse sentiment label. A pretrained instruction-tuned causal language model is used rather than training a model from scratch. Prompt variants and decoding controls make generation behavior inspectable, while a Pydantic schema checks the shape and some source consistency of structured results. Validated minutes can also produce proposed task or follow-up-meeting calls; the included tools are local in-memory simulations, not integrations with real services.

The project uses the SAMSum dataset and its human-written summaries as references. Existing prepared files contain 120 balanced test examples, 30 validation examples, and a 300-example training pool for few-shot demonstrations. The test sample contains 40 conversations in each of three dialogue-length buckets. The implementation includes baseline, prompt-comparison, decoding, structured-output, tool-simulation, evaluation, and notebook workflows. Planned evaluation includes ROUGE overlap, length, a heuristic unsupported-name indicator, JSON validity, and qualitative failure review. Those result files are currently empty, so this report does not infer or fabricate model quality. It instead gives the reproducible design, current implementation, responsible-use controls, limitations, and the measurements required to complete the evaluation.

## 1. Problem Statement and Motivation

### 1.1 User and scenario

The target user is a team lead, project coordinator, or colleague who needs to recover decisions and next steps from an informal conversation without rereading the entire transcript. The intended input is a text dialogue with speaker labels, for example lines in the form `Name: utterance`. The intended output is a short human-readable summary plus structured meeting minutes that can be reviewed and copied into a project workflow.

The current prototype is designed for SAMSum-style conversations and experimentation. It does not assume that an arbitrary chat is a formal meeting. A message such as “I’ll send the draft tomorrow” should be distinguished from background discussion, and a question that was not resolved should remain an open question rather than becoming an invented decision.

### 1.2 Why use a generative language model?

Conversation summarization requires resolving references across turns, combining related statements, and expressing the result in new wording. Extractive rules could select sentences, but would often preserve conversational fragments and miss relationships. A pretrained language model can perform abstractive synthesis while following a task-specific format. The model is not treated as an authority: outputs are constrained and validated where possible, and users should check them against the source before acting.

### 1.3 Scope and non-goals

The prototype demonstrates prompt construction, autoregressive generation, decoding controls, structured output, tool interfaces, and evaluation design. It does not train a large language model, verify every generated fact, provide legal or compliance-grade minutes, or connect to live calendars and task trackers. Any external action would require a separately designed integration, access controls, and human confirmation.

## 2. Dataset Description and Preparation

### 2.1 Dataset

The project selects the approved D1 option, SAMSum, a human-annotated dialogue summarization dataset. Each record used here contains an `id`, a `dialogue`, and a `summary`. Dialogue text is the model input; the dataset summary is the reference for automatic comparison. Speaker labels are often embedded in the dialogue text rather than supplied as a separate participant field.

The project loads the Hugging Face dataset identifier `knkarthick/samsum`. The exact repository revision and dataset library version were not recorded in the prepared sample files; they should be pinned for a fully reproducible final run.

### 2.2 Prepared samples and measurements

The following values were computed directly from the CSVs currently in `data/`:

| Prepared file | Dataset split and intended use | Records | Bucket distribution |
|---|---|---:|---|
| `eval_samples.csv` | Test split; held-out evaluation | 120 | 40 short, 40 medium, 40 long |
| `dev_samples.csv` | Validation split; prompt exploration | 30 | 10 short, 10 medium, 10 long |
| `fewshot_pool.csv` | Train split; few-shot examples | 300 | 100 short, 100 medium, 100 long |

Buckets are derived from newline count plus one, used as an approximate turn count: short is 1–8 turns, medium is 9–15, and long is 16 or more. The sampling code uses `random_state=42` within each bucket. The prepared files have no missing `dialogue` or `summary` values in these sampled records.

For the 120-record evaluation sample, dialogue length averages 114.69 whitespace-delimited words (median 94; range 18–424). Reference summaries average 21.37 words (median 20; range 6–54). These are descriptive statistics for the prepared sample, not the full SAMSum corpus. Dialogue and summary lengths are measured by whitespace tokenization, not by the model tokenizer.

### 2.3 Representative example

One record from the prepared test sample illustrates the task:

**Dialogue**

> Jacob: Emily!! stop the busses .. I am not on my way..  
> Emily: Hurry up!! the busses are about to leave..  
> Jacob: Count to 10 and i will reach the parking lot  
> Emily: 10..9..1 where are you...

**Dataset reference summary**

> Emily is waiting for Jacob while the buses have almost left.

This example shows that a useful summary may merge several turns into a single concise sentence. The reference is dataset-authored; it is not a model output produced by this project.

### 2.4 Preprocessing and split hygiene

Preprocessing copies the dataset frames, adds the approximate turn count and bucket, and samples each split independently. Few-shot demonstrations are sourced from the training split, while the validation split is available for prompt development and the test split should remain reserved for final evaluation. The evaluation runner should be configured so prompt selection is completed on validation data before test metrics are examined. The current prompt comparison script reads `eval_samples.csv`; using those test outputs to make prompt choices would compromise the intended held-out test protocol. For final results, switch prompt selection to `dev_samples.csv` and run the chosen configuration once on test data.

## 3. Generative-Model Families and Model Choice

Generative models learn a distribution from examples, but their representations, objectives, and suitable output types differ.

| Family | Core idea | Typical strength | Fit for this project |
|---|---|---|---|
| Generative adversarial networks (GANs) | A generator learns to create samples that a discriminator cannot distinguish from real examples. | Realistic synthetic images and other continuous or media-like data. | Poor fit for producing coherent, controllable meeting minutes in natural language. |
| Variational autoencoders (VAEs) | An encoder maps input into a probabilistic latent representation; a decoder reconstructs or generates samples from it. | Structured latent spaces and probabilistic reconstruction/generation. | Could model text, but a general-purpose instruction interface and strong dialogue summarization are not the primary design advantage here. |
| Autoregressive language models | Predict the next token conditioned on preceding context, repeatedly generating a sequence. | Contextual text generation and instruction-following workflows. | Strongest fit for dialogue-to-summary text and schema-directed output. |
| Diffusion models | Iteratively denoise a sample from noise toward a learned data distribution. | High-quality image, audio, and other iterative-generation tasks. | Not a natural fit for producing token-by-token summaries and tool arguments in this application. |

The implementation uses `Qwen/Qwen2.5-1.5B-Instruct`, loaded through Hugging Face Transformers. This is a pretrained causal language model with an instruction-tuned chat interface; the project does not train it. The model choice is a practical baseline for experimentation, not a claim that it is optimal. Final work should record the precise model revision, Transformers/PyTorch versions, hardware, runtime, and any quantization or device settings.

## 4. LLM Architecture and Generation in MinuteMate

1. **Tokenization:** The chat messages and conversation are converted into model-specific token IDs. Tokens may be words, word pieces, punctuation, or special chat markers. Long source dialogues consume more context tokens than short ones.
2. **Embeddings:** Token IDs are mapped to learned vector representations, with positional information enabling the model to represent order.
3. **Transformer and attention:** Transformer blocks use causal self-attention so each position can use earlier context without seeing future generated tokens. This supports relating a decision in one turn to its owner or clarification in another.
4. **Language-modelling head:** The final hidden representation is projected to vocabulary logits, representing scores for possible next tokens.
5. **Autoregressive generation:** The model repeatedly selects a next token, appends it to the prefix, and predicts again until an end token or `max_new_tokens` limit. The generation helper applies the tokenizer’s chat template and returns only generated tokens.
6. **Decoding:** Greedy decoding chooses the highest-scoring next token each step. Sampling modifies the next-token distribution; temperature adjusts its sharpness, while top-k or top-p restricts the candidate set. Different settings can affect both wording and factual stability.
7. **Post-processing:** Structured responses are extracted as JSON and validated. Plain summaries can be compared with dataset references using ROUGE. Tool proposals are derived from validated minutes; the current simulation is local and opt-in in the integrated pipeline.

This architecture enables coherent synthesis, but fluency is not evidence that an output is factually supported. Every result remains subject to source review.

## 5. Proposed Application Architecture

```text
SAMSum / user dialogue
        |
        v
Preprocess and create prompt
        |
        v
Tokenizer chat template -> pretrained causal LM
        |
        v
Greedy or sampling decoder
        |
        +--------------------------+
        |                          |
        v                          v
Plain summary                JSON extraction
        |                          |
ROUGE and length metrics     Pydantic validation
                                   |
                                   v
                        Proposed task / follow-up calls
                                   |
                                   v
                     Optional local in-memory simulation
```

The system has four main components:

- **Prompt templates:** reusable functions implement zero-shot, few-shot, role-based, constrained, and detailed summaries.
- **Generation engine:** loads the model on CUDA, Apple MPS when available, or CPU, and runs autoregressive inference.
- **Structured extraction:** requests machine-readable minutes, parses a JSON object, validates the schema, and records parse/validation failures with a bounded retry count.
- **Tool router and tools:** proposes task creation for action items and a follow-up event for open questions. Tool argument models define the inputs and responses are recorded in in-memory dictionaries.

## 6. Baseline LLM Implementation

The baseline uses a simple user instruction followed by the conversation and a `Summary:` cue:

```text
Summarize the following conversation:
{dialogue}

Summary:
```

The model is loaded once for the experiment and uses greedy decoding (`do_sample=False`). The baseline script saves the example ID, length bucket, prompt identifier, decoding label, generated text, gold reference, and elapsed seconds. The configured baseline runner targets all 120 prepared test records and periodically writes intermediate results.

**Current result status:** `experiments/e1_baseline.csv` is 0 bytes. The code defines the baseline, but this workspace does not contain completed baseline outputs. No sample model response, throughput value, or quality score is asserted in this report.

## 7. Prompt-Engineering Experiments

The reusable prompt functions support five variants. Comparing all five exceeds the minimum of four prompt variants.

| ID | Prompt strategy | Key intervention |
|---|---|---|
| `p1_zero_shot` | Zero-shot | Basic instruction to summarize the conversation. |
| `p2_few_shot` | Few-shot | Includes three conversation/reference-summary demonstrations selected from the matching training length bucket. |
| `p3_role` | Role/context | Frames the model as a careful minute taker writing for a manager. |
| `p4_constrained` | Constraint-based / concise | Requests a professional summary in at most 50 words and prioritizes decisions, actions, and important points. |
| `p5_detailed` | Detailed | Requests a detailed account of decisions, actions, and important points. |

The role prompt also tells the model to treat the conversation as untrusted source data and not invent people, decisions, owners, deadlines, or actions. It asks for clear professional language and de-emphasizes trivial details. The prompt comparison script uses greedy decoding to reduce decoding variability while prompts are compared.

The project-specific expectation is to compare concise and detailed prompting. That comparison is represented by `p4_constrained` and `p5_detailed`. A final comparison table should report at least ROUGE-1/2/L, average output words, unsupported-name flags, and a short human review for each variant. Keep test data out of prompt selection: run prompt development on validation records, freeze the selected prompt, and then evaluate on the test sample.

**Current result status:** `experiments/e2_prompt_comparision.csv` is 0 bytes; no prompt winner is established. Note that the current script uses the evaluation/test CSV rather than the validation CSV, so the final reported comparison should be rerun under the split-hygiene procedure above.

## 8. Decoding Experiments

The decoding script compares the following configurations using the same constrained prompt:

| Label | Sampling | Temperature | Top-p | Top-k | Repeats |
|---|---|---:|---:|---:|---:|
| `greedy` | No | — | — | — | 1 |
| `t0.3` | Yes | 0.3 | 1.0 | 0 (disabled) | 3 |
| `t0.7` | Yes | 0.7 | 1.0 | 0 (disabled) | 3 |
| `t1.0` | Yes | 1.0 | 1.0 | 0 (disabled) | 3 |
| `top0.7_topp0.9` | Yes | 0.7 | 0.9 | library default | 3 |
| `top0.7_topk50` | Yes | 0.7 | library default | 50 | 3 |

For decoding, the current script selects 40 examples from the prepared evaluation sample and sets Torch seeds for sampled repetitions. The experiment output is intended to show both quality and variability: compare reference overlap and length, and inspect how much repeated outputs differ for the same dialogue. A useful qualitative review should check whether sampling adds useful coverage or instead changes names, decisions, or action-item assignments.

**Current result status:** `experiments/e3_decoding.csv` is 0 bytes. The implementation script writes to `e3_decoding_comparison.csv`; both expected evidence paths are currently empty or absent. No decoding-quality or variability conclusion can yet be drawn.

## 9. Adaptation Strategy: Prompting, RAG, and PEFT

| Strategy | What it changes | Benefits here | Costs / limitations | Decision |
|---|---|---|---|---|
| Prompting | Instructions, examples, formatting constraints, and decoding inputs. | Fast to iterate; no training; directly expresses concise/detailed styles and output schema. | Sensitive to wording; does not guarantee factuality or valid JSON. | **Selected for the prototype.** |
| Retrieval-augmented generation (RAG) | Retrieves external passages and adds them to model context. | Useful if minutes must apply current policies, project glossaries, or reference documents. | Adds retrieval, access control, relevance, and citation requirements; unnecessary for self-contained transcript summarization. | Not needed for the core SAMSum task. Consider only for a defined external-knowledge use case. |
| Fine-tuning / PEFT | Updates model parameters or a parameter-efficient adapter using task examples. | Could improve a stable domain style or structured format after sufficient governed data and evaluation exist. | Requires curated data, compute, model/version governance, and risk of learning dataset artifacts; not justified by this prototype alone. | Deferred; actual adaptation is not required for this report’s implementation. |

Prompting is the simplest adaptation that addresses the present task. Fine-tuning should be considered only after the prompts have been evaluated and a persistent failure pattern justifies the additional cost and governance burden.

## 10. Structured Output and Tool / Function Calling

### 10.1 Output schema

`MeetingMinutes` validates these fields:

| Field | Type | Purpose |
|---|---|---|
| `participants` | list of strings | Identified speakers; checked against labels parsed from the dialogue when validation context is provided. |
| `summary` | non-empty string | Short overview of the conversation. |
| `key_points` | list of strings | Important topics and facts. |
| `decision` | list of strings | Explicit decisions. |
| `action_items` | list of `{owner, task, due}` objects | Candidate next steps; task has a minimum length. |
| `open_questions` | list of strings | Unresolved questions. |
| `sentiment` | `positive`, `neutral`, or `negative` | Coarse conversation-level label. |

The schema constrains format and restricts participants to parsed speakers, but it does **not** prove that an action item or decision is grounded in the dialogue. The included safety helper can check exact substring support for an action text, but full semantic grounding, source-message citations, and confidence estimates are not wired into the main extraction validator. These remain important improvements before real-world use.

Illustrative JSON for the small example “Alex: I will send the revised agenda by Friday. Priya: That works.” (not a model-generated experiment result):

```json
{
  "participants": ["Alex", "Priya"],
  "summary": "Alex will send the revised agenda by Friday, and Priya agrees.",
  "key_points": ["The revised agenda needs to be sent."],
  "decision": [],
  "action_items": [
    {"owner": "Alex", "task": "Send the revised agenda", "due": "by Friday"}
  ],
  "open_questions": [],
  "sentiment": "positive"
}
```

### 10.2 Tool interface

Two Pydantic argument schemas are implemented:

| Tool | Inputs | Simulated result |
|---|---|---|
| `create_task` | `owner: str`, `task: str`, optional `due: str` | Task ID, `created` status, and the accepted input values. |
| `schedule_event` | `title: str`, `participants: list[str]`, `when: str` | Event ID, `scheduled` status, and the accepted input values. |

The router proposes a `create_task` call for each extracted action item. If open questions exist, it proposes a follow-up event with the listed participants and `when="TBD"`. A representative simulated invocation is:

```python
run_tool(
    "create_task",
    {"owner": "Alex", "task": "Send the revised agenda", "due": "by Friday"},
)
```

Representative response shape:

```json
{
  "task_id": "TSK-0001",
  "status": "created",
  "owner": "Alex",
  "task": "Send the revised agenda",
  "due": "by Friday"
}
```

The ID is illustrative; actual IDs depend on in-memory process state. The supplied tool implementation does not contact an external service. The integrated pipeline defaults to not executing tools; the E5 experiment script deliberately invokes the local simulation to demonstrate the interface. In production, require explicit user confirmation, authorization checks, idempotency, audit logging, and validated dates before any external side effect.

## 11. Programmatic Prompting and Integrated Workflow

Prompts are parameterized functions that accept a dialogue, and the few-shot function accepts training examples as tuples or dialogue/summary mappings. A typical generation call is:

```python
from src.minutes.engine import load_model, generate_text
from src.minutes.prompts import SYSTEM_ROLE, constrained

model, tokenizer = load_model("Qwen/Qwen2.5-1.5B-Instruct")
answer = generate_text(
    model,
    tokenizer,
    user_text=constrained(dialogue),
    system_prompt=SYSTEM_ROLE,
    max_new_tokens=128,
    do_sample=False,
)
```

`run_pipeline` integrates JSON extraction, validation, tool proposal, and optional simulation. It rejects empty dialogue and mismatched model/tokenizer arguments, returns validation logs on failure, and keeps tool execution off by default. An equivalent executable notebook is available at `notebooks/MinuteMate.ipynb`; `scripts/build_notebook.py` can rebuild it. Model inference requires the selected model and its dependencies to be available.

## 12. Evaluation Design and Results

### 12.1 Criteria

The planned evaluation uses more than four relevant criteria:

1. **Content overlap:** ROUGE-1, ROUGE-2, and ROUGE-L F-measures against SAMSum references.
2. **Concision:** generated summary word count, compared with the reference and the concise prompt’s 50-word target.
3. **Unsupported-name heuristic:** capitalized output words not found in the dialogue, treated as a flag for manual review and not a definitive hallucination detector.
4. **Structured reliability:** JSON parse rate, schema validation rate, retries, and validation error categories.
5. **Tool behavior:** argument-schema validity, safe behavior on invalid arguments, and whether proposed calls correspond to validated action items/open questions.
6. **Human error review:** manual review of omissions, unsupported facts, incorrect ownership/deadlines, and misleading decisions.

ROUGE measures surface overlap and can penalize valid paraphrases; it is not a factuality metric. The name heuristic can flag ordinary capitalized words and miss names that are not capitalized. These measures must be interpreted together with targeted human review.

### 12.2 Result availability

| Experiment | Planned evidence | Current status |
|---|---|---|
| E1 baseline | At least 3 outputs, ideally all 120 test sample predictions; ROUGE and runtime. | Not measured here; `e1_baseline.csv` is empty. |
| E2 prompts | Five prompt variants, comparison table, concise versus detailed analysis. | Not measured here; `e2_prompt_comparision.csv` is empty. |
| E3 decoding | Greedy and temperature/top-p/top-k examples; repeat variability. | Not measured here; `e3_decoding.csv` is empty. |
| E4 structured output | JSON examples, parse/schema validity and retry counts. | Not measured here; `e4_structured.csv` is empty. |
| E5 tools | Input schema and representative local invocation/results. | Implementation exists; saved output `e5_tool_calls.csv` is empty. The illustrative invocation above is not a recorded run. |
| E6 failure analysis | At least three reviewed failure cases and their categories. | Not measured here; `e6_failure_cases.csv` is empty. |
| Integrated workflow | Notebook or equivalent executable demonstration. | Implementation is present; full model inference has not been verified for this report. |

No aggregate model scores are included because the relevant saved outputs have no records. Once runs complete, report mean and dispersion by prompt/decoding configuration, include sample counts, and preserve representative outputs with IDs. Do not select the best prompt on the test set.

### 12.3 Failure-analysis procedure

For at least three examples, inspect transcript, reference, generated summary, structured output, and any proposed tool calls side by side. Categorize each failure using one or more labels:

| Failure type | Review question | Practical control |
|---|---|---|
| Omission | Was a key decision or action left out? | Detailed prompt comparison; add decision/action checklist. |
| Unsupported addition | Did the output introduce a fact absent from the source? | Quote evidence for action items; human review before use. |
| Wrong attribution | Was an action assigned to the wrong person? | Validate participants and owner evidence; leave owner null when uncertain. |
| Deadline error | Was a date invented or normalized incorrectly? | Preserve source wording; do not infer dates without a deterministic parser and confirmation. |
| Open-question conversion | Was an unresolved topic presented as a decision? | Keep decisions and open questions as separate schema fields. |
| Invalid structure | Was JSON malformed or inconsistent with the schema? | Parse/validate, bounded retry, report failure rather than assuming success. |
| Prompt injection | Did content inside the transcript alter model behavior? | Treat transcript as untrusted data; do not execute transcript instructions. |

The `evaluate.py` workflow can generate low-overlap examples and possible unsupported-name flags once experiment outputs exist. These automated signals should select cases for review, not replace the human analysis.

## 13. Responsible Use and Risk Controls

- **Hallucination and attribution:** Fluent summaries can invent participants, decisions, deadlines, or tasks. Keep the source transcript available, validate participant names against parsed speakers, represent uncertain owners as null, and require human approval before acting.
- **Prompt injection:** A transcript may contain text that looks like an instruction to the model. Separate system and user messages, tell the model to treat conversation text as data, and never let transcript instructions directly authorize a tool call.
- **Privacy and confidentiality:** Conversations may include sensitive personal or business data. Collect only what is needed, obtain permission, restrict access, redact PII where feasible, and avoid sending confidential records to unapproved services. Review model-provider retention and processing terms before deployment.
- **Data leakage:** Keep few-shot demonstrations in the training split, prompt development on validation, and final comparison on a held-out test sample. Record dataset revisions and sampling seeds.
- **Tool side effects:** The supplied tools are in-memory demonstrations. Any live task/calendar integration would require explicit confirmation, authentication, authorization, duplicate prevention, input validation, and an audit trail.
- **Reliability and bias:** Summaries can omit nuance, dialect, disagreement, or minority viewpoints. Evaluate across dialogue lengths and conversation styles and retain an option to inspect the original transcript.
- **Misuse:** Do not present generated minutes as a verbatim record or use sentiment as a performance judgment about individuals. Clearly label outputs as machine-generated drafts.

## 14. Limitations

1. The dataset is short, informal dialogue and may not represent long meetings, domain-specific terminology, multilingual conversation, or real organizational records.
2. The current experiment output files are empty, so no empirical model quality, latency, or decoding conclusions are available yet.
3. The prepared test sample is balanced by approximate turn count rather than selected to represent all deployment distributions.
4. The current E2 script reads test examples while comparing prompts; final prompt selection should move to validation to avoid test-set tuning.
5. ROUGE is reference-overlap based; the name detector is a weak heuristic. Neither proves factuality or completeness.
6. Pydantic checks output types and the participant list, but does not semantically prove that every action, decision, due date, or point appears in the source.
7. `speakers_of` assumes a colon-delimited speaker format. It can misinterpret colons in other transcript formats.
8. The follow-up scheduling tool uses a placeholder date (`TBD`), and the tool store is process-local, non-persistent, and not connected to a calendar/task service.
9. Model behavior can vary with model revision, decoding settings, hardware, library versions, and prompt changes. These should be recorded with completed experimental results.

## 15. Conclusion and Future Work

MinuteMate implements a complete prototype path from dialogue input to generated summary or validated structured minutes, with reusable prompts, controlled decoding, evaluation helpers, and local tool simulations. The SAMSum subset is prepared with separated training, validation, and test files and balanced dialogue-length buckets. The design selects prompting over retrieval or fine-tuning because the core task is self-contained summarization and prompt iteration is the least costly adaptation.

The central remaining task is evidence collection. Run the baseline and comparisons, move prompt development to validation, freeze the chosen configuration, evaluate on test data, populate the result and failure-analysis tables, and record execution environment/model revision. Then review at least three concrete failure cases and add source-grounded action-item verification before considering real external tools. The resulting report should make clear which findings are measured and which are design decisions.

## References

1. Gliwa, B., Mochol, I., Biesek, M., and Wawer, A. (2019). “SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization.” *Proceedings of the 2nd Workshop on New Frontiers in Summarization*, 70–79. https://doi.org/10.18653/v1/D19-5409
2. Hugging Face Datasets. “knkarthick/samsum” dataset repository. https://huggingface.co/datasets/knkarthick/samsum
3. Qwen Team. “Qwen2.5-1.5B-Instruct” model card. https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
4. Hugging Face. *Transformers documentation: Text generation strategies and causal language models.* https://huggingface.co/docs/transformers/
5. Pydantic. *Pydantic documentation: Models and validation context.* https://docs.pydantic.dev/
6. Lin, C.-Y. (2004). “ROUGE: A Package for Automatic Evaluation of Summaries.” *Text Summarization Branches Out*, 74–81. https://aclanthology.org/W04-1013/

## Appendix A. Reproducibility and Commands

Run commands from the repository root in an environment with the project dependencies installed. Dataset preparation downloads/loads the dataset; model experiments may download model weights and require substantial memory.

```powershell
python -m scripts.explore_data
python -m scripts.prepare_data
python -m scripts.run_e1_baseline
python -m scripts.run_e2_prompts
python -m scripts.run_e3_decoding
python -m scripts.run_e4_structured
python -m scripts.run_e5_tools
python -m scripts.evaluate
python -m scripts.build_notebook
```

The `scripts/run_all.sh` orchestration file provides the corresponding sequential run in a Bash environment. Save model ID/revision, Python and library versions, operating system, device, seeds, sample counts, and command-line options alongside final results.

## Appendix B. Prompt and Schema Samples

**Concise instruction**

```text
Write a concise summary of the following conversation in at most 50 words,
highlighting key decisions, action items, and important points. Avoid including
trivial details or off-topic discussions. Use clear and professional language.
```

**Detailed instruction**

```text
Write a detailed summary of the following conversation, including key decisions,
action items, and important points. Avoid including trivial details or off-topic
discussions. Use clear and professional language.
```

**Structured-output shape**

```json
{
  "participants": [],
  "summary": "One to three sentences summarizing the conversation",
  "key_points": [],
  "decision": [],
  "action_items": [
    {"owner": null, "task": "A task explicitly agreed in the conversation", "due": null}
  ],
  "open_questions": [],
  "sentiment": "positive, neutral, or negative"
}
```

## Appendix C. Final Results To Add Before Submission

- [ ] E1: At least three model outputs and a measured baseline score.
- [ ] E2: Validation-selected prompt variants, final held-out comparison table, concise-versus-detailed analysis.
- [ ] E3: Greedy and sampling examples, score/length comparison, repeated-output variability.
- [ ] E4: Valid and invalid JSON examples; schema success rate and retry/error counts.
- [ ] E5: Recorded tool schema and representative invocation/result from the implemented local simulation.
- [ ] E6: At least three manually inspected failure cases with transcript evidence and corrective action.
- [ ] Environment, model revision, library versions, seeds, and run timestamps.
- [ ] Group/member details and instructor dataset registration confirmation.
