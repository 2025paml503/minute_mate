import json, re
from pydantic import ValidationError
from .schema import MeetingMinuetes, SCHEMA_TEXT
from .engine import load_model, generate_text

SYSTEM_JSON = ("You are a helpful assistant that extracts structured meeting minutes from a transcript, a careful minute taker. You will be given a meeting transcript, and you will output a JSON object that conforms to the following schema:\n\n" + SCHEMA_TEXT)

def json_prompt(dialogue):
    return f"Produce minutes for this conversation using exactly this Json format:\n\n{SCHEMA_TEXT}\n\nConversation:\n{dialogue}\n\nMinutes in JSON:"

def extract_json(text):
    """Find the first {...} block in the text, fix trailing commas, and return it as a dict. Return None if no JSON block is found."""
    text = re.sub(r",\s*}", "}", text)
    start, depth = text.find("{"), 0
    if start == -1:
        return None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                break
    else:
        return None
    candidate = re.sub(r",\s*}", "}", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

def speakers_of(dialogue):
    """Return a set of speaker names from the dialogue, assuming each line starts with 'Speaker: '."""
    speakers = set()
    for line in dialogue.splitlines():
        if ':' in line:
            speaker = line.split(':', 1)[0].strip()
            if speaker:
                speakers.add(speaker)
    return speakers

def get_minutes(model, tokenizer, dialogue, max_retries=1):
    log = {"attempts": 0, "errors": [], "raw": []}
    user = json_prompt(dialogue)
    for attempt in range(max_retries + 1):
        log["attempts"] += 1
        raw = generate_text(model, tokenizer, user_text=user, system_prompt=SYSTEM_JSON, max_new_tokens=256, do_sample=False)
        log["raw"].append(raw)
        data = extract_json(raw)
        if data is None:
            log["errors"].append("Failed to extract JSON from generated text")
        else:
            try:
                minutes = MeetingMinuetes.model_validate(data, context={"speakers": speakers_of(dialogue)})
                return minutes, log
            except ValidationError as e:
                log["errors"].append(f"Invalid meeting minutes format: {e}")
        user = f"Previous attempt failed. Please produce valid meeting minutes in JSON format for this conversation:\n\n{dialogue}\n\nMinutes in JSON:" 
    return None, log
