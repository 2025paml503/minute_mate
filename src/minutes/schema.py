from typing import Literal
from xml.parsers.expat import model
from pydantic import BaseModel, Field, field_validator, ValidationInfo

class ActionItem(BaseModel):
    owner: str | None = None
    task: str | Field(min_length=3)
    due: str | None = None

class MeetingMinuetes(BaseModel):
    participants: list[str]
    summary: str
    key_points: list[str] = []
    decision: list[str] = []
    action_items: list[ActionItem] = []
    open_questions: list[str] = []
    sentiment: Literal["positive", "neutral", "negative"] = "neutral"

    @field_validator("participants")
    @classmethod
    def names_must_be_speakers(cls, names, info: ValidationInfo):
        allowed = (info.context or {}).get("speakers")
        if allowed is not None:
            bad = [n for n in names if n not in allowed]
            if bad:
                raise ValueError(f"participants must be in speakers: {bad}")
        return names

SCHEMA_TEXT = """{
  "participants": ["Alice", ...],
  "summary": "1-3 sentences summarizing the meeting",
  "key_points": ["short bullet points of key points", ...],
  "decision": ["short bullet points of decisions made", ...],
  "action_items": [{"owner": "name or null", "task": "what to do", "due": "words from chat or null"}],
  "open_questions": ["unanswered questions from the meeting", ...],
  "sentiment": "positive|neutral|negative"
}"""