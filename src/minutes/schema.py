"""Validated machine-readable schema for extracted meeting minutes."""

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class ActionItem(BaseModel):
    """A candidate action item extracted from the conversation."""

    owner: str | None = None
    task: str = Field(min_length=3)
    due: str | None = None


class MeetingMinutes(BaseModel):
    """Structured summary grounded in the speakers and content of a transcript."""

    participants: list[str] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    key_points: list[str] = Field(default_factory=list)
    decision: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    sentiment: Literal["positive", "neutral", "negative"] = "neutral"

    @field_validator("participants")
    @classmethod
    def names_must_be_speakers(
        cls,
        names: list[str],
        info: ValidationInfo,
    ) -> list[str]:
        allowed = (info.context or {}).get("speakers")
        if allowed is not None:
            bad = [n for n in names if n not in allowed]
            if bad:
                raise ValueError(f"participants must be in speakers: {bad}")
        return names


# Preserve the spelling used by the original structured-output implementation.
MeetingMinuetes = MeetingMinutes


SCHEMA_TEXT = """{
  "participants": [],
  "summary": "One to three sentences summarizing the conversation",
  "key_points": [],
  "decision": [],
  "action_items": [
    {"owner": null, "task": "A task explicitly agreed in the conversation", "due": null}
  ],
  "open_questions": [],
  "sentiment": "positive, neutral, or negative"
}"""