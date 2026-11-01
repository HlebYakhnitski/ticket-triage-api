from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Category(str, Enum):
    BILLING = "billing"
    BUG = "bug"
    HOW_TO = "how_to"
    ACCOUNT_ACCESS = "account_access"
    FEATURE_REQUEST = "feature_request"
    OTHER = "other"


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TicketIn(BaseModel):
    subject: str = Field(min_length=1, max_length=300)
    body: str = Field(min_length=1, max_length=10_000)
    customer_tier: Literal["free", "pro", "enterprise"] = "free"

    @field_validator("subject", "body")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class LLMTriage(BaseModel):
    """The exact shape the model must return - used as the tool's input schema."""

    category: Category
    priority: Priority
    summary: str = Field(min_length=1, max_length=200, description="One sentence for the support agent")
    confidence: float = Field(ge=0, le=1)


class TriageResult(LLMTriage):
    source: Literal["llm", "fallback"]
    notes: list[str] = Field(default_factory=list)
