"""A small local classifier used when the LLM is unavailable, slow or returns garbage."""
from __future__ import annotations

import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from triage.schemas import Category, LLMTriage, Priority, TicketIn

DATA = Path(__file__).resolve().parent.parent / "data" / "tickets.jsonl"

URGENT = re.compile(r"\b(outage|down for everyone|all users|data loss|charged (twice|again)|double charged)\b", re.I)
HIGH = re.compile(r"\b(can'?t|cannot|unable to|locked out|blocked|not working|error|failed)\b", re.I)
LOW = re.compile(r"\b(would be nice|suggestion|typo|minor|cosmetic)\b", re.I)


def load_examples(path: Path = DATA) -> tuple[list[str], list[str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [f"{r['subject']} {r['body']}" for r in rows], [r["category"] for r in rows]


def train(texts: list[str], labels: list[str]) -> Pipeline:
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        LogisticRegression(max_iter=2000, C=5.0),
    )
    return model.fit(texts, labels)


def rule_priority(text: str, tier: str) -> Priority:
    if URGENT.search(text):
        priority = Priority.URGENT
    elif HIGH.search(text):
        priority = Priority.HIGH
    elif LOW.search(text):
        priority = Priority.LOW
    else:
        priority = Priority.NORMAL
    if tier == "enterprise" and priority in (Priority.LOW, Priority.NORMAL):
        priority = Priority.HIGH if priority is Priority.NORMAL else Priority.NORMAL
    return priority


class FallbackClassifier:
    def __init__(self, model: Pipeline | None = None) -> None:
        self.model = model or train(*load_examples())

    def classify(self, ticket: TicketIn) -> LLMTriage:
        text = f"{ticket.subject} {ticket.body}"
        probs = self.model.predict_proba([text])[0]
        best = probs.argmax()
        return LLMTriage(
            category=Category(self.model.classes_[best]),
            priority=rule_priority(text, ticket.customer_tier),
            summary=ticket.subject[:200],
            confidence=round(float(probs[best]), 3),
        )
