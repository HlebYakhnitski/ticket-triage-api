from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from triage.fallback import FallbackClassifier
from triage.llm import LLMClassifier, LLMError
from triage.schemas import TicketIn, TriageResult

log = logging.getLogger(__name__)


@dataclass
class Metrics:
    requests: int = 0
    llm_ok: int = 0
    llm_failed: int = 0
    fallback: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def snapshot(self) -> dict:
        lat = np.array(self.latencies_ms[-1000:]) if self.latencies_ms else np.array([0.0])
        return {
            "requests": self.requests,
            "llm_ok": self.llm_ok,
            "llm_failed": self.llm_failed,
            "fallback": self.fallback,
            "fallback_rate": round(self.fallback / self.requests, 3) if self.requests else 0.0,
            "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
            "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        }


class TriageService:
    """LLM first; the local model whenever the LLM is not configured or fails.

    The response always says which path produced it, so downstream users and
    dashboards can tell a model decision from a fallback guess.
    """

    def __init__(self, fallback: FallbackClassifier, llm: LLMClassifier | None = None) -> None:
        self.fallback, self.llm = fallback, llm
        self.metrics = Metrics()

    def triage(self, ticket: TicketIn) -> TriageResult:
        start = time.perf_counter()
        self.metrics.requests += 1
        try:
            if self.llm is not None:
                try:
                    result = self.llm.classify(ticket)
                    self.metrics.llm_ok += 1
                    return TriageResult(**result.model_dump(), source="llm")
                except LLMError as err:
                    self.metrics.llm_failed += 1
                    log.warning("LLM triage failed, using fallback: %s", err)
                    notes = ["llm unavailable or returned invalid output; local model used"]
            else:
                notes = ["llm not configured; local model used"]
            self.metrics.fallback += 1
            result = self.fallback.classify(ticket)
            return TriageResult(**result.model_dump(), source="fallback", notes=notes)
        finally:
            self.metrics.latencies_ms.append((time.perf_counter() - start) * 1000)
