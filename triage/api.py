"""HTTP API.

    uvicorn triage.api:app --reload
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI

from triage import __version__
from triage.fallback import FallbackClassifier
from triage.llm import AnthropicJSONClient, LLMClassifier
from triage.schemas import TicketIn, TriageResult
from triage.service import TriageService

app = FastAPI(title="Ticket triage", version=__version__)


@lru_cache
def get_service() -> TriageService:
    llm = LLMClassifier(AnthropicJSONClient()) if os.environ.get("ANTHROPIC_API_KEY") else None
    return TriageService(FallbackClassifier(), llm)


Service = Annotated[TriageService, Depends(get_service)]


@app.post("/triage", response_model=TriageResult)
def triage(ticket: TicketIn, service: Service) -> TriageResult:
    return service.triage(ticket)


@app.get("/health")
def health(service: Service) -> dict:
    return {"status": "ok", "llm_configured": service.llm is not None, "version": __version__}


@app.get("/metrics")
def metrics(service: Service) -> dict:
    return service.metrics.snapshot()
