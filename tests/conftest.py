import pytest
from fastapi.testclient import TestClient

from triage.api import app, get_service
from triage.fallback import FallbackClassifier
from triage.llm import LLMClassifier
from triage.service import TriageService

GOOD = {"category": "billing", "priority": "high", "summary": "Customer charged twice", "confidence": 0.93}


class FakeClient:
    """Returns queued responses in order; an Exception instance is raised instead of returned."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def triage(self, system, user, schema):
        self.calls.append(user)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(scope="session")
def fallback() -> FallbackClassifier:
    return FallbackClassifier()


@pytest.fixture
def make_client(fallback):
    def _make(*responses):
        llm = LLMClassifier(FakeClient(*responses)) if responses else None
        service = TriageService(fallback, llm)
        app.dependency_overrides[get_service] = lambda: service
        return TestClient(app), service

    yield _make
    app.dependency_overrides.clear()
