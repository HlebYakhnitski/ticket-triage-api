"""Run the real Anthropic SDK against a mocked HTTP transport - no network, no API key."""
import json

import anthropic
import httpx2 as httpx  # anthropic 2.x ships its own fork of httpx
import pytest

from triage.llm import TOOL_NAME, AnthropicJSONClient, LLMClassifier, LLMError
from triage.schemas import LLMTriage, TicketIn


def sdk_with(content_blocks, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": "test-model",
            "content": content_blocks, "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

    return anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_request_forces_the_triage_tool_and_parses_input():
    seen = []
    answer = {"category": "bug", "priority": "normal", "summary": "PDF export misses logo", "confidence": 0.8}
    sdk = sdk_with([{"type": "tool_use", "id": "tu_1", "name": TOOL_NAME, "input": answer}], seen)
    client = AnthropicJSONClient(model="test-model", sdk_client=sdk)

    result = LLMClassifier(client).classify(TicketIn(subject="PDF export", body="logo missing"))

    assert result == LLMTriage(**answer)
    request = seen[0]
    assert request["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    assert request["tools"][0]["input_schema"]["properties"]["category"]
    assert request["model"] == "test-model"


def test_missing_tool_call_is_an_error():
    sdk = sdk_with([{"type": "text", "text": "I think it is billing"}], [])
    with pytest.raises(LLMError):
        LLMClassifier(AnthropicJSONClient(sdk_client=sdk)).classify(TicketIn(subject="a", body="b"))
