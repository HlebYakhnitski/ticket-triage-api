# ticket-triage-api

A FastAPI service that classifies incoming support tickets — category, priority, a one-line summary and a confidence score — using an LLM with schema-enforced output, validation and a corrective retry, and falling back to a local model whenever the LLM is unavailable or returns something unusable.

```bash
curl -s localhost:8000/triage -H 'content-type: application/json' -d '{
  "subject": "Charged twice this month",
  "body": "I see two charges of $49 on my card for the same month.",
  "customer_tier": "pro"
}'
```
```json
{
  "category": "billing",
  "priority": "urgent",
  "summary": "Customer was charged twice for the same month",
  "confidence": 0.94,
  "source": "llm",
  "notes": []
}
```

Every response says which path produced it (`"source": "llm"` or `"fallback"`), so a dashboard never mistakes a fallback guess for a model decision.

## How a request flows

```
POST /triage ─► validate input (Pydantic) ─► LLM configured? ──no──► local model ─► response (source: fallback)
                                                   │ yes
                                                   ▼
                                    forced tool call, schema = result model
                                                   │
                                     valid? ──yes──► response (source: llm)
                                       │ no
                                       ▼
                         retry once with the validation errors as feedback
                                       │
                                still invalid, or network/timeout error ──► local model
```

- **Structured output by construction.** The model is forced to call one tool whose input schema *is* the Pydantic result model, so the answer arrives as JSON fields, not prose to parse.
- **Validate, then retry with feedback — once.** An out-of-range confidence or an invented category is fed back to the model with the exact validation errors; models usually correct that on the second try. A timeout or auth error is not retried blindly — it goes straight to the fallback.
- **Ticket text is data.** Customer text is wrapped in `<ticket>` tags and the system prompt says to treat it as data, not instructions. A test sends "Ignore previous instructions" to check the delimiting.
- **Fallback that always answers.** TF-IDF + logistic regression for the category, keyword rules for priority, with enterprise customers bumped one level.
- **Observable.** `/metrics` reports request counts, LLM successes and failures, fallback rate, and p50/p95 latency.

## The number that matters

The fallback model is trained on a small set of tickets generated from templates (`data/generate_tickets.py`). Evaluated two ways:

| Split | Accuracy |
| --- | ---: |
| Random 5-fold | **1.000** |
| Leave one template out | **0.574** |
| Chance (6 categories) | 0.167 |

The random split is meaningless: sentences generated from the same template land in both train and test, and the model recognises phrasing rather than meaning. Holding out entire templates — wording the model has never seen — gives 57%, which is the honest estimate of how it would do on real tickets. It is why the fallback exists only as a safety net and why its responses are labelled. Run `python scripts/evaluate_fallback.py` to reproduce.

## Run it

```bash
pip install -e ".[dev]"
uvicorn triage.api:app --reload             # fallback only
ANTHROPIC_API_KEY=... uvicorn triage.api:app  # LLM first, fallback on failure
```

Or with Docker:

```bash
docker build -t ticket-triage .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... ticket-triage
```

Interactive docs at http://localhost:8000/docs. `ANTHROPIC_MODEL` overrides the default model.

## Tests

```bash
pytest
```

17 tests. The API is tested through FastAPI's `TestClient` with a fake LLM that can return valid output, invalid output or raise — covering the retry-with-feedback path, the fallback on persistent invalid output and on exceptions, input validation, delimiting of ticket text, and the metrics counters. The Anthropic integration runs the **real SDK** against a mocked HTTP transport, checking that the request forces the tool call and that a reply without one is rejected. No network or API key needed.

## Layout

```
triage/
  schemas.py    request, LLM output and response models
  llm.py        prompt, forced tool call, validation + retry
  fallback.py   local classifier and priority rules
  service.py    LLM-or-fallback orchestration and metrics
  api.py        FastAPI routes
data/           ticket generator and the generated set
scripts/        fallback evaluation
```

## License

MIT
