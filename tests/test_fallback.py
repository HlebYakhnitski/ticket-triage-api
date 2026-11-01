from sklearn.model_selection import train_test_split

from triage.fallback import FallbackClassifier, load_examples, rule_priority, train
from triage.schemas import Priority, TicketIn


def test_random_split_overstates_accuracy():
    """Same-template sentences leak across a random split - near-perfect scores are expected
    and meaningless. Kept as a test so the gap to the honest estimate stays visible."""
    texts, labels = load_examples()
    x_tr, x_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.25, random_state=0, stratify=labels)
    assert train(x_tr, y_tr).score(x_te, y_te) > 0.95


def test_unseen_templates_still_beat_chance():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from evaluate_fallback import leave_one_template_out_accuracy

    assert leave_one_template_out_accuracy() > 0.4    # 6 categories: chance is 0.167


def test_classifies_obvious_cases(fallback):
    assert fallback.classify(TicketIn(subject="Locked out", body="I cannot log in after 2FA reset")).category == \
        "account_access"
    assert fallback.classify(TicketIn(subject="Refund please", body="We were billed after cancelling")).category == \
        "billing"


def test_confidence_is_a_probability(fallback):
    result = fallback.classify(TicketIn(subject="hello", body="question"))
    assert 0 <= result.confidence <= 1


def test_priority_rules():
    assert rule_priority("we were double charged", "free") is Priority.URGENT
    assert rule_priority("I cannot export", "free") is Priority.HIGH
    assert rule_priority("would be nice to have dark mode", "free") is Priority.LOW
    assert rule_priority("general question", "free") is Priority.NORMAL


def test_enterprise_tier_bumps_priority():
    assert rule_priority("general question", "enterprise") is Priority.HIGH
    assert rule_priority("would be nice to have", "enterprise") is Priority.NORMAL
    assert rule_priority("outage for all users", "enterprise") is Priority.URGENT


def test_model_can_be_injected():
    model = train(["refund my money", "app crashes on save"], ["billing", "bug"])
    assert FallbackClassifier(model).classify(TicketIn(subject="refund", body="money")).category == "billing"
