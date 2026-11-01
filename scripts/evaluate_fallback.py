"""Evaluate the fallback classifier two ways, to show why the split matters.

A random split puts sentences generated from the same template in both train and test,
so the model recognises phrasing instead of understanding the ticket. Holding out whole
templates measures what happens on wording the model has never seen.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from generate_tickets import TEMPLATES, fill  # noqa: E402

from triage.fallback import load_examples, train  # noqa: E402


def random_split_accuracy() -> float:
    texts, labels = load_examples()
    model = train(texts[:2], labels[:2])  # any fitted pipeline; cross_val_score clones it
    return float(cross_val_score(model, texts, labels, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean())


def leave_one_template_out_accuracy(per_template: int = 6, seed: int = 1) -> float:
    rng = random.Random(seed)
    items = [(cat, i, s, b) for cat, temps in TEMPLATES.items() for i, (s, b) in enumerate(temps)]
    correct = total = 0
    for cat, i, subject, body in items:
        train_x, train_y = [], []
        for cat2, j, s2, b2 in items:
            if (cat2, j) == (cat, i):
                continue
            for _ in range(per_template):
                train_x.append(f"{fill(s2, rng)} {fill(b2, rng)}")
                train_y.append(cat2)
        model = train(train_x, train_y)
        predictions = model.predict([f"{fill(subject, rng)} {fill(body, rng)}" for _ in range(per_template)])
        correct += sum(p == cat for p in predictions)
        total += per_template
    return correct / total


if __name__ == "__main__":
    chance = 1 / len(TEMPLATES)
    print(f"random 5-fold split:          {random_split_accuracy():.3f}")
    print(f"leave-one-template-out:       {leave_one_template_out_accuracy():.3f}")
    print(f"chance ({len(TEMPLATES)} categories):        {chance:.3f}")
