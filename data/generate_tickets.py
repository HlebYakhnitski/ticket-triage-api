"""Generate a small labelled ticket set from templates (deterministic).

Templated data is far more regular than real tickets, so accuracy measured on it is
an upper bound. It exists to make the fallback model and tests reproducible; replace
it with real labelled tickets before trusting the numbers.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

TEMPLATES = {
    "billing": [
        ("Charged twice this month", "I see two charges of {amount} on my card for the same {period}."),
        ("Need a refund", "We cancelled last week but were still billed {amount}. Can we get a refund?"),
        ("Invoice for our subscription", "Our accountant needs a VAT invoice for the {period} plan payment."),
        ("Wrong amount on my bill", "My plan is {plan} but the last bill was {amount}, which looks too high."),
        ("Update payment card", "Our company card expired, how do we change the card you charge?"),
        ("Downgrade pricing question", "If we downgrade from {plan} mid-cycle, do we get credit for the rest?"),
    ],
    "bug": [
        ("PDF export broken", "When I export an invoice to PDF the {part} is missing. Started {when}."),
        ("Error when saving", "Saving a draft shows 'something went wrong' and nothing is saved. {when}."),
        ("Totals calculated wrong", "Line items add up to {amount} but the invoice total shows a different number."),
        ("Page keeps loading", "The {page} page spins forever in {browser}. Other pages work."),
        ("Emails not sent", "Clients say they never receive invoice emails since {when}. Status says sent."),
        ("Duplicate numbers", "Two invoices got the same number after the {when} update, which breaks our records."),
    ],
    "how_to": [
        ("How do I add tax", "How can I add {tax} to invoices for clients in {country}?"),
        ("Recurring invoices", "Is there a way to send the same invoice every {period} automatically?"),
        ("Change invoice template", "Where do I change the logo and colours on my invoice template?"),
        ("Partial payment question", "A client paid half. How do I record a partial payment?"),
        ("Export for accountant", "What is the best way to export all {period} invoices for my accountant?"),
        ("Add a team member", "How do I give my bookkeeper access to the {page} page only?"),
    ],
    "account_access": [
        ("Locked out", "I cannot log in, it says my account is locked after too many attempts."),
        ("Lost my phone 2FA", "I lost the phone with my authenticator app and cannot sign in."),
        ("Password reset email missing", "I requested a password reset {when} but no email arrived."),
        ("Can't access account", "Our owner left the company and nobody can access the account now."),
        ("Login loop", "After entering my password in {browser} it sends me back to the login page."),
        ("Change account email", "I need to change the email I use to sign in because I lost access to the old one."),
    ],
    "feature_request": [
        ("Suggestion: dark mode", "Would be nice to have a dark mode for working late."),
        ("Feature idea", "Please add support for {country} e-invoicing standard, we would pay more for it."),
        ("Integration request", "Any plans to integrate with {tool}? It would save us hours each {period}."),
        ("Bulk edit wish", "It would help a lot to edit due dates for many invoices at once."),
        ("API request", "Could you add a webhook when a client opens an invoice for the first time?"),
        ("Custom fields", "We would love custom fields on clients, like a purchase order number."),
    ],
    "other": [
        ("Partnership", "We are an agency and would like to discuss reselling your product."),
        ("Thanks", "Just wanted to say the new {page} page is great, thank you!"),
        ("Data residency question", "Where is our data stored? Our legal team is asking about {country}."),
        ("Job application", "Are you hiring support engineers in {country}?"),
        ("Press inquiry", "I am writing an article about invoicing tools and would like a quote."),
        ("Security questionnaire", "Our procurement team needs you to fill in a vendor security questionnaire."),
    ],
}
SLOTS = {
    "amount": ["$49", "$99", "120 EUR", "$1,200", "35 GBP"],
    "period": ["month", "quarter", "year", "week"],
    "plan": ["Starter", "Pro", "Business"],
    "part": ["logo", "tax line", "footer", "client address"],
    "when": ["yesterday", "this morning", "since Monday", "after the last update", "two days ago"],
    "page": ["reports", "clients", "expenses", "invoices"],
    "browser": ["Chrome", "Firefox", "Safari", "Edge"],
    "tax": ["VAT", "sales tax", "GST"],
    "country": ["Germany", "Poland", "Canada", "Italy", "Spain"],
    "tool": ["Xero", "QuickBooks", "Slack", "Zapier"],
}


def fill(text: str, rng: random.Random) -> str:
    for key, options in SLOTS.items():
        text = text.replace("{" + key + "}", rng.choice(options))
    return text


def main(per_template: int = 6) -> None:
    rng = random.Random(42)
    rows = []
    for category, templates in TEMPLATES.items():
        for subject, body in templates:
            for _ in range(per_template):
                rows.append({"subject": fill(subject, rng), "body": fill(body, rng), "category": category})
    rng.shuffle(rows)
    out = Path(__file__).with_name("tickets.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} tickets to {out}")


if __name__ == "__main__":
    main()
