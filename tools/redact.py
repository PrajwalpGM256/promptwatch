import re

BODY_CAP = 2000

_TRACKING_HOSTS = (
    "tracking.icims.com",
    "link.aquent.com",
    "cdn.mcauto-images-production.sendgrid.net",
    "app.hireflix.com",
    "outlook.office.com/bookwithme",
    "recruiting2.ultipro.com",
    "jobs.smartrecruiters.com",
)

_HOST_ALTERNATION = "|".join(re.escape(host) for host in _TRACKING_HOSTS)
_PLACEHOLDER_URL = "https://example.com/t/a1b2c3"

_PATTERNS = [
    (re.compile(rf"https?://\S*?(?:{_HOST_ALTERNATION})\S*"), _PLACEHOLDER_URL),
    (re.compile(r"https?://\S{80,}"), _PLACEHOLDER_URL),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "noreply@example.com"),
    (re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "555-010-0000"),
    (re.compile(r"\bR-?\d{5,}\b"), "R000100000"),
    (re.compile(r"\b\d{4}-\d{5}\b"), "2026-50000"),
]

_INVISIBLE = dict.fromkeys(
    ord(char) for char in "‌​﻿͏­\u200E\u200F"
)


def clean(text: str) -> str:
    """Strip invisible spacer characters and collapse runaway whitespace."""
    text = text.translate(_INVISIBLE).replace(" ", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"(?:\n\s*){3,}", "\n\n", text)
    return text.strip()


def redact(text: str, extra: list[tuple[str, str]] | None = None) -> str:
    """Replace PII and tracking tokens with same-shape placeholders.

    `extra` supplies per-batch replacements such as recruiter names, applied
    before the shared patterns.
    """
    for pattern, replacement in extra or []:
        text = re.sub(pattern, replacement, text)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def cap(text: str, limit: int = BODY_CAP) -> str:
    """Truncate to `limit` characters on a word boundary with a marker."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "\n\n[...truncated]"
