"""parsing.py — extract the answer letter from a model's raw output.

Vendored and merged from training/eval_base.py:parse_letter (strict + loose
word-boundary regex, avoids false positives inside words like "Based"/"By") and
EzelinyumEvaluator/parsing.py (hedging + refusal detection). A refusal or a
genuine hedge must NOT be silently coerced into a letter — it is scored wrong,
not lucky.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Strict: an isolated A/B/C/D (bare letter, "A.", " A ").
_STRICT = re.compile(r"\b([ABCD])\b")
# Loose: a letter bounded by non-letters on both sides (catches "A:", "(A)") but
# not letters embedded in words.
_LOOSE = re.compile(r"(?:^|[^A-Za-z])([ABCD])(?=[\s\.\):,!?]|$)")

_HEDGE = re.compile(
    r"\b(both|either|neither|all of|none of|cannot|can't|unable|"
    r"not sure|unclear|ambiguous|impossible to)\b",
    re.I,
)


@dataclass
class Parsed:
    letter: str | None
    hedged: bool
    refused: bool
    raw: str


def parse_letter(text: str | None) -> str | None:
    """Bare letter extractor — the hot path used by metrics. None if no isolated
    A/B/C/D is present (scored wrong, not leniently guessed)."""
    if not text:
        return None
    upper = str(text).upper()
    m = _STRICT.search(upper)
    if m:
        return m.group(1)
    m = _LOOSE.search(upper)
    return m.group(1) if m else None


def parse_answer(text: str | None, options: dict | None = None) -> Parsed:
    """Rich parse: letter + hedging/refusal flags + fallback to option text.

    `options` (e.g. {"A": "Yes", "B": "No"}) lets us recover a letter when the
    model echoes the full option value instead of the letter.
    """
    if text is None:
        return Parsed(None, hedged=False, refused=True, raw="")

    raw = str(text).strip()
    upper = raw.upper()
    hedged = bool(_HEDGE.search(raw))

    letter = parse_letter(raw)

    # Fallback: model echoed the option VALUE rather than its letter.
    if letter is None and options:
        for L, v in options.items():
            if L not in {"A", "B", "C", "D"}:
                continue
            vu = str(v).strip().upper()
            if vu and (vu in upper or upper in vu):
                letter = L
                break

    refused = letter is None
    return Parsed(letter=letter, hedged=hedged, refused=refused, raw=raw)
