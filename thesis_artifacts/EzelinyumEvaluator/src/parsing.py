"""Answer parsing — robust, conservative, audit-aware.

Audit findings addressed:
- Hedging detection ("both", "either", "cannot", multiple letters) -> hedged=True
- Refusals / unparseable -> None (counted wrong, never randomly assigned)
- Per-record option space: only accept letters that are actually offered.
- Case-, whitespace-, punctuation-tolerant.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


_HEDGE_PATTERNS = [
    re.compile(r"\bboth\b", re.I),
    re.compile(r"\beither\b", re.I),
    re.compile(r"\bneither\b", re.I),
    re.compile(r"\bcannot\b", re.I),
    re.compile(r"\bunable\b", re.I),
    re.compile(r"\bnot\s+sure\b", re.I),
    re.compile(r"\bunclear\b", re.I),
    re.compile(r"\bambiguous\b", re.I),
    re.compile(r"\bnone of\b", re.I),
    re.compile(r"\ball of\b", re.I),
]

# Strict: the response IS one letter (or "(A)", "A.", "A:", "A)") possibly with whitespace.
_STRICT_RE = re.compile(r"^\s*\(?\s*([A-Z])\s*[\)\.\:]?\s*$")
# Loose: standalone letter, surrounded by word boundaries.
_LOOSE_RE = re.compile(r"\b([A-Z])\b")
# "Option A" / "Answer: A" / "The answer is A" patterns.
_LABELED_RE = re.compile(
    r"(?:option|answer|choice|choose|select)\s*[:\-]?\s*\(?([A-Z])\)?",
    re.I,
)


@dataclass
class ParsedAnswer:
    letter: Optional[str]  # the chosen letter, or None if unparseable / refused
    hedged: bool           # multiple letters mentioned, or hedging words present
    refused: bool          # explicit refusal / "cannot tell" / empty
    raw_response: str
    parse_path: str        # "strict" | "labeled" | "loose" | "none"


def parse_answer(raw: str, valid_letters: set[str]) -> ParsedAnswer:
    """Parse a free-form generation into one of `valid_letters`, or None.

    Parameters
    ----------
    raw : str
        Model's generated text.
    valid_letters : set[str]
        The set of acceptable answer letters for THIS record (e.g. {"A","B"}
        for binary topics, {"A","B","C","D"} for 4-way). Letters outside this
        set are rejected even if they appear in the text.
    """
    if raw is None:
        return ParsedAnswer(None, False, True, "", "none")

    text = raw.strip()
    if not text:
        return ParsedAnswer(None, False, True, raw, "none")

    valid = {c.upper() for c in valid_letters}

    # Strict single-letter response.
    m = _STRICT_RE.match(text)
    if m:
        letter = m.group(1).upper()
        if letter in valid:
            return ParsedAnswer(letter, False, False, raw, "strict")

    # "Option A" / "answer is A" pattern.
    m = _LABELED_RE.search(text)
    if m:
        letter = m.group(1).upper()
        if letter in valid:
            # Check for hedging *despite* a labeled answer.
            hedged = any(p.search(text) for p in _HEDGE_PATTERNS)
            # Check for multiple distinct letters mentioned.
            all_letters = set(_LOOSE_RE.findall(text.upper())) & valid
            if len(all_letters) > 1:
                hedged = True
            return ParsedAnswer(letter, hedged, False, raw, "labeled")

    # Loose: first valid-letter word boundary match.
    all_matches = _LOOSE_RE.findall(text.upper())
    valid_matches = [c for c in all_matches if c in valid]
    if valid_matches:
        letter = valid_matches[0]
        hedged = (
            any(p.search(text) for p in _HEDGE_PATTERNS)
            or len(set(valid_matches)) > 1
        )
        return ParsedAnswer(letter, hedged, False, raw, "loose")

    # Refusal check.
    refused = any(p.search(text) for p in _HEDGE_PATTERNS[3:8])  # cannot/unable/not sure/unclear/ambiguous
    return ParsedAnswer(None, False, refused, raw, "none")


# Self-test
if __name__ == "__main__":
    cases = [
        ("A", {"A", "B", "C", "D"}, "A", False),
        ("(B).", {"A", "B", "C", "D"}, "B", False),
        ("The answer is C.", {"A", "B", "C", "D"}, "C", False),
        ("Both A and C", {"A", "B", "C", "D"}, "A", True),
        ("I cannot tell from the image.", {"A", "B", "C", "D"}, None, False),
        ("", {"A", "B", "C", "D"}, None, False),
        ("Option A is correct", {"A", "B"}, "A", False),
        ("C", {"A", "B"}, None, False),  # rejected: not in valid set
        ("BAD answer", {"A", "B", "C", "D"}, None, False),  # 'BAD' not bounded letters
    ]
    for raw, valid, want_letter, want_hedged in cases:
        p = parse_answer(raw, valid)
        ok = p.letter == want_letter and p.hedged == want_hedged
        print(f"{'OK' if ok else 'FAIL'} parse({raw!r}, {valid}) -> letter={p.letter} hedged={p.hedged} path={p.parse_path}")
