"""Deterministic sentence-level claim extraction baseline."""

import re


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_BULLET_PREFIX = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_HEADING = re.compile(r"^[^.!?]{1,80}:$")


def extract_claims(response: str) -> list[str]:
    """Split response text into sentence-level units; this is not semantic atomization."""
    if not isinstance(response, str):
        raise TypeError("response must be a string")

    claims: list[str] = []
    for line in response.splitlines():
        line = _BULLET_PREFIX.sub("", line).strip()
        if not line or _HEADING.fullmatch(line):
            continue
        claims.extend(
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY.split(line)
            if sentence.strip()
        )
    return claims
