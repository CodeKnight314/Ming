from __future__ import annotations

import re

_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_NON_WHITESPACE_RE = re.compile(r"\S+")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def count_cjk_characters(text: str) -> int:
    return len(_CJK_CHAR_RE.findall(text or ""))


def count_language_aware_tokens(text: str) -> int:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return 0

    whitespace_tokens = len(_NON_WHITESPACE_RE.findall(cleaned))
    cjk_chars = count_cjk_characters(cleaned)

    # Use character-style counting for languages that typically do not separate
    # words with whitespace. For mixed-language text, prefer the larger signal.
    if cjk_chars and whitespace_tokens <= max(8, cjk_chars // 4):
        return cjk_chars

    return whitespace_tokens


def tokenize_for_overlap(text: str) -> list[str]:
    cleaned = normalize_whitespace(text).lower()
    if not cleaned:
        return []

    cjk_chars = count_cjk_characters(cleaned)
    whitespace_tokens = cleaned.split()
    if cjk_chars and len(whitespace_tokens) <= max(8, cjk_chars // 4):
        return [char for char in cleaned if _CJK_CHAR_RE.match(char)]

    return re.findall(r"[a-z0-9]+", cleaned)
