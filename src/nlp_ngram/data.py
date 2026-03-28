from __future__ import annotations

import re
from pathlib import Path

ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "inc.",
    "corp.",
    "co.",
    "ltd.",
    "vs.",
    "u.s.",
    "u.k.",
    "n.v.",
    "plc.",
    "nov.",
    "oct.",
    "dec.",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[^\w\s]")


def load_raw_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []

    words = normalized.split(" ")
    sentences: list[list[str]] = []
    current: list[str] = []

    for index, word in enumerate(words):
        current.append(word)
        if not _is_sentence_boundary(words, index):
            continue
        sentences.append(current)
        current = []

    if current:
        sentences.append(current)

    return [" ".join(sentence).strip() for sentence in sentences if sentence]


def tokenize_sentence(sentence: str, lowercase: bool = True) -> list[str]:
    tokens = TOKEN_PATTERN.findall(sentence)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens


def load_and_tokenize_corpus(path: str | Path, lowercase: bool = True) -> list[list[str]]:
    text = load_raw_text(path)
    sentences = split_sentences(text)
    return [tokenize_sentence(sentence, lowercase=lowercase) for sentence in sentences]


def _is_sentence_boundary(words: list[str], index: int) -> bool:
    word = words[index]
    if not re.search(r"[.!?][\"')\]]*$", word):
        return False

    stripped = re.sub(r"[\"')\]]+$", "", word).lower()
    if stripped in ABBREVIATIONS:
        return False

    if re.fullmatch(r"[A-Z]\.", word):
        return False

    if index + 1 >= len(words):
        return True

    next_word = words[index + 1].lstrip("\"'([")
    if not next_word:
        return False

    return next_word[:1].isupper() or next_word[:1].isdigit()

