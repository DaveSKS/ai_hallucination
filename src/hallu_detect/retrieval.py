"""Small dependency-free BM25 retriever for user-supplied evidence."""

import math
import re
from dataclasses import dataclass


_TOKEN = re.compile(r"\b[\w'-]+\b", re.UNICODE)


@dataclass(frozen=True)
class EvidenceChunk:
    text: str
    source: str
    score: float


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN.findall(text)]


def _split_chunks(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = chunk_words - overlap_words
    return [
        " ".join(words[start : start + chunk_words])
        for start in range(0, len(words), step)
    ]


class BM25Retriever:
    """Indexes supplied documents in memory and ranks their text chunks per query."""

    def __init__(
        self,
        documents: list[str | dict[str, str]],
        *,
        chunk_words: int = 180,
        overlap_words: int = 30,
    ) -> None:
        if chunk_words < 1 or overlap_words < 0 or overlap_words >= chunk_words:
            raise ValueError("chunk_words must be positive and overlap_words smaller than it")

        self._chunks: list[tuple[str, str, list[str]]] = []
        for index, document in enumerate(documents):
            if isinstance(document, str):
                text, source = document, f"document-{index + 1}"
            elif isinstance(document, dict):
                text = document.get("text", "")
                source = document.get("source", f"document-{index + 1}")
                if not isinstance(text, str) or not isinstance(source, str):
                    raise TypeError("document text and source must be strings")
            else:
                raise TypeError("each document must be a string or an object with text/source")

            for chunk in _split_chunks(text, chunk_words, overlap_words):
                tokens = _tokens(chunk)
                if tokens:
                    self._chunks.append((chunk, source, tokens))

        self._average_length = (
            sum(len(tokens) for _, _, tokens in self._chunks) / len(self._chunks)
            if self._chunks
            else 0.0
        )
        self._document_frequency: dict[str, int] = {}
        for _, _, tokens in self._chunks:
            for token in set(tokens):
                self._document_frequency[token] = self._document_frequency.get(token, 0) + 1

    def search(self, query: str, *, top_k: int = 3) -> list[EvidenceChunk]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_tokens = set(_tokens(query))
        if not query_tokens or not self._chunks:
            return []

        total_chunks = len(self._chunks)
        k1, b = 1.5, 0.75
        ranked: list[EvidenceChunk] = []
        for text, source, tokens in self._chunks:
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            length = len(tokens)
            score = 0.0
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_frequency = math.log(
                    1 + (total_chunks - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b + b * length / self._average_length
                )
                score += inverse_frequency * frequency * (k1 + 1) / denominator
            if score > 0:
                ranked.append(EvidenceChunk(text=text, source=source, score=score))

        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]
