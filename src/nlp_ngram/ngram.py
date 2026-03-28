from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class NgramLanguageModel:
    order: int
    unk_token: str = "<unk>"
    start_token: str = "<s>"
    end_token: str = "</s>"
    vocabulary: set[str] = field(default_factory=set)
    token_counts: Counter[str] = field(default_factory=Counter)
    ngram_counts: Counter[tuple[str, ...]] = field(default_factory=Counter)
    context_counts: Counter[tuple[str, ...]] = field(default_factory=Counter)
    is_fitted: bool = False

    def fit(self, sentences: list[list[str]], min_count: int = 1) -> None:
        if self.order < 1:
            raise ValueError("order must be >= 1")
        if not sentences:
            raise ValueError("sentences must not be empty")

        flat_counts = Counter(token for sentence in sentences for token in sentence)
        self.vocabulary = {
            token for token, count in flat_counts.items() if count >= min_count
        }
        self.vocabulary.update({self.unk_token, self.start_token, self.end_token})
        self.token_counts = Counter()
        self.ngram_counts = Counter()
        self.context_counts = Counter()

        for sentence in sentences:
            normalized = [token if token in self.vocabulary else self.unk_token for token in sentence]
            self.token_counts.update(normalized)
            padded = self._pad_sentence(normalized)

            for index in range(len(padded) - self.order + 1):
                ngram = tuple(padded[index : index + self.order])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        self.is_fitted = True

    def probability(self, context: tuple[str, ...], token: str, alpha: float = 0.0) -> float:
        self._require_fit()
        normalized_context = self._normalize_context(context)
        normalized_token = self._normalize_token(token)
        ngram = normalized_context + (normalized_token,)
        numerator = self.ngram_counts[ngram] + alpha
        prediction_vocabulary_size = len(self.vocabulary - {self.start_token})

        if self.order == 1:
            denominator = sum(self.ngram_counts.values()) + alpha * prediction_vocabulary_size
        else:
            denominator = self.context_counts[normalized_context] + alpha * prediction_vocabulary_size

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def sentence_log_probability(self, sentence: list[str], alpha: float = 0.0) -> float:
        self._require_fit()
        normalized = [self._normalize_token(token) for token in sentence]
        padded = self._pad_sentence(normalized)
        total = 0.0

        for index in range(len(padded) - self.order + 1):
            ngram = tuple(padded[index : index + self.order])
            context = ngram[:-1]
            token = ngram[-1]
            probability = self.probability(context, token, alpha=alpha)
            if probability <= 0.0:
                return float("-inf")
            total += math.log(probability)

        return total

    def perplexity(self, sentences: list[list[str]], alpha: float = 0.0) -> float:
        self._require_fit()
        total_log_probability = 0.0
        total_predictions = 0

        for sentence in sentences:
            normalized = [self._normalize_token(token) for token in sentence]
            padded = self._pad_sentence(normalized)
            total_predictions += len(padded) - self.order + 1
            total_log_probability += self.sentence_log_probability(sentence, alpha=alpha)

        if total_predictions <= 0:
            raise ValueError("no predictions available for perplexity")
        if total_log_probability == float("-inf"):
            return float("inf")

        return math.exp(-total_log_probability / total_predictions)

    def top_ngrams(self, limit: int = 10) -> list[tuple[tuple[str, ...], int]]:
        self._require_fit()
        return self.ngram_counts.most_common(limit)

    def _pad_sentence(self, sentence: list[str]) -> list[str]:
        return [self.start_token] * (self.order - 1) + sentence + [self.end_token]

    def _normalize_context(self, context: tuple[str, ...]) -> tuple[str, ...]:
        expected = self.order - 1
        if len(context) != expected:
            raise ValueError(f"context must have length {expected}")
        return tuple(self._normalize_token(token) for token in context)

    def _normalize_token(self, token: str) -> str:
        return token if token in self.vocabulary else self.unk_token

    def _require_fit(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("model must be fitted before use")
