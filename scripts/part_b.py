from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field

import nltk


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"
MIN_COUNT = 3  # assignment requirement: tokens appearing < 3 times -> UNK
TRAIN_FILES = 170
# total treebank files = 199, test = 29


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_treebank() -> tuple[list[list[str]], list[list[str]]]:
    """Load treebank corpus and split into train (170 files) / test (29 files).

    Returns (train_sentences, test_sentences) where each sentence is a list of tokens.
    """
    nltk.download("treebank", quiet=True)
    from nltk.corpus import treebank

    files = treebank.fileids()
    train_files = files[:TRAIN_FILES]
    test_files = files[TRAIN_FILES:]

    train_sents = [list(sent) for f in train_files for sent in treebank.sents(f)]
    test_sents = [list(sent) for f in test_files for sent in treebank.sents(f)]
    return train_sents, test_sents


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_original(sentences: list[list[str]]) -> list[list[str]]:
    """No preprocessing — return as-is."""
    return [list(s) for s in sentences]


def preprocess_lowercase(sentences: list[list[str]]) -> list[list[str]]:
    """Convert all tokens to lowercase."""
    return [[tok.lower() for tok in s] for s in sentences]


def preprocess_abstract_digits(sentences: list[list[str]]) -> list[list[str]]:
    """Replace every digit character with '#'.

    E.g. '33' -> '##', '8.4%' -> '#.#%', '$352.7' -> '$###.#'
    """
    return [[re.sub(r"\d", "#", tok) for tok in s] for s in sentences]


PREPROCESSING = {
    "original": preprocess_original,
    "lowercase": preprocess_lowercase,
    "abstract_digits": preprocess_abstract_digits,
}


# ---------------------------------------------------------------------------
# N-gram Language Model
# ---------------------------------------------------------------------------

@dataclass
class NgramLM:
    """N-gram language model with add-k smoothing."""

    order: int  # 2 for bigram, 3 for trigram
    k: float = 1.0

    vocabulary: set[str] = field(default_factory=set, repr=False)
    ngram_counts: Counter = field(default_factory=Counter, repr=False)
    context_counts: Counter = field(default_factory=Counter, repr=False)
    _fitted: bool = field(default=False, repr=False)

    # ---- Training ----

    def fit(self, sentences: list[list[str]], min_count: int = MIN_COUNT) -> None:
        """Train the model: build vocabulary, replace rare tokens, count n-grams."""
        # Count raw token frequencies
        raw_counts = Counter(tok for sent in sentences for tok in sent)

        # Build vocabulary: tokens with count >= min_count + special tokens
        self.vocabulary = {tok for tok, c in raw_counts.items() if c >= min_count}
        self.vocabulary.update({BOS, EOS, UNK})

        # Replace rare tokens with UNK and count n-grams
        self.ngram_counts = Counter()
        self.context_counts = Counter()

        for sent in sentences:
            normalized = [tok if tok in self.vocabulary else UNK for tok in sent]
            padded = self._pad(normalized)

            for i in range(len(padded) - self.order + 1):
                ngram = tuple(padded[i : i + self.order])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        self._fitted = True

    # ---- Probability ----

    def log_prob(self, context: tuple[str, ...], token: str) -> float:
        """Compute log P(token | context) with add-k smoothing."""
        self._check_fitted()
        ctx = tuple(self._map(t) for t in context)
        tok = self._map(token)
        ngram = ctx + (tok,)

        # Prediction vocab: everything except BOS (BOS can never be predicted)
        v_size = len(self.vocabulary) - 1  # exclude BOS

        numerator = self.ngram_counts[ngram] + self.k
        denominator = self.context_counts[ctx] + self.k * v_size

        if denominator == 0:
            return float("-inf")
        return math.log(numerator / denominator)

    def prob(self, context: tuple[str, ...], token: str) -> float:
        """Compute P(token | context) with add-k smoothing."""
        lp = self.log_prob(context, token)
        return math.exp(lp) if lp != float("-inf") else 0.0

    # ---- Perplexity ----

    def perplexity(self, sentences: list[list[str]]) -> float:
        """Compute perplexity over test sentences.

        Perplexity = exp( -1/N * sum(ln P(g_i)) )
        where g_i are all bigrams/trigrams and N is their total count.
        """
        self._check_fitted()
        total_log_prob = 0.0
        n_ngrams = 0

        for sent in sentences:
            normalized = [self._map(tok) for tok in sent]
            padded = self._pad(normalized)

            for i in range(len(padded) - self.order + 1):
                ngram = tuple(padded[i : i + self.order])
                context = ngram[:-1]
                token = ngram[-1]
                total_log_prob += self.log_prob(context, token)
                n_ngrams += 1

        if n_ngrams == 0:
            raise ValueError("No n-grams found in test data")
        return math.exp(-total_log_prob / n_ngrams)

    # ---- Sentence generation ----

    def generate(self, start_words: list[str] | None = None, max_len: int = 50, seed: int | None = None) -> list[str]:
        """Generate a sentence by sampling next tokens proportionally to their probability.

        Starts with BOS (+ optional start_words), ends at EOS or max_len.
        Never produces UNK tokens in output.
        """
        self._check_fitted()
        rng = random.Random(seed)

        # Build initial context
        if start_words is None:
            start_words = []
        sentence = list(start_words)
        # Context is the last (order-1) tokens, starting with BOS padding
        context_tokens = [BOS] * (self.order - 1) + sentence

        # All tokens that can be predicted (exclude BOS and UNK)
        predictable = sorted(self.vocabulary - {BOS, UNK})

        for _ in range(max_len):
            ctx = tuple(context_tokens[-(self.order - 1):])

            # Compute probabilities for all predictable tokens
            probs = []
            for tok in predictable:
                p = self.prob(ctx, tok)
                probs.append(p)

            total = sum(probs)
            if total == 0:
                break

            # Weighted random choice
            r = rng.random() * total
            cumulative = 0.0
            chosen = predictable[-1]
            for tok, p in zip(predictable, probs):
                cumulative += p
                if r <= cumulative:
                    chosen = tok
                    break

            if chosen == EOS:
                break

            sentence.append(chosen)
            context_tokens.append(chosen)

        return sentence

    # ---- Helpers ----

    def _pad(self, sentence: list[str]) -> list[str]:
        """Add BOS/EOS padding. BOS is repeated (order-1) times."""
        return [BOS] * (self.order - 1) + sentence + [EOS]

    def _map(self, token: str) -> str:
        """Map out-of-vocabulary tokens to UNK."""
        if token in self.vocabulary:
            return token
        return UNK

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before use")


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Part B: N-gram Language Models")
    print("=" * 60)

    # Load data
    train_raw, test_raw = load_treebank()
    print(f"Training sentences: {len(train_raw)}")
    print(f"Test sentences:     {len(test_raw)}\n")

    # B.1 — Perplexity table
    configs = [
        ("Bigrams (k=1)",    2, 1.0),
        ("Bigrams (k=0.01)", 2, 0.01),
        ("Trigrams (k=1)",   3, 1.0),
        ("Trigrams (k=0.01)", 3, 0.01),
    ]
    modes = ["original", "lowercase", "abstract_digits"]

    print(f"{'Model':<20s} {'Original':>10s} {'Lowercase':>10s} {'Abs.Digits':>10s}")
    print("-" * 52)

    for label, order, k in configs:
        row = []
        for mode in modes:
            train = PREPROCESSING[mode](train_raw)
            test = PREPROCESSING[mode](test_raw)
            model = NgramLM(order=order, k=k)
            model.fit(train, min_count=MIN_COUNT)
            pp = model.perplexity(test)
            row.append(f"{pp:.2f}")
        print(f"{label:<20s} {row[0]:>10s} {row[1]:>10s} {row[2]:>10s}")

    # B.2 — Sentence generation
    print(f"\n{'=' * 60}")
    print("Sentence Generation (original text, k=0.01)")
    print("=" * 60)

    train_orig = PREPROCESSING["original"](train_raw)
    start_options = [["The"], ["Mr."], ["In"]]

    for order, name in [(2, "BIGRAM"), (3, "TRIGRAM")]:
        model = NgramLM(order=order, k=0.01)
        model.fit(train_orig, min_count=MIN_COUNT)
        print(f"\n{name} MODEL:")
        for i, start in enumerate(start_options):
            sent = model.generate(start_words=start, max_len=30, seed=i)
            print(f"  {i+1}. {' '.join(sent)}")
