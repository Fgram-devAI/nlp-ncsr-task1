from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import spacy
from transformers import BertTokenizer


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_corpus(path: str | Path) -> str:
    """Read the raw WSJ corpus file."""
    return Path(path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. Tokenization (three methods)
# ---------------------------------------------------------------------------

def tokenize_nltk(text: str) -> list[str]:
    """Tokenize using nltk.word_tokenize."""
    nltk.download("punkt_tab", quiet=True)
    return nltk.word_tokenize(text)


def tokenize_spacy(text: str, batch_size: int = 10_000) -> list[str]:
    """Tokenize using spaCy en_core_web_sm (tokenizer only, no pipeline)."""
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "tagger"])
    nlp.max_length = len(text) + 1000
    tokens: list[str] = []
    # Process in chunks to avoid memory issues
    chunks = [text[i:i + batch_size] for i in range(0, len(text), batch_size)]
    for chunk in chunks:
        doc = nlp(chunk)
        tokens.extend(tok.text for tok in doc if not tok.is_space)
    return tokens


def tokenize_bert(text: str) -> list[str]:
    """Tokenize using BertTokenizer (bert-base-cased)."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    return tokenizer.tokenize(text)


# ---------------------------------------------------------------------------
# 3. Corpus statistics
# ---------------------------------------------------------------------------

def compute_statistics(tokens: list[str]) -> dict:
    """Compute #tokens, #types, TTR, hapax legomena, hapax dislegomena."""
    freq = Counter(tokens)
    n_tokens = len(tokens)
    n_types = len(freq)
    ttr = n_types / n_tokens if n_tokens else 0.0
    hapax_legomena = sum(1 for count in freq.values() if count == 1)
    hapax_dislegomena = sum(1 for count in freq.values() if count == 2)
    return {
        "tokens": n_tokens,
        "types": n_types,
        "ttr": ttr,
        "hapax_legomena": hapax_legomena,
        "hapax_dislegomena": hapax_dislegomena,
        "freq": freq,
    }


# ---------------------------------------------------------------------------
# 4. Sentence-level comparison
# ---------------------------------------------------------------------------

def get_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK sent_tokenize."""
    nltk.download("punkt_tab", quiet=True)
    return nltk.sent_tokenize(text)


def pick_random_sentence(text: str, min_words: int = 15, seed: int = 42) -> str:
    """Pick a random sentence with at least min_words words."""
    sentences = get_sentences(text)
    candidates = [s for s in sentences if len(s.split()) >= min_words]
    rng = random.Random(seed)
    return rng.choice(candidates)


def compare_tokenizers_on_sentence(sentence: str) -> dict[str, list[str]]:
    """Show tokens produced by each method for a single sentence."""
    return {
        "NLTK": tokenize_nltk(sentence),
        "spaCy": tokenize_spacy(sentence),
        "BERT": tokenize_bert(sentence),
    }


# ---------------------------------------------------------------------------
# 5. Top types covering 30% of tokens
# ---------------------------------------------------------------------------

def top_types_covering_fraction(freq: Counter, total_tokens: int, fraction: float = 0.30) -> list[tuple[str, int]]:
    """Return the most frequent types whose cumulative count reaches fraction of total."""
    target = total_tokens * fraction
    cumulative = 0
    result = []
    for word, count in freq.most_common():
        result.append((word, count))
        cumulative += count
        if cumulative >= target:
            break
    return result


# ---------------------------------------------------------------------------
# 6. Zipf's law analysis
# ---------------------------------------------------------------------------

def find_best_zipf_constant(freq: Counter, total_tokens: int, candidates: list[float] | None = None) -> float:
    """Find the value of A (from 0.1 to 1.0) that best fits Zipf's law.

    Zipf's law: P(rank r) = A / r
    We minimise sum of squared differences between observed and predicted probabilities.
    """
    if candidates is None:
        candidates = [round(x * 0.1, 1) for x in range(1, 11)]

    ranked_freqs = [count for _, count in freq.most_common()]
    observed_probs = [c / total_tokens for c in ranked_freqs]
    ranks = list(range(1, len(ranked_freqs) + 1))

    best_a = candidates[0]
    best_sse = float("inf")

    for a in candidates:
        predicted = [a / r for r in ranks]
        sse = sum((obs - pred) ** 2 for obs, pred in zip(observed_probs, predicted))
        if sse < best_sse:
            best_sse = sse
            best_a = a

    return best_a


def plot_zipf(
    freq: Counter,
    total_tokens: int,
    best_a: float,
    title: str = "Zipf's Law",
    save_path: str | None = None,
) -> plt.Figure:
    """Create log-log plot comparing observed frequencies with Zipf prediction."""
    ranked = freq.most_common()
    ranks = list(range(1, len(ranked) + 1))
    observed_freqs = [count for _, count in ranked]
    predicted_freqs = [best_a * total_tokens / r for r in ranks]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(ranks, observed_freqs, "b.", markersize=2, alpha=0.6, label="Observed")
    ax.loglog(ranks, predicted_freqs, "r-", linewidth=1.5, label=f"Zipf (A={best_a})")
    ax.set_xlabel("Rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part A: Tokenization & Zipf's Law")
    parser.add_argument("--corpus", type=str, default="wsj_untokenized.txt",
                        help="Path to the WSJ corpus file")
    args = parser.parse_args()

    print("=" * 60)
    print("Part A: Tokens, Types, Zipf's Law")
    print("=" * 60)

    # Load corpus
    raw_text = load_corpus(args.corpus)
    print(f"Corpus loaded: {len(raw_text):,} characters\n")

    # A.1 — Tokenize with all three methods and compute statistics
    print("Tokenizing...")
    all_tokens = {
        "NLTK": tokenize_nltk(raw_text),
        "spaCy": tokenize_spacy(raw_text),
        "BERT": tokenize_bert(raw_text),
    }
    all_stats = {name: compute_statistics(toks) for name, toks in all_tokens.items()}

    print(f"\n{'':20s} {'NLTK':>10s} {'spaCy':>10s} {'BERT':>10s}")
    print("-" * 52)
    for key in ["tokens", "types", "ttr", "hapax_legomena", "hapax_dislegomena"]:
        label = key.replace("_", " ").title()
        vals = [all_stats[m][key] for m in ["NLTK", "spaCy", "BERT"]]
        if key == "ttr":
            print(f"{label:20s} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}")
        else:
            print(f"{label:20s} {vals[0]:>10,} {vals[1]:>10,} {vals[2]:>10,}")

    # A.2 — Random sentence comparison
    print(f"\n{'=' * 60}")
    sentence = pick_random_sentence(raw_text, min_words=15, seed=42)
    print(f"Random sentence ({len(sentence.split())} words):")
    print(f"  \"{sentence}\"\n")
    for method, toks in compare_tokenizers_on_sentence(sentence).items():
        print(f"  {method} ({len(toks)} tokens): {toks}")

    # A.3 — Top types covering 30%
    print(f"\n{'=' * 60}")
    print("Top types covering 30% of tokens:")
    for name in ["NLTK", "spaCy", "BERT"]:
        top = top_types_covering_fraction(all_stats[name]["freq"], all_stats[name]["tokens"])
        print(f"  {name}: {len(top)} types")

    # A.4 — Zipf's law
    print(f"\n{'=' * 60}")
    freq_nltk = all_stats["NLTK"]["freq"]
    total_nltk = all_stats["NLTK"]["tokens"]
    best_a = find_best_zipf_constant(freq_nltk, total_nltk)
    print(f"Best Zipf constant A = {best_a}")
    plot_zipf(freq_nltk, total_nltk, best_a,
              title=f"Zipf's Law — NLTK (A={best_a})",
              save_path="output/zipf_plot.png")
    print("Plot saved to output/zipf_plot.png")
