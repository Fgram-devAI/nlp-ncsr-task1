from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_and_tokenize_corpus
from .ngram import NgramLanguageModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="N-gram language model helper CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_parser = subparsers.add_parser("stats", help="Show corpus and model statistics.")
    _add_shared_arguments(stats_parser)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate perplexity on a held-out split.")
    _add_shared_arguments(eval_parser)
    eval_parser.add_argument("--train-ratio", type=float, default=0.8)
    eval_parser.add_argument("--alpha", type=float, default=1.0)
    return parser


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--min-count", type=int, default=1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sentences = load_and_tokenize_corpus(args.corpus)

    if args.command == "stats":
        run_stats(sentences, args.order, args.min_count)
        return

    if args.command == "evaluate":
        run_evaluation(
            sentences=sentences,
            order=args.order,
            min_count=args.min_count,
            train_ratio=args.train_ratio,
            alpha=args.alpha,
        )


def run_stats(sentences: list[list[str]], order: int, min_count: int) -> None:
    model = NgramLanguageModel(order=order)
    model.fit(sentences, min_count=min_count)
    token_total = sum(len(sentence) for sentence in sentences)
    print(f"sentences: {len(sentences)}")
    print(f"tokens: {token_total}")
    print(f"vocabulary: {len(model.vocabulary)}")
    print(f"top_{order}grams:")
    for ngram, count in model.top_ngrams():
        print(f"  {ngram} -> {count}")


def run_evaluation(
    sentences: list[list[str]],
    order: int,
    min_count: int,
    train_ratio: float,
    alpha: float,
) -> None:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    split_index = max(1, int(len(sentences) * train_ratio))
    train_sentences = sentences[:split_index]
    dev_sentences = sentences[split_index:]

    if not dev_sentences:
        raise ValueError("development split is empty; lower train_ratio")

    model = NgramLanguageModel(order=order)
    model.fit(train_sentences, min_count=min_count)
    perplexity = model.perplexity(dev_sentences, alpha=alpha)

    print(f"train_sentences: {len(train_sentences)}")
    print(f"dev_sentences: {len(dev_sentences)}")
    print(f"order: {order}")
    print(f"alpha: {alpha}")
    print(f"perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main()

