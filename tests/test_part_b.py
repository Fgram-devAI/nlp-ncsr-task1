import math

from part_b import (
    NgramLM, preprocess_original, preprocess_lowercase,
    preprocess_abstract_digits, BOS, EOS, UNK,
)


# --- Preprocessing tests ---

def test_preprocess_original_returns_copy() -> None:
    sents = [["Hello", "World"]]
    result = preprocess_original(sents)
    assert result == sents
    result[0][0] = "changed"
    assert sents[0][0] == "Hello"  # original not mutated


def test_preprocess_lowercase() -> None:
    sents = [["The", "CAT", "Sat"]]
    result = preprocess_lowercase(sents)
    assert result == [["the", "cat", "sat"]]


def test_preprocess_abstract_digits() -> None:
    sents = [["$352.7", "8.4%", "33", "hello"]]
    result = preprocess_abstract_digits(sents)
    assert result == [["$###.#", "#.#%", "##", "hello"]]


# --- NgramLM tests ---

def test_bigram_fit_builds_vocabulary() -> None:
    model = NgramLM(order=2, k=1.0)
    model.fit([["a", "b", "c"], ["a", "b", "c"]], min_count=1)
    assert "a" in model.vocabulary
    assert BOS in model.vocabulary
    assert EOS in model.vocabulary
    assert UNK in model.vocabulary


def test_bigram_unk_replaces_rare_tokens() -> None:
    model = NgramLM(order=2, k=1.0)
    # "rare" appears once, min_count=3 -> replaced with UNK
    model.fit([["common"] * 5 + ["rare"]], min_count=3)
    assert "common" in model.vocabulary
    assert "rare" not in model.vocabulary


def test_bigram_probability_sums_roughly_to_one() -> None:
    model = NgramLM(order=2, k=1.0)
    model.fit([["a", "b"], ["a", "c"], ["b", "a"]], min_count=1)
    total = sum(model.prob(("a",), tok) for tok in model.vocabulary if tok != BOS)
    assert math.isclose(total, 1.0, rel_tol=1e-6)


def test_trigram_probability_with_smoothing() -> None:
    model = NgramLM(order=3, k=0.01)
    model.fit([["a", "b", "c"], ["a", "b", "d"]], min_count=1)
    # P(c | a, b) should be higher than P(x | a, b) for unseen x
    p_seen = model.prob(("a", "b"), "c")
    p_unseen = model.prob(("a", "b"), "d")
    # Both seen once, should be equal
    assert math.isclose(p_seen, p_unseen, rel_tol=1e-6)


def test_perplexity_is_finite() -> None:
    model = NgramLM(order=2, k=1.0)
    model.fit([["a", "b", "c"], ["d", "e", "f"]], min_count=1)
    pp = model.perplexity([["a", "b"]])
    assert pp > 0
    assert not math.isinf(pp)


def test_perplexity_lower_for_seen_data() -> None:
    train = [["the", "cat", "sat"]] * 10
    model = NgramLM(order=2, k=0.01)
    model.fit(train, min_count=1)
    pp_seen = model.perplexity([["the", "cat", "sat"]])
    pp_unseen = model.perplexity([["sat", "the", "cat"]])
    assert pp_seen < pp_unseen


def test_generate_produces_tokens() -> None:
    model = NgramLM(order=2, k=0.01)
    model.fit([["the", "cat", "sat"]] * 10, min_count=1)
    sent = model.generate(start_words=["the"], max_len=10, seed=42)
    assert len(sent) >= 1
    assert sent[0] == "the"
    assert UNK not in sent


def test_generate_no_unk_in_output() -> None:
    model = NgramLM(order=2, k=1.0)
    model.fit([["a", "b"]] * 5, min_count=1)
    for seed in range(10):
        sent = model.generate(max_len=20, seed=seed)
        assert UNK not in sent


def test_padding_bigram() -> None:
    model = NgramLM(order=2, k=1.0)
    padded = model._pad(["a", "b"])
    assert padded == [BOS, "a", "b", EOS]


def test_padding_trigram() -> None:
    model = NgramLM(order=3, k=1.0)
    padded = model._pad(["a", "b"])
    assert padded == [BOS, BOS, "a", "b", EOS]
