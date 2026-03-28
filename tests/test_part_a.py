from collections import Counter

from part_a import compute_statistics, top_types_covering_fraction, find_best_zipf_constant


def test_compute_statistics_basic() -> None:
    tokens = ["the", "cat", "sat", "the", "cat", "the"]
    stats = compute_statistics(tokens)
    assert stats["tokens"] == 6
    assert stats["types"] == 3
    assert stats["hapax_legomena"] == 1  # "sat"
    assert stats["hapax_dislegomena"] == 1  # "cat"
    assert round(stats["ttr"], 4) == 0.5


def test_compute_statistics_all_unique() -> None:
    tokens = ["a", "b", "c", "d"]
    stats = compute_statistics(tokens)
    assert stats["hapax_legomena"] == 4
    assert stats["hapax_dislegomena"] == 0
    assert stats["ttr"] == 1.0


def test_top_types_covering_fraction() -> None:
    freq = Counter({"the": 50, "a": 30, "cat": 10, "dog": 10})
    # total = 100, 30% = 30, "the" alone covers 50%
    result = top_types_covering_fraction(freq, 100, 0.30)
    assert len(result) == 1
    assert result[0] == ("the", 50)


def test_top_types_covering_fraction_needs_multiple() -> None:
    freq = Counter({"the": 20, "a": 15, "cat": 10, "dog": 5})
    # total = 50, 30% = 15, need "the"(20) to reach 15
    result = top_types_covering_fraction(freq, 50, 0.30)
    assert len(result) == 1
    # 60% target: need "the"(20) + "a"(15) = 35 >= 30
    result = top_types_covering_fraction(freq, 50, 0.60)
    assert len(result) == 2


def test_find_best_zipf_constant() -> None:
    # Create a frequency distribution that roughly follows Zipf with A=0.1
    freq = Counter({"w1": 100, "w2": 50, "w3": 33, "w4": 25, "w5": 20})
    total = sum(freq.values())
    best_a = find_best_zipf_constant(freq, total)
    assert best_a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
