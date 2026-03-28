import math

from nlp_ngram.ngram import NgramLanguageModel


def test_bigram_probability_with_add_one_smoothing() -> None:
    model = NgramLanguageModel(order=2)
    model.fit([["a", "b"], ["a", "c"]])
    probability = model.probability(("a",), "b", alpha=1.0)
    assert math.isclose(probability, 2 / 7)


def test_unknown_tokens_map_to_unk() -> None:
    model = NgramLanguageModel(order=1)
    model.fit([["known"], ["known"]], min_count=2)
    assert model.probability((), "missing", alpha=1.0) > 0.0
