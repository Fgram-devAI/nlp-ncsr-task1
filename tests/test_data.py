from nlp_ngram.data import split_sentences, tokenize_sentence


def test_split_sentences_keeps_common_abbreviations() -> None:
    text = "Mr. Vinken arrived. Dr. Smith stayed."
    sentences = split_sentences(text)
    assert sentences == ["Mr. Vinken arrived.", "Dr. Smith stayed."]


def test_tokenize_sentence_splits_punctuation() -> None:
    tokens = tokenize_sentence("Kent cigarettes, in 1956.", lowercase=True)
    assert tokens == ["kent", "cigarettes", ",", "in", "1956", "."]

