from math import log10

from summa.summarizer import _count_common_words

from bm25 import bm25pluseps_weights as _bm25pluseps_weights


def original_textrank_similarity(s1, s2):
    words_sentence_one = s1.split()
    words_sentence_two = s2.split()

    common_word_count = _count_common_words(
        words_sentence_one, words_sentence_two)

    log_s1 = log10(len(words_sentence_one))
    log_s2 = log10(len(words_sentence_two))

    if log_s1 + log_s2 == 0:
        return 0

    return common_word_count / (log_s1 + log_s2)


def bm25pluseps_weights_similarity_factory(sentences, eps=0.25):
    # Deduplicate
    sentences = list(set([x.token for x in sentences]))
    weights = _bm25pluseps_weights(
        [sentence.split() for sentence in sentences], eps=eps)
    sentence_index = {sentence: i for i, sentence in enumerate(sentences)}

    def bm25plus_weights_similarity(s1, s2):
        return weights[sentence_index[s1]][sentence_index[s2]]

    return bm25plus_weights_similarity
