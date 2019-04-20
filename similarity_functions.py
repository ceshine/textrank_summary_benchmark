from math import log10

import numpy as np
import pandas as pd
import tensorflow as tf
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


def setence_embeddings_similarity_factory(sentences, sentence_input, sentence_emb, batch_size=32):
    df_sent = pd.DataFrame(
        [(x.text, x.token) for x in sentences],
        columns=("text", "token")
    )
    # # remove extremely short sentences
    # df_sent = df_sent[df_sent.text.str.len() > 5]
    # Deduplicate
    df_sent = df_sent.drop_duplicates()
    sentence_index = {row["token"]: i for i, row in df_sent.iterrows()}
    sentence_embeddings = []
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(),
                     tf.tables_initializer()])
        for i in range(0, len(sentences), batch_size):
            sentence_embeddings.append(
                session.run(
                    sentence_emb,
                    feed_dict={
                        sentence_input: df_sent.iloc[
                            i:(i+batch_size), 0]
                    }
                )
            )
    sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
    similarity_matrix = sentence_embeddings @ sentence_embeddings.T

    def cosine_similarity(s1, s2):
        return similarity_matrix[sentence_index[s1], sentence_index[s2]]

    return cosine_similarity
