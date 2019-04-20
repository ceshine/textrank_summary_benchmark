"""Using cosine similarity with sentence embeddings from Universal Sentence Encoder."""
import os
import logging
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece

tf.logging.set_verbosity(logging.ERROR)

os.environ["TFHUB_CACHE_DIR"] = "/mnt/SSD_Data/tf_hub_cache/"

MODELS = {
    "xling": "https://tfhub.dev/google/universal-sentence-encoder-xling-many/1",
    "large": "https://tfhub.dev/google/universal-sentence-encoder-large/3",
    # base does not work with GPU
    "base": "https://tfhub.dev/google/universal-sentence-encoder/2"
}


def get_model(model_name="large"):
    tf.reset_default_graph()
    # Define graph
    sentence_input = tf.placeholder(tf.string, shape=(None))
    encoder = hub.Module(MODELS[model_name])
    # For evaluation we use exactly normalized rather than
    # approximately normalized.
    sentence_emb = tf.nn.l2_normalize(encoder(sentence_input), axis=1)
    return {"sentence_input": sentence_input, "sentence_emb": sentence_emb}
