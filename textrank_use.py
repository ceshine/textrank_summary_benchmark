from pathlib import Path
from functools import partial
from itertools import chain

from tqdm import tqdm
from rouge import Rouge
import numpy as np
import tensorflow as tf
from prettyprinter import pprint
from summa.preprocessing.textcleaner import clean_text_by_sentences
from summa.pagerank_weighted import pagerank_weighted_scipy as _pagerank
from summa.commons import build_graph as _build_graph
from summa.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from summa.summarizer import (
    _set_graph_edge_weights, _format_results,
    _extract_most_important_sentences
)

from sentence_encoder import get_model

RESULTS_DIR = Path("results/")
RESULTS_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data/")

# MODELS = ["base"]  # cpu
MODELS = ["large", "xling"]  # gpu


def get_sentence_embeddings(session, sentences, sentence_input, sentence_emb, batch_size=128):
    sentence_embeddings = []
    for i in range(0, len(sentences), batch_size):
        sentence_embeddings.append(
            session.run(
                sentence_emb,
                feed_dict={
                    sentence_input: [
                        x.text for x in sentences[i:(i+batch_size)]
                    ]
                }
            )
        )
    sentence_embeddings = np.concatenate(sentence_embeddings, axis=0)
    return sentence_embeddings


def cosine_similarity(similarity_matrix, id_1, id_2):
    return similarity_matrix[id_1, id_2]


def _add_scores_to_sentences(sentences, scores):
    for i, sentence in enumerate(sentences):
        # Adds the score to the object if it has one.
        if i in scores:
            sentence.score = scores[i]
        else:
            sentence.score = 0


def summarize(sentences, similarity_matrix, ratio=0.2, split=False):
    # Creates the graph and calculates the similarity coefficient for every pair of nodes.
    graph = _build_graph([i for i in range(len(sentences))])

    _set_graph_edge_weights(graph, similarity_func=partial(
        cosine_similarity, similarity_matrix))

    # Remove all nodes with all edges weights equal to zero.
    _remove_unreachable_nodes(graph)

    # PageRank cannot be run in an empty graph.
    if len(graph.nodes()) == 0:
        return [] if split else ""

    # Ranks the tokens using the PageRank algorithm. Returns dict of sentence -> score
    pagerank_scores = _pagerank(graph)

    # Adds the summa scores to the sentence objects.
    _add_scores_to_sentences(sentences, pagerank_scores)

    # Extracts the most important sentences with the selected criterion.
    extracted_sentences = _extract_most_important_sentences(
        sentences, ratio, words=None)

    # Sorts the extracted sentences by apparition order in the original text.
    extracted_sentences.sort(key=lambda s: s.index)
    # print([(x.index, x.score) for x in extracted_sentences[:2]])
    return _format_results(extracted_sentences, split, score=None)


def process_sentence_batch(sess, batch, fout, model):
    sentence_embeddings = get_sentence_embeddings(
        sess, list(chain(*batch)), **model)
    idx_tmp = 0
    for sentences in batch:
        embeddings_slice = sentence_embeddings[
            idx_tmp: idx_tmp+len(sentences)]
        result = summarize(
            sentences,
            embeddings_slice @ embeddings_slice.T,
            ratio=0.2
        )
        idx_tmp += len(sentences)
        result = result.replace("\n", " ")
        fout.write(result + "\n")


def cnndm_inference(batch_size=128):
    """Summary generation for the CNN/DailyMail dataset"""
    for model_name in MODELS:
        print("Processing \"{}\"".format(model_name))
        print("=" * 20)
        model = get_model(model_name)
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(),
                         tf.tables_initializer()])
            batch = []
            with open(str(DATA_DIR / "cnndm" / "test.txt.src")) as fin:
                with open(str(RESULTS_DIR / "cnndm_use_{}.pred".format(model_name)), "w") as fout:
                    for line in tqdm(fin.readlines()):
                        batch.append(clean_text_by_sentences(line, "english"))
                        if len(batch) == batch_size:
                            process_sentence_batch(session, batch, fout, model)
                            batch = []
                    if len(batch) > 0:
                        process_sentence_batch(session, batch, fout, model)


def cnndm_eval():
    """Evaluation for the CNN/DailyMail dataset"""
    for model_name in MODELS:
        print("Evaluating \"{}\"".format(model_name))
        print("=" * 20)
        with open(str(RESULTS_DIR / "cnndm_use_{}.pred".format(model_name))) as fin:
            predictions = fin.read().split("\n")[:-1]
        with open(str(DATA_DIR / "cnndm" / "test.txt.tgt.tagged")) as fin:
            references = fin.read().replace("<t> ", "").replace(
                "</t> ", "").split("\n")[:-1]
        assert all([len(x) > 0 for x in predictions])
        scores = Rouge().get_scores(predictions, references, avg=True)
        pprint(scores)


if __name__ == "__main__":
    cnndm_inference()
    cnndm_eval()
