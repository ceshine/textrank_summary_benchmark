from pathlib import Path
from functools import partial

from tqdm import tqdm
from rouge import Rouge
from prettyprinter import pprint
from summa.summarizer import summarize

# from sentence_encoder import get_model
from similarity_functions import (
    original_textrank_similarity, bm25pluseps_weights_similarity_factory,
    setence_embeddings_similarity_factory
)

SIM_FUNCS = {
    "original": lambda _: original_textrank_similarity,
    "bm25pluseps025": partial(bm25pluseps_weights_similarity_factory, eps=0.25),
    # "sent_emb_large": partial(setence_embeddings_similarity_factory, **get_model("large"))
}

RESULTS_DIR = Path("results/")
RESULTS_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data/")


def cnndm_inference():
    """Summary generation for the CNN/DailyMail dataset"""
    for func_name, func in SIM_FUNCS.items():
        print(f"Processing \"{func_name}\"")
        print("=" * 20)
        with open(DATA_DIR / "cnndm" / "test.txt.src") as fin:
            with open(RESULTS_DIR / f"cnndm_{func_name}.pred", "w") as fout:
                for line in tqdm(fin.readlines()):
                    result = summarize(
                        line, ratio=0.2, similarity_func_factory=func)
                    result = result.replace("\n", " ")
                    fout.write(result + "\n")


def cnndm_eval():
    """Evaluation for the CNN/DailyMail dataset"""
    for func_name in SIM_FUNCS:
        print(f"Evaluating \"{func_name}\"")
        print("=" * 20)
        with open(RESULTS_DIR / f"cnndm_{func_name}.pred") as fin:
            predictions = fin.read().split("\n")[:-1]
        with open(DATA_DIR / "cnndm" / "test.txt.tgt.tagged") as fin:
            references = fin.read().replace("<t> ", "").replace(
                "</t> ", "").split("\n")[:-1]
        assert all([len(x) > 0 for x in predictions])
        scores = Rouge().get_scores(predictions, references, avg=True)
        pprint(scores)


if __name__ == "__main__":
    cnndm_inference()
    cnndm_eval()
