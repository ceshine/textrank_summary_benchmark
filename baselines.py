import argparse
from pathlib import Path

from tqdm import tqdm
from rouge import Rouge
import numpy as np
from prettyprinter import pprint
from summa.preprocessing.textcleaner import clean_text_by_sentences

RESULTS_DIR = Path("results/")
RESULTS_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data/")


def head(sentences, ratio):
    length = max(len(sentences) * ratio, 1)
    return sentences[:int(length)]


def random(sentences, ratio):
    length = max(int(len(sentences) * ratio), 1)
    return np.random.choice(np.array(sentences), size=length)


BASELINES = {
    "random": random,
    "head": head
}


def cnndm_inference(ratio):
    """Summary generation for the CNN/DailyMail dataset"""
    for func_name, func in BASELINES.items():
        print("Processing \"{}\"".format(func_name))
        print("=" * 20)
        with open(str(DATA_DIR / "cnndm" / "test.txt.src")) as fin:
            with open(str(
                RESULTS_DIR / "cnndm_baseline_{}_{}.pred".format(
                    func_name, int(ratio * 100))),
                    "w") as fout:
                for line in tqdm(fin.readlines()):
                    sentences = clean_text_by_sentences(line, "english")
                    result = " ".join(
                        [x.text for x in func(sentences, ratio=ratio)])
                    fout.write(result + "\n")


def cnndm_eval(ratio):
    """Evaluation for the CNN/DailyMail dataset"""
    for func_name in BASELINES:
        print("Evaluating \"{}\"".format(func_name))
        print("=" * 20)
        with open(str(RESULTS_DIR / "cnndm_baseline_{}_{}.pred".format(
                func_name, int(ratio * 100)))) as fin:
            predictions = fin.read().split("\n")[:-1]
        with open(str(DATA_DIR / "cnndm" / "test.txt.tgt.tagged")) as fin:
            references = fin.read().replace("<t> ", "").replace(
                "</t> ", "").split("\n")[:-1]
        assert all([len(x) > 0 for x in predictions])
        scores = Rouge().get_scores(predictions, references, avg=True)
        pprint(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate summaries using simple heuristics.')
    parser.add_argument('ratio', type=float, default=0.1, nargs='?',
                        help='Use "ratio * n_total_sentences" in summaries.')
    args = parser.parse_args()
    cnndm_inference(args.ratio)
    cnndm_eval(args.ratio)
