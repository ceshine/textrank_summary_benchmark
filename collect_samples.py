"""Samples are print to stdout"""
import random
import argparse
from pathlib import Path

SOURCES = {
    "Random@0.1": "_baseline_random_10",
    "Head@0.1": "_baseline_head_10",
    "TextRank@0.1": "_original_10",
    "BM25+eps.25@0.1": "_bm25pluseps025_10",
    "USE-base@0.1": "_use_base_10",
    "USE-large@0.1": "_use_large_10",
    "USE-xling@0.1": "_use_xling_10",
}


DATA_DIR = Path("data/")
RESULTS_DIR = Path("results/")


def main(n_samples):
    for i in range(n_samples):
        print("=" * 20)
        print(f"Sample {i + 1}")
        print("=" * 20)
        with open(str(DATA_DIR / "cnndm" / "test.txt.src")) as fin:
            articles = fin.readlines()
        sample_idx = random.randint(0, len(articles))
        print("\n" + articles[sample_idx] + "\n")
        del articles
        with open(str(DATA_DIR / "cnndm" / "test.txt.tgt.tagged")) as fin:
            summaries = fin.readlines()
        print("-" * 20)
        print("Reference")
        print("-" * 20)
        print(summaries[sample_idx].replace(
            "<t> ", "").replace("</t>", "") + "\n")
        for name, suffix in SOURCES.items():
            with open(str(RESULTS_DIR / f"cnndm{suffix}.pred")) as fin:
                summaries = fin.readlines()
            print("-" * 20)
            print(name)
            print("-" * 20)
            print(summaries[sample_idx] + "\n")
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Collect and print sample predictions.')
    parser.add_argument('n', type=int, default=1, nargs='?',
                        help='Number of samples.')
    args = parser.parse_args()
    main(args.n)
