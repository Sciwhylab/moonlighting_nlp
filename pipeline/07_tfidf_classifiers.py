#!/usr/bin/env python
"""Stage 07 - TF-IDF + classical classifiers (Kihara-style baseline).

The baseline modelling track from the manuscript: abstracts are vectorised
with word-level TF-IDF (1-3 grams) and the same four-classifier panel used in
stage 06 is trained on the sparse vectors. Comparing this against the
transformer tracks is the central comparison of the paper.

Outputs (in --outdir)
---------------------
tfidf_classifier_probabilities.tsv : per-repeat positive-class probs
tfidf_classifier_per_repeat.tsv    : metrics for every (model, repeat)
tfidf_classifier_summary.tsv       : mean +/- std per model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moonlighting import config  # noqa: E402
from moonlighting.classifiers import run_panel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("tfidf_classifiers")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, default=config.TRAIN_CSV)
    p.add_argument("--test", type=Path, default=config.TEST_CSV)
    p.add_argument("--outdir", type=Path, default=config.OUTPUTS_DIR)
    p.add_argument("--max-features", type=int, default=config.CLASSIFIER.tfidf_max_features)
    p.add_argument("--n-repeats", type=int, default=config.CLASSIFIER.n_repeats)
    p.add_argument("--base-seed", type=int, default=config.CLASSIFIER.base_seed)
    args = p.parse_args()

    train_df = pd.read_csv(args.train, sep="\t")
    test_df = pd.read_csv(args.test, sep="\t")

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        lowercase=True,
        analyzer="word",
        stop_words="english",
        ngram_range=config.CLASSIFIER.tfidf_ngram_range,
        dtype=np.float32,
    )
    X_train = vectorizer.fit_transform(train_df[config.TEXT_COL])
    X_test = vectorizer.transform(test_df[config.TEXT_COL])
    log.info("TF-IDF: train %s, test %s", X_train.shape, X_test.shape)

    probs, per_repeat, summary = run_panel(
        X_train, train_df[config.LABEL_COL],
        X_test, test_df[config.LABEL_COL],
        test_ids=test_df[config.ID_COL],
        n_repeats=args.n_repeats, base_seed=args.base_seed,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    probs.to_csv(args.outdir / "tfidf_classifier_probabilities.tsv", sep="\t", index=False)
    per_repeat.to_csv(args.outdir / "tfidf_classifier_per_repeat.tsv", sep="\t")
    summary.to_csv(args.outdir / "tfidf_classifier_summary.tsv", sep="\t")
    log.info("\n%s", summary.to_string())


if __name__ == "__main__":
    main()
