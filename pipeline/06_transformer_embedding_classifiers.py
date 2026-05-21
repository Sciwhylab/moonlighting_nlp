#!/usr/bin/env python
"""Stage 06 - Transformer embeddings + classical classifiers.

An alternative to fine-tuning (stage 05): the pre-trained model is used as a
*frozen* feature extractor. The [CLS] hidden state of each abstract becomes a
fixed-length embedding, and the four-classifier panel (logistic regression,
linear SVM, random forest, gradient boosting) is trained on those embeddings.

This consolidates the two near-duplicate original scripts
(`transformerencoding-classifiers.py` and `transformer-linear_regression.py`)
into one and fixes the broken id/label bookkeeping they contained.

Outputs (in --outdir)
---------------------
embedding_classifier_probabilities.tsv : per-repeat positive-class probs
embedding_classifier_per_repeat.tsv    : metrics for every (model, repeat)
embedding_classifier_summary.tsv       : mean +/- std per model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moonlighting import config  # noqa: E402
from moonlighting.classifiers import run_panel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("embedding_classifiers")


def extract_cls_embeddings(encoded: Dataset, model, tokenizer, device) -> np.ndarray:
    """Return the [CLS] (first-token) hidden state for every example."""

    def _batch(batch):
        inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
        with torch.no_grad():
            last_hidden = model(**inputs).last_hidden_state
        return {"hidden_state": last_hidden[:, 0].cpu().numpy()}

    hidden = encoded.map(_batch, batched=True, batch_size=50)
    return np.array(hidden["hidden_state"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, default=config.TRAIN_CSV)
    p.add_argument("--test", type=Path, default=config.TEST_CSV)
    p.add_argument("--outdir", type=Path, default=config.OUTPUTS_DIR)
    p.add_argument("--model", choices=config.MODEL_CHECKPOINTS, default=config.DEFAULT_MODEL)
    p.add_argument("--n-repeats", type=int, default=config.CLASSIFIER.n_repeats)
    p.add_argument("--base-seed", type=int, default=config.CLASSIFIER.base_seed)
    args = p.parse_args()

    ckpt = config.MODEL_CHECKPOINTS[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Extracting embeddings with %s on %s", ckpt, device)

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt).to(device)

    train_df = pd.read_csv(args.train, sep="\t")
    test_df = pd.read_csv(args.test, sep="\t")
    ds = DatasetDict(
        train=Dataset.from_pandas(train_df, preserve_index=False),
        test=Dataset.from_pandas(test_df, preserve_index=False),
    )

    def tokenize(batch):
        return tokenizer(batch[config.TEXT_COL], padding=True, truncation=True,
                         max_length=config.TRANSFORMER.max_length)

    encoded = ds.map(tokenize, batched=True, batch_size=None, remove_columns=[config.ID_COL])
    encoded.set_format("torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])

    X_train = extract_cls_embeddings(encoded["train"], model, tokenizer, device)
    X_test = extract_cls_embeddings(encoded["test"], model, tokenizer, device)
    y_train = np.array(encoded["train"]["label"])
    y_test = np.array(encoded["test"]["label"])
    log.info("Embeddings: train %s, test %s", X_train.shape, X_test.shape)

    probs, per_repeat, summary = run_panel(
        X_train, y_train, X_test, y_test,
        test_ids=test_df[config.ID_COL],
        n_repeats=args.n_repeats, base_seed=args.base_seed,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    probs.to_csv(args.outdir / "embedding_classifier_probabilities.tsv", sep="\t", index=False)
    per_repeat.to_csv(args.outdir / "embedding_classifier_per_repeat.tsv", sep="\t")
    summary.to_csv(args.outdir / "embedding_classifier_summary.tsv", sep="\t")
    log.info("\n%s", summary.to_string())


if __name__ == "__main__":
    main()
