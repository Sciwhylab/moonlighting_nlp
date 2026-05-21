#!/usr/bin/env python
"""Stage 05 - Fine-tune a transformer with 5-fold cross-validation.

This is the headline model of the manuscript: a pre-trained biomedical
language model (PubMedBERT by default; BERT and BioBERT are also configured
for the Supp. Table 5 comparison) fine-tuned for sequence classification.

For each of the K stratified folds of the training set, a fresh model is
fine-tuned on K-1 folds, validated on the held-out fold, and then used to
predict on the *fixed* test set. The per-fold test-set logits are saved so
that stage 08 can compute per-fold and ensemble metrics.

Outputs (in --outdir)
---------------------
fold{i}_test_logits.npy        : (n_test, 2) raw logits on the test set
transformer_fold_runstats.tsv  : training/eval metrics returned per fold
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moonlighting import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("finetune_transformer")


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def load_dataset_dict(train_csv: Path, test_csv: Path) -> DatasetDict:
    train_df = pd.read_csv(train_csv, sep="\t")
    test_df = pd.read_csv(test_csv, sep="\t")
    return DatasetDict(
        train=Dataset.from_pandas(train_df, preserve_index=False),
        test=Dataset.from_pandas(test_df, preserve_index=False),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, default=config.TRAIN_CSV)
    p.add_argument("--test", type=Path, default=config.TEST_CSV)
    p.add_argument("--outdir", type=Path, default=config.OUTPUTS_DIR)
    p.add_argument("--model", choices=config.MODEL_CHECKPOINTS, default=config.DEFAULT_MODEL,
                   help="Which pre-trained checkpoint to fine-tune.")
    p.add_argument("--n-splits", type=int, default=config.TRANSFORMER.n_splits)
    p.add_argument("--epochs", type=int, default=config.TRANSFORMER.num_train_epochs)
    p.add_argument("--seed", type=int, default=config.TRANSFORMER.seed)
    args = p.parse_args()

    cfg = config.TRANSFORMER
    ckpt = config.MODEL_CHECKPOINTS[args.model]
    args.outdir.mkdir(parents=True, exist_ok=True)
    log.info("Fine-tuning %s (%s)", args.model, ckpt)

    tokenizer = AutoTokenizer.from_pretrained(ckpt)

    def tokenize(batch):
        return tokenizer(batch[config.TEXT_COL], padding=True, truncation=True, max_length=cfg.max_length)

    ds = load_dataset_dict(args.train, args.test)
    encoded = ds.map(tokenize, batched=True, batch_size=None)
    keep = ["label", "input_ids", "token_type_ids", "attention_mask"]
    keep = [c for c in keep if c in encoded["train"].column_names]
    encoded.set_format("torch", columns=keep)

    train_labels = np.array(ds["train"][config.LABEL_COL])
    folds = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = folds.split(np.zeros(len(train_labels)), train_labels)

    run_stats = []
    for i, (train_idx, val_idx) in enumerate(splits):
        log.info("=== Fold %d/%d ===", i + 1, args.n_splits)
        fold = DatasetDict(
            train=encoded["train"].select(train_idx),
            validation=encoded["train"].select(val_idx),
            test=encoded["test"],
        )

        model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
        targs = TrainingArguments(
            output_dir=str(args.outdir / f"{args.model}-fold{i}"),
            num_train_epochs=args.epochs,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            weight_decay=cfg.weight_decay,
            logging_steps=len(fold["train"]),
            disable_tqdm=False,
            push_to_hub=False,
            log_level="error",
        )
        trainer = Trainer(
            model=model,
            args=targs,
            compute_metrics=compute_metrics,
            train_dataset=fold["train"],
            eval_dataset=fold["validation"],
            tokenizer=tokenizer,
        )
        trainer.train()

        preds = trainer.predict(fold["test"])
        np.save(args.outdir / f"fold{i}_test_logits.npy", preds.predictions)
        run_stats.append({"fold": i + 1, **preds.metrics})
        log.info("Fold %d test metrics: %s", i + 1, preds.metrics)

    pd.DataFrame(run_stats).to_csv(
        args.outdir / "transformer_fold_runstats.tsv", sep="\t", index=False
    )
    log.info("Saved per-fold logits and run stats to %s", args.outdir)


if __name__ == "__main__":
    main()
