#!/usr/bin/env python
"""Stage 03 - Train/test split.

Produces the fixed `train_data_transformer.csv` / `test_data_transformer.csv`
pair that every downstream modelling stage consumes. The split is stratified
on the label so the moonlighting/non-moonlighting ratio is preserved, and the
seed is fixed so the held-out test set is identical across all models (this is
what makes the transformer, embedding and TF-IDF results directly comparable).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moonlighting import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("split_dataset")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=config.PROCESSED, help="Cleaned TSV from stage 02.")
    p.add_argument("--train-out", type=Path, default=config.TRAIN_CSV)
    p.add_argument("--test-out", type=Path, default=config.TEST_CSV)
    p.add_argument("--test-size", type=float, default=0.1, help="Fraction held out for testing.")
    p.add_argument("--seed", type=int, default=config.TRANSFORMER.seed)
    args = p.parse_args()

    df = pd.read_csv(args.input, sep="\t")
    train, test = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df[config.LABEL_COL]
    )

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.train_out, sep="\t", index=False)
    test.to_csv(args.test_out, sep="\t", index=False)

    log.info("Train: %d rows (%d positive)", len(train), int((train[config.LABEL_COL] == 1).sum()))
    log.info("Test:  %d rows (%d positive)", len(test), int((test[config.LABEL_COL] == 1).sum()))
    log.info("Wrote %s and %s", args.train_out, args.test_out)


if __name__ == "__main__":
    main()
