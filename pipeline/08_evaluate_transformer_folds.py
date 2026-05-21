#!/usr/bin/env python
"""Stage 08 - Evaluate transformer cross-validation predictions.

Loads the per-fold test-set logits saved by stage 05, converts them to
positive-class probabilities, and reports:

  * per-fold metrics          (each fold scored against the true test labels)
  * the cross-validation mean +/- std summary (Supp. Table 3 / 4 style)
  * ensemble metrics          (averaging the per-fold probabilities)
  * protein-level metrics      (probabilities averaged within each protein id)

Abstract-level evaluation treats every abstract as one example; protein-level
evaluation first averages the predictions of all abstracts belonging to a
protein, matching the two-level reporting in the manuscript.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moonlighting import config  # noqa: E402
from moonlighting.metrics import aggregate_scores, get_score, scores_to_frame  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("evaluate")


def load_fold_probabilities(outdir: Path, n_folds: int) -> list[np.ndarray]:
    probs = []
    for i in range(n_folds):
        logits = np.load(outdir / f"fold{i}_test_logits.npy", allow_pickle=True)
        probs.append(softmax(logits, axis=1)[:, 1])
    return probs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=config.OUTPUTS_DIR,
                   help="Directory holding fold{i}_test_logits.npy from stage 05.")
    p.add_argument("--test", type=Path, default=config.TEST_CSV,
                   help="Test CSV providing the true labels and protein ids.")
    p.add_argument("--n-folds", type=int, default=config.TRANSFORMER.n_splits)
    args = p.parse_args()

    test_df = pd.read_csv(args.test, sep="\t")
    y_true = test_df[config.LABEL_COL].astype(int).to_numpy()

    fold_probs = load_fold_probabilities(args.outdir, args.n_folds)

    # --- per-fold + cross-validation summary --------------------------------
    fold_scores = [get_score(y_true, fp) for fp in fold_probs]
    per_fold = scores_to_frame(fold_scores, index=[f"fold_{i + 1}" for i in range(args.n_folds)])
    cv_summary = aggregate_scores(fold_scores)

    # --- ensemble (mean of fold probabilities) ------------------------------
    ensemble_prob = np.mean(np.vstack(fold_probs), axis=0)
    ensemble_score = pd.Series(get_score(y_true, ensemble_prob))

    # --- protein-level (average within each protein id) ---------------------
    protein_block = ""
    if config.ID_COL in test_df.columns:
        prot = pd.DataFrame({
            "id": test_df[config.ID_COL].to_numpy(),
            "prob": ensemble_prob,
            "label": y_true,
        })
        grouped = prot.groupby("id").agg(prob=("prob", "mean"), label=("label", "max"))
        protein_score = pd.Series(get_score(grouped["label"], grouped["prob"]))
        protein_block = "\n\n--- Protein-level (ensemble) ---\n" + protein_score.to_string()
        protein_score.to_csv(args.outdir / "transformer_protein_level_score.tsv", sep="\t")

    # --- write + report -----------------------------------------------------
    per_fold.to_csv(args.outdir / "transformer_foldwise_scores.tsv", sep="\t")
    cv_summary.to_csv(args.outdir / "transformer_cv_summary.tsv", sep="\t", header=["mean_pm_std"])
    ensemble_score.to_csv(args.outdir / "transformer_ensemble_score.tsv", sep="\t")

    log.info("\n--- Per-fold (abstract level) ---\n%s", per_fold.to_string())
    log.info("\n--- 5-fold CV summary (mean +/- std) ---\n%s", cv_summary.to_string())
    log.info("\n--- Ensemble (abstract level) ---\n%s%s", ensemble_score.to_string(), protein_block)


if __name__ == "__main__":
    main()
