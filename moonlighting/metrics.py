"""Classification metrics shared by all modelling stages.

The manuscript reports accuracy, F1, Matthews correlation coefficient (MCC),
specificity, sensitivity, precision and AUC for every dataset/model
combination. `get_score` computes that exact set from a probability vector,
using the standard 0.5 decision threshold. `aggregate_scores` turns several
per-fold / per-repeat score dicts into the "mean +/- std" rows reported in
Supplementary Tables 3-5.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)

# Order in which metrics appear in the manuscript tables.
METRIC_ORDER = [
    "accuracy_score",
    "f1_score",
    "matthews_corrcoef",
    "specificity_score",
    "sensitivity_score",
    "precision_score",
    "auc",
]


def get_score(y_true, pred_prob, threshold: float = 0.5) -> dict[str, float]:
    """Compute the manuscript's metric panel from predicted probabilities.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0 = non-moonlighting, 1 = moonlighting).
    pred_prob : array-like
        Predicted probability of the positive (moonlighting) class.
    threshold : float
        Decision threshold for the hard-label metrics.

    Notes
    -----
    `recall_score` and `sensitivity_score` are numerically identical (both are
    the true-positive rate); the manuscript reports sensitivity, so `recall`
    is omitted here to avoid a redundant column.
    """
    y_true = np.asarray(y_true)
    pred_prob = np.asarray(pred_prob, dtype=float)
    pred = pred_prob > threshold

    fpr, tpr, _ = roc_curve(y_true, pred_prob)
    return {
        "accuracy_score": accuracy_score(y_true, pred),
        "f1_score": f1_score(y_true, pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, pred),
        "specificity_score": specificity_score(y_true, pred),
        "sensitivity_score": sensitivity_score(y_true, pred),
        "precision_score": precision_score(y_true, pred, zero_division=0),
        "auc": auc(fpr, tpr),
    }


def scores_to_frame(score_dicts: Iterable[dict], index: Iterable[str]) -> pd.DataFrame:
    """Stack per-fold/per-repeat score dicts into a tidy DataFrame."""
    df = pd.DataFrame(list(score_dicts), index=list(index))
    return df[METRIC_ORDER]


def aggregate_scores(score_dicts: Iterable[dict], decimals: int = 3) -> pd.Series:
    """Return a 'mean +/- std' summary row across folds/repeats."""
    df = pd.DataFrame(list(score_dicts))[METRIC_ORDER]
    mean, std = df.mean(), df.std(ddof=0)
    return pd.Series(
        {m: f"{mean[m]:.{decimals}f}\u00b1{std[m]:.{decimals}f}" for m in METRIC_ORDER}
    )
