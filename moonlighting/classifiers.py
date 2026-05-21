"""The four classical classifiers benchmarked in the manuscript.

Stages 06 (PubMedBERT embeddings) and 07 (TF-IDF) feed different feature
matrices into the *same* panel of models -- logistic regression, linear SVM,
random forest and gradient boosting -- so the model definitions and the
repeat loop live here once.

Faithfulness note: the original notebooks looped five times with a *fixed*
random seed, so every "repeat" produced identical numbers and the reported
std was therefore zero. Here each repeat uses ``base_seed + r`` so the
mean +/- std is meaningful. Set ``n_repeats=1`` to recover a single
deterministic run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn.svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .metrics import aggregate_scores, get_score, scores_to_frame

# Column label -> human-readable model name, in a stable order.
MODELS = ["logistic_regression", "svm", "random_forest", "gradient_boosting"]


def build_classifiers(seed: int) -> dict:
    """Instantiate the four classifiers for one repeat with a given seed."""
    return {
        "logistic_regression": LogisticRegression(
            C=2, class_weight="balanced", max_iter=3000, random_state=seed
        ),
        "svm": sklearn.svm.SVC(probability=True, kernel="linear", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=10, criterion="entropy", random_state=seed
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=1, random_state=seed
        ),
    }


def run_panel(
    X_train,
    y_train,
    X_test,
    y_test,
    test_ids,
    n_repeats: int,
    base_seed: int,
):
    """Train all four classifiers ``n_repeats`` times and score them.

    Returns
    -------
    probs_df : DataFrame
        Long format, one row per (repeat, test sample): the positive-class
        probability from each model plus the protein id and true label.
    per_repeat : DataFrame
        One row per (model, repeat) with the full metric panel.
    summary : DataFrame
        One row per model: mean +/- std across repeats.
    """
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()
    test_ids = list(test_ids)

    prob_records: list[pd.DataFrame] = []
    score_rows: dict[str, list[dict]] = {m: [] for m in MODELS}

    for r in range(n_repeats):
        seed = base_seed + r
        clfs = build_classifiers(seed)

        repeat_probs = {
            "repeat": r + 1,
            "id": test_ids,
            "true_label": y_test,
        }
        for name, clf in clfs.items():
            clf.fit(X_train, y_train)
            prob = clf.predict_proba(X_test)[:, 1]
            repeat_probs[f"{name}_prob"] = prob
            score_rows[name].append(get_score(y_test, prob))

        prob_records.append(pd.DataFrame(repeat_probs))

    probs_df = pd.concat(prob_records, ignore_index=True)

    per_repeat = pd.concat(
        {
            name: scores_to_frame(rows, index=[f"repeat_{i + 1}" for i in range(len(rows))])
            for name, rows in score_rows.items()
        },
        names=["model", "repeat"],
    )

    summary = pd.DataFrame(
        {name: aggregate_scores(rows) for name, rows in score_rows.items()}
    ).T
    summary.index.name = "model"

    return probs_df, per_repeat, summary
