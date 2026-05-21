"""Central configuration for the moonlighting-protein NLP pipeline.

Every tunable value the manuscript depends on lives here so that a run can be
reproduced or pointed at a different dataset variant by editing one file
(or by overriding on the command line, which each script also supports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Directory layout
# --------------------------------------------------------------------------- #
# Resolved relative to the repository root (the parent of this package).
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
MODELS_DIR = REPO_ROOT / "models"

# Canonical intermediate files produced by the pipeline stages.
RAW_MERGED = DATA_DIR / "raw_merged_data.csv"            # stage 01
PROCESSED = DATA_DIR / "processed_cleaned_data.csv"      # stage 02
TRAIN_CSV = DATA_DIR / "train_data_transformer.csv"      # stage 03
TEST_CSV = DATA_DIR / "test_data_transformer.csv"        # stage 03

# --------------------------------------------------------------------------- #
# Pre-trained language models compared in the manuscript (Supp. Table 5)
# --------------------------------------------------------------------------- #
# PubMedBERT is the model used for the headline results; BERT and BioBERT were
# benchmarked against it. The PubMedBERT id is taken verbatim from the original
# scripts. The BERT/BioBERT ids are the standard public checkpoints; confirm
# they match the exact versions you benchmarked before re-running Table 5.
MODEL_CHECKPOINTS = {
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "bert": "bert-base-uncased",
}
DEFAULT_MODEL = "pubmedbert"

# --------------------------------------------------------------------------- #
# Hyperparameters (from the original TrainingArguments / k-fold setup)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TransformerConfig:
    max_length: int = 512
    num_train_epochs: int = 2
    learning_rate: float = 2e-5
    batch_size: int = 32
    weight_decay: float = 0.01
    n_splits: int = 5            # 5-fold stratified cross-validation
    seed: int = 0


@dataclass(frozen=True)
class ClassifierConfig:
    # TF-IDF vectoriser settings used for the Kihara-style baseline (stage 07).
    tfidf_max_features: int = 20_000
    tfidf_ngram_range: tuple[int, int] = (1, 3)
    # Number of times the classical-model panel is repeated (stages 06/07).
    n_repeats: int = 5
    base_seed: int = 0


TRANSFORMER = TransformerConfig()
CLASSIFIER = ClassifierConfig()

# Columns expected in the model-ready CSVs.
ID_COL = "id"
TEXT_COL = "text"
LABEL_COL = "label"
