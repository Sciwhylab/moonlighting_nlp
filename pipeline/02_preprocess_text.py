#!/usr/bin/env python
"""Stage 02 - Preprocess abstract text.

Applies the NLTK cleaning chain (clean -> stop-word removal -> lemmatisation)
to every abstract and writes a model-ready table. The classical (TF-IDF)
models consume the cleaned text; the transformer models can use either the raw
or cleaned text depending on the dataset variant being evaluated.

Output columns: [id, text, label] where `text` is the cleaned abstract.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moonlighting import config  # noqa: E402
from moonlighting.preprocessing import ensure_nltk_resources, finalpreprocess  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("preprocess_text")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=config.RAW_MERGED, help="Raw merged TSV from stage 01.")
    p.add_argument("--output", type=Path, default=config.PROCESSED, help="Output TSV with cleaned text.")
    args = p.parse_args()

    log.info("Ensuring NLTK resources are available ...")
    ensure_nltk_resources()

    df = pd.read_csv(args.input, sep="\t")
    log.info("Cleaning %d abstracts (this can take a while) ...", len(df))
    df["clean_text"] = df["Abstract"].astype(str).map(finalpreprocess)

    # Reshape into the canonical id/text/label table used downstream.
    out = (
        df[["Protein_id", "clean_text", "Label"]]
        .rename(columns={"Protein_id": config.ID_COL,
                         "clean_text": config.TEXT_COL,
                         "Label": config.LABEL_COL})
        .dropna(subset=[config.TEXT_COL, config.LABEL_COL])
        .reset_index(drop=True)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, sep="\t", index=False)
    log.info("Wrote %s (%d rows)", args.output, len(out))


if __name__ == "__main__":
    main()
