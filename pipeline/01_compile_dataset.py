#!/usr/bin/env python
"""Stage 01 - Compile the raw labelled dataset.

Combines moonlighting (positive) and non-moonlighting (negative) abstracts
into a single labelled table. Abstracts whose PMID appears in *both* classes
are dropped so that no article is used as evidence for both labels.

Inputs
------
--positive : a single TSV with the moonlighting abstracts.
--negative : one or more TSVs (glob) with the non-moonlighting abstracts.
Both must contain at least the columns: Protein_id, Abstract, PMID.

Output
------
A TSV with columns [Protein_id, Abstract, Label] where Label is 1 for
moonlighting and 0 for non-moonlighting.
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moonlighting import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("compile_dataset")


def compile_dataset(positive_path: Path, negative_glob: str) -> pd.DataFrame:
    pos = pd.read_csv(positive_path, sep="\t")
    neg_files = sorted(glob.glob(negative_glob))
    if not neg_files:
        raise FileNotFoundError(f"No negative files matched: {negative_glob}")
    log.info("Loaded 1 positive file and %d negative file(s).", len(neg_files))
    neg = pd.concat((pd.read_csv(f, sep="\t") for f in neg_files), ignore_index=True)

    # Remove abstracts (PMIDs) that appear in both classes.
    overlap = set(neg["PMID"]).intersection(pos["PMID"])
    log.info("Dropping %d PMIDs that overlap between classes.", len(overlap))
    neg = neg.loc[~neg["PMID"].isin(overlap)].copy()
    pos = pos.loc[~pos["PMID"].isin(overlap)].copy()

    neg["Label"] = 0
    pos["Label"] = 1

    merged = pd.concat([neg, pos], ignore_index=True)
    merged = merged[["Protein_id", "Abstract", "Label"]].dropna()
    merged = merged.reset_index(drop=True)
    log.info(
        "Compiled %d abstracts (positive=%d, negative=%d).",
        len(merged),
        int((merged["Label"] == 1).sum()),
        int((merged["Label"] == 0).sum()),
    )
    return merged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--positive", type=Path, required=True, help="TSV of positive (moonlighting) abstracts.")
    p.add_argument("--negative", type=str, required=True, help="Glob for negative-abstract TSV(s).")
    p.add_argument("--output", type=Path, default=config.RAW_MERGED, help="Output TSV path.")
    args = p.parse_args()

    df = compile_dataset(args.positive, args.negative)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep="\t", header=True, index=False)
    log.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
