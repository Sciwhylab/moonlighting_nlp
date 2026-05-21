#!/usr/bin/env python
"""Stage 04 - Word-cloud EDA.

Reproduces the exploratory word clouds: the most frequent terms in the
positive and negative abstracts are collected with a count vectoriser, and the
terms *unique* to each class (set difference) are rendered as a word cloud.
This is the supplementary EDA figure, not a modelling step.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import STOPWORDS, WordCloud

matplotlib.use("Agg")  # headless: write files instead of opening windows

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moonlighting import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("wordcloud")


def top_terms(texts: pd.Series, max_features: int = 100) -> set[str]:
    vec = CountVectorizer(max_df=0.85, max_features=max_features)
    vec.fit(texts)
    return set(vec.get_feature_names_out())


def plot_wordcloud(words: set[str], out_path: Path) -> None:
    text = " ".join(words)
    cloud = WordCloud(stopwords=set(STOPWORDS), background_color="black", colormap="gray").generate(text)
    plt.figure(figsize=(15, 10))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    cloud.to_file(str(out_path))
    plt.close()
    log.info("Wrote %s", out_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=config.PROCESSED, help="Cleaned TSV from stage 02.")
    p.add_argument("--outdir", type=Path, default=config.OUTPUTS_DIR)
    p.add_argument("--max-features", type=int, default=100)
    args = p.parse_args()

    df = pd.read_csv(args.input, sep="\t").dropna(subset=[config.LABEL_COL, config.TEXT_COL])
    pos_terms = top_terms(df.loc[df[config.LABEL_COL] == 1, config.TEXT_COL], args.max_features)
    neg_terms = top_terms(df.loc[df[config.LABEL_COL] == 0, config.TEXT_COL], args.max_features)

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_wordcloud(pos_terms - neg_terms, args.outdir / "wordcloud_positive.png")
    plot_wordcloud(neg_terms - pos_terms, args.outdir / "wordcloud_negative.png")


if __name__ == "__main__":
    main()
