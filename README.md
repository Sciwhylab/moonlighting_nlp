# Moonlighting-protein prediction from PubMed abstracts

Clean, reorganised code for the moonlighting-protein NLP study. The original
analysis (2023) was developed across several Jupyter notebooks; this repository
restructures it into a reproducible, modular pipeline that mirrors the
manuscript's methodology (Supplementary Figure 1: dataset selection → data size
→ language model → classifier → performance quantification). (Citation of publication will be added when available)

The biological task is binary text classification: given the PubMed abstracts
associated with a protein, predict whether the protein is **moonlighting**
(label 1) or **non-moonlighting** (label 0).

## Layout

```
moonlighting_nlp/
├── moonlighting/                 # importable library (written once, reused everywhere)
│   ├── config.py                 # paths, model checkpoints, hyperparameters
│   ├── metrics.py                # get_score + mean±std aggregation (the manuscript panel)
│   ├── preprocessing.py          # NLTK clean → stopword → lemmatise chain
│   └── classifiers.py            # the 4-classifier panel (LR / SVM / RF / GBM)
├── pipeline/                     # numbered, runnable stages
│   ├── 01_compile_dataset.py
│   ├── 02_preprocess_text.py
│   ├── 03_split_dataset.py
│   ├── 04_wordcloud.py
│   ├── 05_finetune_transformer_kfold.py
│   ├── 06_transformer_embedding_classifiers.py
│   ├── 07_tfidf_classifiers.py
│   └── 08_evaluate_transformer_folds.py
├── requirements.txt
└── pyproject.toml
```

Stages read from / write to `data/` and `outputs/` (created on first run);
all paths are defined in `moonlighting/config.py` and overridable per-script.

## How each stage maps to the manuscript

| Stage | Manuscript element | Original script(s) |
|-------|--------------------|--------------------|
| 01 compile dataset            | Dataset construction; PMID de-duplication between classes (Supp. Fig. 2) | `code1_data_compiling` |
| 02 preprocess text            | NLTK preprocessing of abstracts                                          | `code1c_rawtoprocessed` |
| 03 split dataset              | Fixed stratified train/test split (Supp. Table 1)                        | (was inline in `hf-vk`) |
| 04 word cloud                 | Class-distinctive term EDA figure                                        | `code2_TFIDF_based_word_cloud` |
| 05 fine-tune transformer      | **Headline model**: PubMedBERT + classifier head, 5-fold CV (Supp. Tables 3–5) | `hf-vk-3-modified...kfold` |
| 06 embedding + classifiers    | Frozen-transformer embeddings + classical ML                            | `transformerencoding-classifiers`, `transformer-linear_regression` |
| 07 TF-IDF + classifiers       | Baseline (Kihara-style) TF-IDF + classical ML                            | `TFIDF-classifiers...repeat` |
| 08 evaluate folds             | Per-fold, ensemble, abstract- and protein-level scores (Supp. Tables 3–4) | `fold_analysis_transformer` |

The three modelling tracks (05, 06, 07) all consume the **same** fixed test set
from stage 03, which is what makes their metrics directly comparable.

## Running the pipeline

```bash
pip install -r requirements.txt        # or: pip install -e .

# Data preparation
python pipeline/01_compile_dataset.py --positive positive.tsv --negative 'negative_df*'
python pipeline/02_preprocess_text.py
python pipeline/03_split_dataset.py
python pipeline/04_wordcloud.py        # optional EDA

# Modelling (each writes scores to outputs/)
python pipeline/05_finetune_transformer_kfold.py --model pubmedbert
python pipeline/06_transformer_embedding_classifiers.py
python pipeline/07_tfidf_classifiers.py

# Evaluate the fine-tuned transformer's folds + ensemble
python pipeline/08_evaluate_transformer_folds.py
```

To reproduce the Supp. Table 5 model comparison, re-run stages 05 and 08 with
`--model bert` and `--model biobert`.

## What changed from the original notebooks (and why)

These are deliberate, documented changes — none alter the modelling intent:

1. **De-duplicated shared code.** `get_score`, the NLTK chain, the
   `TrainingArguments`, and the four-classifier loop were copy-pasted across
   files. They now live once in the `moonlighting` package.
2. **Meaningful repeats.** The classical-classifier scripts looped 5× with a
   *fixed* seed, so every repeat was identical and the reported std was 0. The
   panel now uses `base_seed + r` per repeat; pass `--n-repeats 1` for a single
   deterministic run.
3. **Fixed broken bookkeeping** in the embedding script: the `id`/`label`
   columns were being truncated/padded by hand (`ID[0:41930]`, `LAB*5`) and a
   `cput()` typo. Ids and labels are now carried through correctly.
4. **Decoupled stage 05 ↔ 08.** Stage 05 saves plain `(n_test, 2)` logit arrays
   per fold instead of pickled `PredictionOutput` namedtuples; stage 08 reads
   them and gets the true labels from the test CSV. This removes the fragile
   object-array indexing in the original `fold_analysis`.
5. **Headless plotting** (`matplotlib Agg`) and removed all Jupyter
   `get_ipython()` magics, dead/commented exploration cells, and duplicate
   `to_csv` writes.
6. **CLI + config.** Every stage is an `argparse` script with sensible defaults
   in `config.py`, so no hard-coded relative paths.

## Open items to confirm before publication

- **Dataset variants.** The manuscript benchmarks several dataset constructions
  (redundant / non-redundant / merged / PubMed-unfiltered / reviewed /
  function-related / randomly-labelled controls) and equal-abstract sampling
  (50–500 abstracts/protein, Supp. Tables 1–2). The provided scripts operate on
  one prepared train/test pair, so that generation code is **not** reconstructed
  here — point each modelling stage at the relevant prepared CSVs via `--train`
  / `--test`, or add a stage 0 that emits each variant.
- **BERT / BioBERT checkpoints.** Only the PubMedBERT id appeared in the
  original code. The BERT/BioBERT ids in `config.py` are the standard public
  checkpoints; confirm they match the exact versions used for Supp. Table 5.

## Data Availability
The PMID's linked to each protein that was used in the paper is given in the uniprot-filtered-organism__Homo+sapiens+(Human)+[9606]_+AND+review file uploaded in the Zenodo link. The list of PMIDS linked to the DNA Binding proteins is listed in the files protein_pmids_unfiltered_negative and protein_pmids_unfiltered_positive.

The datasets used in this study are publicly available on Zenodo:
DOI: 10.5281/zenodo.19104404
