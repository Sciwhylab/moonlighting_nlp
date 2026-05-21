"""Abstract text preprocessing with NLTK.

This reproduces the cleaning chain from the original `code1c` notebook export:
lower-casing and punctuation/number stripping, English stop-word removal, and
POS-aware lemmatisation. The per-step helpers are kept separate so individual
stages can be tested or swapped, and `finalpreprocess` chains them in the same
order as the manuscript.
"""

from __future__ import annotations

import re
import string

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Resources required by the cleaning chain. Names changed across NLTK releases,
# so we try the modern names and fall back silently to the legacy ones.
_REQUIRED = [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "wordnet",
    "omw-1.4",
    "stopwords",
]


def ensure_nltk_resources() -> None:
    """Download the NLTK data needed for tokenisation/POS/lemmatisation."""
    for resource in _REQUIRED:
        try:
            nltk.download(resource, quiet=True)
        except Exception:  # noqa: BLE001 - a missing optional name is non-fatal
            pass


_PUNCT_RE = re.compile("[%s]" % re.escape(string.punctuation))
_HTML_RE = re.compile("<.*?>")
_lemmatizer = WordNetLemmatizer()
_STOPWORDS = None  # populated lazily after resources are guaranteed present


def _get_stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def basic_clean(text: str) -> str:
    """Lower-case, strip HTML, punctuation, digits and collapse whitespace."""
    text = str(text).lower().strip()
    text = _HTML_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\[[0-9]*\]", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in _get_stopwords())


def _wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default, also covers nouns


def lemmatize(text: str) -> str:
    tagged = nltk.pos_tag(word_tokenize(text))
    return " ".join(_lemmatizer.lemmatize(tok, _wordnet_pos(tag)) for tok, tag in tagged)


def finalpreprocess(text: str) -> str:
    """Full chain: basic clean -> stop-word removal -> lemmatisation."""
    return lemmatize(remove_stopwords(basic_clean(text)))
