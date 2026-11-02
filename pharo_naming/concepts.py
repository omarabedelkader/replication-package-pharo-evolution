"""Conceptual cohesion of a package (CCP), after Santos et al. (CSMR-WCRE 2014).

Every class is a document made of the words of its name and of the selectors
of its methods.  Documents are weighted with TF-IDF over all the classes of a
release, and the CCP of a group of classes is the mean cosine similarity over
all its pairs of distinct classes.

With unit vectors the sum of all pairwise similarities is
``(||sum of the vectors||^2 - n) / 2``, so the CCP of a group is computed from
the sum of its vectors without forming the pairs.  The same identity gives the
mean similarity of all pairs of a release, which is also the expected CCP of a
random group of classes of any size: a family is conceptually cohesive when
its CCP exceeds that value.

Santos et al. use the identifiers and the comments of the source code and
reduce the vectors with latent semantic indexing.  Our corpus holds names
only, so this is an identifier-based adaptation of their metric.
"""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse


# Words that carry no concept of their own: articles, prepositions,
# conjunctions, and the keyword parts that only introduce an argument.
STOP_WORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "do", "for", "from", "if", "in",
    "into", "is", "it", "of", "on", "or", "the", "to", "with", "without",
    "all", "any", "new", "no", "not", "my", "self", "super", "value",
})
MIN_WORD_LENGTH = 2

_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")


def split_identifier(identifier: str) -> list[str]:
    """Words of a class name or selector: ``at:put:`` gives ``at, put`` and
    ``OBBrowserModel`` gives ``ob, browser, model``.  Digits and operator
    characters separate words and are dropped."""
    return [word.lower() for word in _WORD.findall(identifier)]


def normalize_word(word: str) -> str:
    """Merge the plural and the singular form of a word."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("sses", "xes", "ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def terms(identifier: str) -> list[str]:
    """Informative, normalized words of an identifier."""
    return [
        normalize_word(word)
        for word in split_identifier(identifier)
        if len(word) >= MIN_WORD_LENGTH and word not in STOP_WORDS
    ]


def prefix_words(class_names: Iterable[str], share: float = 0.5) -> set[str]:
    """Words that start at least ``share`` of the class names of a group.

    These are project prefixes such as ``Athens`` or ``Cly``: they make the
    names of a group alike without saying anything about what its classes do.
    """
    names = list(class_names)
    first = Counter(words[0] for words in map(terms, names) if words)
    return {word for word, count in first.items() if count >= share * len(names)}


def tfidf_matrix(documents: Sequence[Counter[str]]) -> sparse.csr_matrix:
    """Unit-length TF-IDF rows, one per document.

    Term frequency is sublinear (1 + log tf) and the inverse document
    frequency is log(N / df), so a word used by every class weighs nothing.
    A document without any weighted word is a zero row.
    """
    vocabulary: dict[str, int] = {}
    rows, columns, counts = [], [], []
    for row, document in enumerate(documents):
        for word, count in document.items():
            rows.append(row)
            columns.append(vocabulary.setdefault(word, len(vocabulary)))
            counts.append(count)

    shape = (len(documents), len(vocabulary))
    frequency = sparse.csr_matrix(
        (np.asarray(counts, dtype=float), (rows, columns)), shape=shape
    )
    document_frequency = np.bincount(columns, minlength=len(vocabulary))
    idf = np.log(len(documents) / np.maximum(document_frequency, 1))

    weights = frequency.copy()
    weights.data = 1.0 + np.log(weights.data)
    weights = weights @ sparse.diags(idf)
    norms = np.sqrt(np.asarray(weights.multiply(weights).sum(axis=1)).ravel())
    scale = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    return sparse.csr_matrix(sparse.diags(scale) @ weights)


def mean_pairwise_similarity(vectors: sparse.csr_matrix) -> float:
    """Mean cosine similarity over all pairs of distinct rows (unit rows).

    A zero row has similarity zero with every other row."""
    n = vectors.shape[0]
    if n < 2:
        return math.nan
    total = np.asarray(vectors.sum(axis=0)).ravel()
    self_similarity = vectors.multiply(vectors).sum()
    return float((total @ total - self_similarity) / (n * (n - 1)))


def random_group_similarities(
    vectors: sparse.csr_matrix,
    size: int,
    repetitions: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """CCP of ``repetitions`` random groups of ``size`` rows."""
    return np.array([
        mean_pairwise_similarity(
            vectors[generator.choice(vectors.shape[0], size=size, replace=False)]
        )
        for _ in range(repetitions)
    ])


def between_group_similarity(first: sparse.csr_matrix, second: sparse.csr_matrix) -> float:
    """Mean cosine similarity over all pairs made of one row of each group.

    With unit rows it is the product of the two sums divided by the number of
    pairs.  Compared with the CCP of each group, it tells whether two groups
    are as alike as the classes inside them."""
    if first.shape[0] == 0 or second.shape[0] == 0:
        return math.nan
    total_first = np.asarray(first.sum(axis=0)).ravel()
    total_second = np.asarray(second.sum(axis=0)).ravel()
    return float(total_first @ total_second / (first.shape[0] * second.shape[0]))
