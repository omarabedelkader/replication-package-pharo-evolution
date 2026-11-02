"""Whole-system prefix cohesion of every release, and its trend (printed).

    python -m pharo_naming.analysis.global_analysis
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ..corpus import IdentifierRecord, read_corpus
from ..paths import CORPUS_DIR
from ..releases import release_index
from ..vocabulary import (
    IDENTIFIER_KINDS,
    WEIGHTINGS,
    identifiers,
    prefix_metric_rows,
)
from .trend_analysis import print_trends, trend_rows


# Prefix lengths of the frequent-prefix table (turnover_analysis).
PREFIX_TABLE_LENGTHS = {"method": 4, "class": 2}


def analyze_global(corpus: dict[str, list[IdentifierRecord]]) -> list[dict]:
    """Metric rows of every release, identifier kind, weighting and prefix length."""
    metric_rows: list[dict] = []

    for version, records in corpus.items():
        for identifier_kind in IDENTIFIER_KINDS:
            for weighting in WEIGHTINGS:
                group = {
                    "version": version,
                    "release_index": release_index(version),
                    "identifier_kind": identifier_kind,
                    "weighting": weighting,
                }
                rows = prefix_metric_rows(
                    identifiers(records, identifier_kind, weighting)
                )
                metric_rows.extend({**group, **row} for row in rows)

    return metric_rows


def frequent_prefixes(names: list[str], length: int, count: int | None) -> list[tuple[str, float]]:
    """The ``count`` most frequent prefixes of ``length`` characters (all of
    them when ``count`` is None), with their share of the vocabulary in percent."""
    counts = Counter(name[:length] for name in names if len(name) > length)
    return [(prefix, 100 * total / len(names)) for prefix, total in counts.most_common(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    args = parser.parse_args()

    corpus = read_corpus(args.corpus_dir)
    metric_rows = analyze_global(corpus)

    print_trends(trend_rows(metric_rows))


if __name__ == "__main__":
    main()
