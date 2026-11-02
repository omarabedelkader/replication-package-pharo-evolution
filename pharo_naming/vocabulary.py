"""Turn declaration records into the identifier lists that are analyzed."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from .corpus import IdentifierRecord
from .metrics import aggregate_prefix_metrics, build_prefix_counts


IDENTIFIER_KINDS = ("method", "class")

# "vocabulary" keeps each distinct identifier once and is the primary analysis;
# "declaration" keeps one entry per declaration.
WEIGHTINGS = ("vocabulary", "declaration")
PRIMARY_WEIGHTING = "vocabulary"

PREFIX_LENGTHS = tuple(range(1, 9))
POOLED_LENGTHS = (1, 2, 3, 4)
POOLED_LABEL = "1-4"
EXACT = "exact"
POOLED = "support_weighted_1_4"

UNASSIGNED_FAMILY = "Unassigned"


def normalize(identifier: str) -> str:
    """Lowercase an identifier so that the analysis ignores capitalization."""
    return identifier.strip().lower()


def package_family(package: str) -> str:
    """The part of a package name before its first hyphen."""
    return package.split("-", 1)[0].strip() or UNASSIGNED_FAMILY


def identifiers(
    records: Iterable[IdentifierRecord],
    identifier_kind: str,
    weighting: str,
) -> list[str]:
    """Normalized selectors or class names of ``records``.

    Selectors come from method declarations; class names from class and trait
    declarations.
    """
    if identifier_kind == "method":
        values = [
            normalize(record.selector)
            for record in records
            if record.kind == "method" and record.selector
        ]
    elif identifier_kind == "class":
        values = [
            normalize(record.class_name)
            for record in records
            if record.kind == "class"
        ]
    else:
        raise ValueError(f"Unknown identifier kind: {identifier_kind}")

    values = [value for value in values if value]
    if weighting == "vocabulary":
        return sorted(set(values))
    if weighting == "declaration":
        return values
    raise ValueError(f"Unknown weighting: {weighting}")


def prefix_metric_rows(values: Sequence[str]) -> list[dict]:
    """Metrics of ``values`` for each prefix length, plus the pooled 1-4 row."""
    counts_by_length = {
        length: build_prefix_counts(values, length)
        for length in PREFIX_LENGTHS
    }

    rows = [
        _metric_row(EXACT, length, counts, len(values))
        for length, counts in counts_by_length.items()
    ]

    pooled: dict[str, Counter[str]] = {}
    for length in POOLED_LENGTHS:
        pooled.update(counts_by_length[length])
    rows.append(_metric_row(POOLED, POOLED_LABEL, pooled, len(values)))

    return rows


def _metric_row(aggregation: str, prefix_length: object, counts: dict, size: int) -> dict:
    aggregate = aggregate_prefix_metrics(counts)
    return {
        "aggregation": aggregation,
        "prefix_length": prefix_length,
        "identifiers": size,
        "prefixes": aggregate.prefixes,
        "support": aggregate.support,
        "cohesion": aggregate.entropy_cohesion,
        "top1_probability": aggregate.top1_probability,
        "effective_branching": aggregate.effective_branching,
    }
