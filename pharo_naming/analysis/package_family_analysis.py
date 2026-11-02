"""Prefix cohesion inside package families.

    python -m pharo_naming.analysis.package_family_analysis

A package family is the part of a package name before its first hyphen, so
``Collections-Abstract`` belongs to ``Collections``.  Families whose vocabulary
is too small to be meaningful are left out, and a family that is absent from
a release produces no row for that release.  The step writes no file: it
prints the numbers reported in RQ2.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median

from scipy.stats import spearmanr

from ..corpus import IdentifierRecord, read_corpus
from ..paths import CORPUS_DIR
from ..releases import release_index
from ..reporting import fmt
from ..vocabulary import (
    IDENTIFIER_KINDS,
    POOLED,
    PRIMARY_WEIGHTING,
    WEIGHTINGS,
    identifiers,
    package_family,
    prefix_metric_rows,
)


# A family is analyzed when it contributes at least this many distinct
# identifiers over all releases together.
MIN_FAMILY_IDENTIFIERS = 20
MIN_TREND_SNAPSHOTS = 6
TREND_THRESHOLD = 0.5
TABLE_FAMILIES = 10
RANGE_THRESHOLD = 0.15

PANEL_TITLES = {"method": "Method selectors", "class": "Class names"}

FamilyRecords = dict[str, dict[str, list[IdentifierRecord]]]  # version -> family -> records


def group_by_family(corpus: dict[str, list[IdentifierRecord]]) -> FamilyRecords:
    grouped: FamilyRecords = {}
    for version, records in corpus.items():
        by_family: dict[str, list[IdentifierRecord]] = defaultdict(list)
        for record in records:
            by_family[package_family(record.package)].append(record)
        grouped[version] = dict(by_family)
    return grouped


def included_families(
    grouped: FamilyRecords,
    identifier_kind: str,
    minimum: int = MIN_FAMILY_IDENTIFIERS,
) -> dict[str, int]:
    """Families large enough to analyze, with their distinct identifier count."""
    vocabulary: dict[str, set[str]] = defaultdict(set)
    for by_family in grouped.values():
        for family, records in by_family.items():
            vocabulary[family].update(
                identifiers(records, identifier_kind, PRIMARY_WEIGHTING)
            )
    return {
        family: len(names)
        for family, names in vocabulary.items()
        if len(names) >= minimum
    }


def analyze_package_families(
    corpus: dict[str, list[IdentifierRecord]],
    minimum: int = MIN_FAMILY_IDENTIFIERS,
) -> tuple[list[dict], list[dict]]:
    """Return per-release metric rows and one summary row per family."""
    grouped = group_by_family(corpus)
    metric_rows: list[dict] = []
    summary_rows: list[dict] = []

    for identifier_kind in IDENTIFIER_KINDS:
        sizes = included_families(grouped, identifier_kind, minimum)
        pooled_rows: dict[str, list[dict]] = defaultdict(list)

        for version, by_family in grouped.items():
            for family in sorted(sizes):
                for weighting in WEIGHTINGS:
                    values = identifiers(
                        by_family.get(family, []), identifier_kind, weighting
                    )
                    if not values:
                        continue
                    for row in prefix_metric_rows(values):
                        row = {
                            "version": version,
                            "release_index": release_index(version),
                            "package_family": family,
                            "identifier_kind": identifier_kind,
                            "weighting": weighting,
                            **row,
                        }
                        metric_rows.append(row)
                        if weighting == PRIMARY_WEIGHTING and row["aggregation"] == POOLED:
                            pooled_rows[family].append(row)

        for family in sorted(pooled_rows):
            summary_rows.append(
                summarize_family(identifier_kind, family, sizes[family], pooled_rows[family])
            )

    return metric_rows, summary_rows


def summarize_family(
    identifier_kind: str,
    family: str,
    distinct_identifiers: int,
    rows: list[dict],
) -> dict:
    """Summarize the pooled, vocabulary-weighted rows of one family."""
    rows = [row for row in rows if row["support"] > 0]
    weights = [row["identifiers"] for row in rows]
    cohesion = [row["cohesion"] for row in rows]
    top1 = [row["top1_probability"] for row in rows]
    total = sum(weights)

    return {
        "identifier_kind": identifier_kind,
        "package_family": family,
        "versions_observed": len(rows),
        "first_version": rows[0]["version"],
        "last_version": rows[-1]["version"],
        "distinct_identifiers": distinct_identifiers,
        "median_identifiers_per_version": float(median(weights)),
        "weighted_mean_cohesion": sum(c * w for c, w in zip(cohesion, weights)) / total,
        "weighted_mean_top1": sum(t * w for t, w in zip(top1, weights)) / total,
        "min_cohesion": min(cohesion),
        "max_cohesion": max(cohesion),
        "range_cohesion": max(cohesion) - min(cohesion),
    }


def family_trend_summary(
    metric_rows: list[dict],
    summary_rows: list[dict],
    identifier_kind: str,
) -> dict:
    """Count the families whose cohesion rises, falls, or does neither.

    Only families observed in at least ``MIN_TREND_SNAPSHOTS`` snapshots are
    counted.  A family rises or falls when the Spearman correlation between
    its cohesion and the snapshot order reaches ``TREND_THRESHOLD`` in
    absolute value.  ``size_correlation`` relates the vocabulary size of a
    family to its mean cohesion over all included families.
    """
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in metric_rows:
        if (
            row["identifier_kind"] == identifier_kind
            and row["weighting"] == PRIMARY_WEIGHTING
            and row["aggregation"] == POOLED
        ):
            series[row["package_family"]].append((row["release_index"], row["cohesion"]))

    counts = {"families": 0, "increasing": 0, "decreasing": 0, "no_trend": 0}
    for points in series.values():
        if len(points) < MIN_TREND_SNAPSHOTS:
            continue
        counts["families"] += 1
        values = [value for _, value in points]
        if max(values) == min(values):
            counts["no_trend"] += 1
            continue
        rho = spearmanr([index for index, _ in points], values).statistic
        if rho >= TREND_THRESHOLD:
            counts["increasing"] += 1
        elif rho <= -TREND_THRESHOLD:
            counts["decreasing"] += 1
        else:
            counts["no_trend"] += 1

    summaries = [row for row in summary_rows if row["identifier_kind"] == identifier_kind]
    counts["size_correlation"] = float(spearmanr(
        [row["median_identifiers_per_version"] for row in summaries],
        [row["weighted_mean_cohesion"] for row in summaries],
    ).statistic)
    return counts


def top_bottom_rows(
    summary_rows: list[dict],
    identifier_kind: str,
    top_k: int = TABLE_FAMILIES,
) -> list[dict]:
    """The ``top_k`` most and least cohesive families, labelled by group."""
    eligible = [
        row for row in summary_rows
        if row["identifier_kind"] == identifier_kind
    ]

    def ranked(sign: int) -> list[dict]:
        return sorted(
            eligible,
            key=lambda row: (
                sign * row["weighted_mean_cohesion"],
                -row["distinct_identifiers"],
                row["package_family"],
            ),
        )[:top_k]

    return [
        {**row, "group": group}
        for group, rows in (("Highest", ranked(-1)), ("Lowest", ranked(1)))
        for row in rows
    ]


def report(metric_rows: list[dict], summary_rows: list[dict], minimum: int = MIN_FAMILY_IDENTIFIERS) -> None:
    """Print the numbers of RQ2: families analyzed, their trends, their range of
    cohesion, and the most and least cohesive ones."""
    for identifier_kind, label in PANEL_TITLES.items():
        summaries = [row for row in summary_rows if row["identifier_kind"] == identifier_kind]
        counts = family_trend_summary(metric_rows, summary_rows, identifier_kind)
        lifetime = [row for row in summaries if row["versions_observed"] >= MIN_TREND_SNAPSHOTS]
        narrow = sum(row["range_cohesion"] < RANGE_THRESHOLD for row in lifetime)
        print(
            f"{label}: {len(summaries)} families with at least {minimum} distinct "
            f"identifiers; Spearman rho between family size and cohesion "
            f"{counts['size_correlation']:+.2f}"
        )
        print(
            f"  {counts['families']} families present in at least {MIN_TREND_SNAPSHOTS} snapshots: "
            f"{counts['increasing']} increasing, {counts['decreasing']} decreasing, "
            f"{counts['no_trend']} no trend (|rho| >= {TREND_THRESHOLD}); {narrow} of them "
            f"({narrow / len(lifetime):.0%}) vary by less than {RANGE_THRESHOLD}"
        )
        for row in top_bottom_rows(summary_rows, identifier_kind):
            print(
                f"  {row['group']:<7} {row['package_family']:<34} {row['versions_observed']:2} "
                f"snapshots, {row['distinct_identifiers']:6,} identifiers, mean cohesion "
                f"{fmt(row['weighted_mean_cohesion'])}, range {fmt(row['range_cohesion'])}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument(
        "--min-identifiers",
        type=int,
        default=MIN_FAMILY_IDENTIFIERS,
        help="smallest family vocabulary analyzed (default: %(default)s)",
    )
    args = parser.parse_args()

    corpus = read_corpus(args.corpus_dir)
    metric_rows, summary_rows = analyze_package_families(corpus, args.min_identifiers)

    report(metric_rows, summary_rows, args.min_identifiers)


if __name__ == "__main__":
    main()
