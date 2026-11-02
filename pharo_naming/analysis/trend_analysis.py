"""Direction and size of the change of each metric over the releases.

Releases are ordered historically and numbered 1, 2, 3, ...; slopes are
therefore expressed per release.
"""

from __future__ import annotations

from collections import defaultdict
import math

from scipy.stats import linregress, spearmanr, theilslopes

from ..reporting import fmt
from ..vocabulary import POOLED, PRIMARY_WEIGHTING


METRICS = ("cohesion", "top1_probability", "effective_branching")
METRIC_LABELS = {"cohesion": "Cohesion", "top1_probability": "Top-1"}
KIND_LABELS = {"method": "Selectors", "class": "Class names"}

def describe_trend(positions: list[float], values: list[float]) -> dict:
    """Summarize how ``values`` change along ``positions``."""
    first, last = values[0], values[-1]
    change = last - first
    spearman = spearmanr(positions, values)
    slope, _, low_slope, high_slope = theilslopes(values, positions, alpha=0.95)

    return {
        "n_versions": len(values),
        "first_value": first,
        "last_value": last,
        "absolute_change": change,
        "relative_change_percent": change / first * 100 if first else math.nan,
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "linear_slope": float(linregress(positions, values).slope),
        "theil_sen_slope": float(slope),
        "theil_sen_low_slope": float(low_slope),
        "theil_sen_high_slope": float(high_slope),
    }


def trend_rows(metric_rows: list[dict]) -> list[dict]:
    """One trend row per identifier kind, weighting, prefix length and metric."""
    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in metric_rows:
        key = (
            row["identifier_kind"],
            row["weighting"],
            row["aggregation"],
            row["prefix_length"],
        )
        groups[key].append(row)

    output = []
    for key, rows in groups.items():
        rows = sorted(rows, key=lambda row: row["release_index"])
        identifier_kind, weighting, aggregation, prefix_length = key

        for metric in METRICS:
            observed = [row for row in rows if not math.isnan(row[metric])]
            if len(observed) < 3:
                continue
            output.append({
                "identifier_kind": identifier_kind,
                "weighting": weighting,
                "aggregation": aggregation,
                "prefix_length": prefix_length,
                "metric": metric,
                "first_version": observed[0]["version"],
                "last_version": observed[-1]["version"],
                **describe_trend(
                    [float(row["release_index"]) for row in observed],
                    [float(row[metric]) for row in observed],
                ),
            })

    return output


def print_trends(rows: list[dict]) -> None:
    """Pooled cohesion and top-1 probability of each identifier kind over the releases."""
    for row in rows:
        if (
            row["weighting"] == PRIMARY_WEIGHTING
            and row["aggregation"] == POOLED
            and row["metric"] in METRIC_LABELS
        ):
            print(
                f"{KIND_LABELS[row['identifier_kind']]}, {METRIC_LABELS[row['metric']]}: "
                f"{fmt(row['first_value'])} in Pharo {row['first_version']} to "
                f"{fmt(row['last_value'])} in Pharo {row['last_version']} "
                f"({row['absolute_change']:+.3f}), Spearman rho {row['spearman_rho']:+.3f}, "
                f"slope {row['linear_slope']:+.4f} per release"
            )
