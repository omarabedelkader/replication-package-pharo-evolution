"""Conceptual cohesion of the whole system, and its relation to prefix cohesion over time.

Called by :mod:`conceptual_cohesion_analysis`, which reports these
correlations in the same table as the family-level ones.

RQ4 compares CCP and prefix cohesion over the families of each release.  This
step asks the same question at the level of the whole system: the system CCP
of a release is the mean similarity of all its pairs of classes (the random
baseline of the family analysis), and it is correlated, over the eighteen
releases, with the whole-system prefix cohesion of class names and selectors.

The releases form one ordered sequence, so two series that both drift with
the composition of the system correlate whatever their relation.  Two
controls separate a shared drift from a relation: the same correlation in the
families present in every release, and the correlation of the changes between
consecutive releases.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ..corpus import IdentifierRecord
from ..vocabulary import PRIMARY_WEIGHTING, identifiers, package_family
from .conceptual_cohesion_analysis import class_documents, family_ccp
from .robustness_analysis import encode_vocabulary, persistent_families, pooled_cohesion

Corpus = dict[str, list[IdentifierRecord]]


def release_series(records: list[IdentifierRecord], families: set[str] | None = None) -> dict:
    """System CCP, prefix cohesion, and size of a release, or of some of its families."""
    if families is not None:
        records = [record for record in records if package_family(record.package) in families]
    documents = class_documents(records)
    _, system_ccp, _ = family_ccp(documents)
    return {
        "ccp": system_ccp,
        "class_pc": pooled_cohesion(encode_vocabulary(identifiers(records, "class", PRIMARY_WEIGHTING))),
        "selector_pc": pooled_cohesion(encode_vocabulary(identifiers(records, "method", PRIMARY_WEIGHTING))),
        "families": len({family for family, _ in documents.values()}),
        "classes": len(documents),
    }


def analyze(corpus: Corpus) -> dict[str, list[dict]]:
    persistent = persistent_families(corpus, "class")
    return {
        "whole": [release_series(records) for records in corpus.values()],
        "persistent": [release_series(records, persistent) for records in corpus.values()],
    }


def correlations(series: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    """Spearman correlations over the releases, for each comparison and view."""
    def column(view: str, key: str) -> np.ndarray:
        return np.array([row[key] for row in series[view]], dtype=float)

    def rho(x, y) -> float:
        # The number of persistent families is constant: no correlation.
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            return np.nan
        return float(spearmanr(x, y).statistic)

    order = np.arange(len(series["whole"]))
    result = {}
    for label, (x, y) in {
        "class_pc_ccp": ("class_pc", "ccp"),
        "selector_pc_ccp": ("selector_pc", "ccp"),
        "families_ccp": ("families", "ccp"),
        "families_class_pc": ("families", "class_pc"),
        "order_ccp": (None, "ccp"),
    }.items():
        def values(view, key):
            return order if key is None else column(view, key)
        result[label] = {
            "whole": rho(values("whole", x), values("whole", y)),
            "persistent": rho(values("persistent", x), values("persistent", y)),
            "changes": rho(np.diff(values("whole", x)), np.diff(values("whole", y))) if x else np.nan,
        }
    return result


def report_system_level(corpus: Corpus) -> dict[str, dict[str, float]]:
    """Print the system-level series and return their correlations."""
    series = analyze(corpus)
    for version, whole, persistent in zip(corpus, series["whole"], series["persistent"]):
        print(
            f"Pharo {version:>4}: system CCP {whole['ccp']:.4f} (persistent {persistent['ccp']:.4f}), "
            f"class PC {whole['class_pc']:.3f} (persistent {persistent['class_pc']:.3f}), "
            f"selector PC {whole['selector_pc']:.3f}, {whole['families']} families"
        )
    rho = correlations(series)
    for key, values in rho.items():
        print(f"{key:<20} whole {values['whole']:+.2f}  persistent {values['persistent']:+.2f}  "
              f"changes {values['changes']:+.2f}")
    return rho
