"""Prefix cohesion: how concentrated the next character is after a prefix.

For a prefix ``p`` the continuation distribution counts which character
follows ``p`` in the identifiers that start with it.  Its Shannon entropy,
normalized by the maximum entropy for the observed number of branches, lies
in [0, 1]; cohesion is one minus that value.  A set of prefixes is summarized
by the support-weighted mean of the per-prefix values.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Mapping


@dataclass(frozen=True)
class PrefixMetrics:
    """Metrics of one prefix."""

    entropy_cohesion: float
    top1_probability: float
    effective_branching: float
    support: int
    branches: int


@dataclass(frozen=True)
class AggregatePrefixMetrics:
    """Support-weighted means over a set of prefixes."""

    entropy_cohesion: float
    top1_probability: float
    effective_branching: float
    support: int
    prefixes: int


EMPTY_AGGREGATE = AggregatePrefixMetrics(math.nan, math.nan, math.nan, 0, 0)


def shannon_entropy(counter: Mapping[str, int]) -> float:
    """Entropy, in bits, of the distribution described by ``counter``."""
    total = sum(counter.values())
    if total <= 0:
        raise ValueError("Continuation distribution is empty.")
    return -sum(
        (count / total) * math.log2(count / total)
        for count in counter.values()
    )


def normalized_entropy(counter: Mapping[str, int]) -> float:
    """Entropy divided by its maximum for the observed number of branches."""
    entropy = shannon_entropy(counter)
    if len(counter) == 1:
        return 0.0
    return entropy / math.log2(len(counter))


def calculate_prefix_metrics(counter: Mapping[str, int]) -> PrefixMetrics:
    total = sum(counter.values())
    entropy = shannon_entropy(counter)

    return PrefixMetrics(
        entropy_cohesion=1.0 - normalized_entropy(counter),
        top1_probability=max(counter.values()) / total,
        effective_branching=2.0 ** entropy,
        support=total,
        branches=len(counter),
    )


def build_prefix_counts(
    identifiers: Iterable[str],
    prefix_length: int,
) -> dict[str, Counter[str]]:
    """Count, for every prefix of ``prefix_length``, the characters that follow it.

    Identifiers too short to continue after the prefix contribute nothing.
    """
    if prefix_length < 1:
        raise ValueError("prefix_length must be >= 1.")

    result: dict[str, Counter[str]] = defaultdict(Counter)
    for identifier in identifiers:
        if len(identifier) > prefix_length:
            result[identifier[:prefix_length]][identifier[prefix_length]] += 1
    return dict(result)


def aggregate_prefix_metrics(
    counts: Mapping[str, Mapping[str, int]],
) -> AggregatePrefixMetrics:
    """Support-weighted mean of the per-prefix metrics.

    Prefixes of different lengths are distinct keys, so the counts of several
    lengths can be merged into one mapping to pool them.
    """
    if not counts:
        return EMPTY_AGGREGATE

    total_support = 0
    cohesion = 0.0
    top1 = 0.0
    branching = 0.0

    for counter in counts.values():
        metrics = calculate_prefix_metrics(counter)
        total_support += metrics.support
        cohesion += metrics.support * metrics.entropy_cohesion
        top1 += metrics.support * metrics.top1_probability
        branching += metrics.support * metrics.effective_branching

    return AggregatePrefixMetrics(
        entropy_cohesion=cohesion / total_support,
        top1_probability=top1 / total_support,
        effective_branching=branching / total_support,
        support=total_support,
        prefixes=len(counts),
    )
