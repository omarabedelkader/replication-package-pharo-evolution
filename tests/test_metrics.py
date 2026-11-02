from collections import Counter
import math

from pharo_naming.metrics import (
    aggregate_prefix_metrics,
    build_prefix_counts,
    calculate_prefix_metrics,
    normalized_entropy,
)


def test_deterministic_continuation():
    counter = Counter({"b": 10})
    metrics = calculate_prefix_metrics(counter)
    assert metrics.entropy_cohesion == 1.0
    assert metrics.top1_probability == 1.0
    assert metrics.effective_branching == 1.0
    assert normalized_entropy(counter) == 0.0


def test_uniform_binary_continuation():
    metrics = calculate_prefix_metrics(Counter({"b": 5, "c": 5}))
    assert metrics.entropy_cohesion == 0.0
    assert metrics.top1_probability == 0.5
    assert metrics.effective_branching == 2.0


def test_unequal_binary_continuation():
    metrics = calculate_prefix_metrics(Counter({"b": 3, "c": 1}))
    entropy = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert math.isclose(metrics.effective_branching, 2.0 ** entropy)
    assert math.isclose(metrics.entropy_cohesion, 1.0 - entropy)
    assert metrics.top1_probability == 0.75


def test_prefix_counts_skip_identifiers_without_continuation():
    counts = build_prefix_counts(["at", "at:", "add:"], 2)
    assert counts == {"at": Counter({":": 1}), "ad": Counter({"d": 1})}


def test_aggregate_is_weighted_by_support():
    counts = {
        "a": Counter({"b": 3}),            # cohesion 1, support 3
        "c": Counter({"d": 1, "e": 1}),    # cohesion 0, support 2
    }
    aggregate = aggregate_prefix_metrics(counts)
    assert aggregate.support == 5
    assert aggregate.prefixes == 2
    assert math.isclose(aggregate.entropy_cohesion, 3 / 5)


def test_aggregate_of_nothing_is_undefined():
    aggregate = aggregate_prefix_metrics({})
    assert aggregate.support == 0
    assert math.isnan(aggregate.entropy_cohesion)
