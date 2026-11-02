import math

import numpy as np

from pharo_naming.analysis.robustness_analysis import (
    encode_vocabulary,
    measurement_variants,
    persistent_families,
    persistent_family_series,
    pooled_cohesion,
    size_controlled_samples,
    slopes,
    without_test_identifiers,
)
from pharo_naming.corpus import IdentifierRecord
from pharo_naming.vocabulary import POOLED, prefix_metric_rows


VOCABULARY = sorted({
    "at:", "at:put:", "add:", "addall:", "asstring", "assymbol", "do:", "detect:",
    "isempty", "isnil", "initialize", "printon:", "printstring", "x", "+", ">=",
})


def method(version, package, selector):
    return IdentifierRecord(version, package, "A", "method", "instance", selector)


def reference(values):
    return next(row for row in prefix_metric_rows(values) if row["aggregation"] == POOLED)


def test_coded_cohesion_matches_the_reference_implementation():
    assert math.isclose(
        pooled_cohesion(encode_vocabulary(VOCABULARY)),
        reference(VOCABULARY)["cohesion"],
    )


def test_coded_cohesion_of_a_sample_matches_the_reference_implementation():
    chosen = np.array([0, 3, 4, 5, 9, 10, 12])
    assert math.isclose(
        pooled_cohesion(encode_vocabulary(VOCABULARY), chosen),
        reference([VOCABULARY[index] for index in chosen])["cohesion"],
    )


def test_resampling_is_reproducible_and_bounded():
    corpus = {
        "1.0": [method("1.0", "Kernel", name) for name in VOCABULARY],
        "1.1": [method("1.1", "Kernel", name) for name in VOCABULARY[:12]],
    }
    first = size_controlled_samples(corpus, "method", 8, repetitions=5, seed=1)
    second = size_controlled_samples(corpus, "method", 8, repetitions=5, seed=1)

    assert first.shape == (2, 5)
    assert np.array_equal(first, second)
    assert np.all((first >= 0) & (first <= 1))


def test_full_size_sample_equals_the_whole_vocabulary():
    corpus = {"1.0": [method("1.0", "Kernel", name) for name in VOCABULARY]}
    samples = size_controlled_samples(corpus, "method", len(VOCABULARY), repetitions=3)
    assert np.allclose(samples, reference(VOCABULARY)["cohesion"])


def test_slopes_are_least_squares_slopes():
    samples = np.array([[1.0, 4.0], [2.0, 2.0], [3.0, 0.0]])
    assert np.allclose(slopes([1, 2, 3], samples), [1.0, -2.0])


def test_only_families_present_in_every_release_are_persistent():
    corpus = {
        "1.0": [method("1.0", "Kernel-Objects", "abc"), method("1.0", "Old-Core", "xyz")],
        "1.1": [method("1.1", "Kernel-Classes", "abd"), method("1.1", "New-Core", "xyw")],
    }

    assert persistent_families(corpus, "method") == {"Kernel"}
    assert persistent_families(corpus, "class") == set()

    series = persistent_family_series(corpus, "method")
    assert series["families"] == ["Kernel"]
    assert series["coverage"] == [0.5, 0.5]


def test_test_convention_identifiers_are_removed():
    corpus = {
        "1.0": [
            method("1.0", "Kernel", "testAdd"),
            method("1.0", "Kernel", "add:"),
            method("1.0", "Kernel", "attest"),
            IdentifierRecord("1.0", "Kernel", "SetTest", "class"),
            IdentifierRecord("1.0", "Kernel", "TestCase", "class"),
        ]
    }

    assert math.isclose(without_test_identifiers(corpus, "method")["test_share"][0], 1 / 3)
    assert without_test_identifiers(corpus, "class")["test_share"] == [0.5]


def test_measurement_variants_use_case_and_occurrences():
    corpus = {
        "1.0": [
            method("1.0", "Kernel", "abC"),
            method("1.0", "Kernel", "abc"),
            IdentifierRecord("1.0", "Kernel", "B", "method", "instance", "abc"),
            method("1.0", "Kernel", "abd"),
        ]
    }
    variants = measurement_variants(corpus, "method")

    # lowercase vocabulary abc, abd: "a" always continues with b, "ab" splits evenly
    assert math.isclose(variants["Top-1 probability"][0], (2 * 1.0 + 2 * 0.5) / 4)
    # keeping case gives three names (abC, abc, abd); occurrences give four
    assert variants["Case preserved"][0] != variants["Occurrence-weighted"][0]
