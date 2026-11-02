import pytest
import math

from pharo_naming.analysis.corpus_statistics import analyze_records
from pharo_naming.analysis.package_family_analysis import (
    analyze_package_families,
    family_trend_summary,
    group_by_family,
    included_families,
)
from pharo_naming.analysis.global_analysis import (
    analyze_global,
    frequent_prefixes,
)
from pharo_naming.analysis.trend_analysis import describe_trend, trend_rows
from pharo_naming.corpus import IdentifierRecord
from pharo_naming.vocabulary import (
    POOLED,
    identifiers,
    normalize,
    package_family,
    prefix_metric_rows,
)


def method(version, package, class_name, selector, side="instance"):
    return IdentifierRecord(version, package, class_name, "method", side, selector)


def klass(version, package, class_name):
    return IdentifierRecord(version, package, class_name, "class")


RECORDS = [
    klass("1.0", "Kernel-Objects", "Object"),
    klass("1.0", "Kernel-Objects", "Boolean"),
    method("1.0", "Kernel-Objects", "Object", "printOn:"),
    method("1.0", "Kernel-Objects", "Boolean", "printOn:"),
    method("1.0", "Kernel-Objects", "Boolean", "PrintOn:"),
    method("1.0", "Kernel-Objects", "Boolean", ","),
]


def test_normalization_only_strips_and_lowercases():
    assert normalize(" PrintOn: ") == "printon:"
    assert normalize(",") == ","


def test_package_family_is_the_text_before_the_first_hyphen():
    assert package_family("Collections-Abstract") == "Collections"
    assert package_family("Kernel") == "Kernel"
    assert package_family("-Odd") == "Unassigned"


def test_vocabulary_collapses_repeated_identifiers():
    assert identifiers(RECORDS, "method", "vocabulary") == [",", "printon:"]
    assert len(identifiers(RECORDS, "method", "declaration")) == 4
    assert identifiers(RECORDS, "class", "vocabulary") == ["boolean", "object"]


def test_statistics_report_the_analyzed_vocabulary():
    row = analyze_records("1.0", RECORDS)
    assert row["packages"] == 1
    assert row["class_declarations"] == 2
    assert row["method_declarations"] == 4
    assert row["unique_selectors"] == 2
    assert row["unique_class_names"] == 2


def test_pooled_row_combines_prefix_lengths_one_to_four():
    rows = prefix_metric_rows(["abcde", "abcdf", "abxyz"])
    by_length = {row["prefix_length"]: row for row in rows}
    pooled = next(row for row in rows if row["aggregation"] == POOLED)

    support = sum(by_length[length]["support"] for length in (1, 2, 3, 4))
    cohesion = sum(
        by_length[length]["support"] * by_length[length]["cohesion"]
        for length in (1, 2, 3, 4)
    ) / support

    assert pooled["support"] == support
    assert math.isclose(pooled["cohesion"], cohesion)


def test_family_threshold_counts_distinct_identifiers_over_all_releases():
    corpus = {
        "1.0": [method("1.0", "Tiny-Core", "A", f"selector{i}") for i in range(12)],
        "1.1": [method("1.1", "Tiny-Core", "A", f"selector{i}") for i in range(6, 18)],
    }
    grouped = group_by_family(corpus)

    assert included_families(grouped, "method", minimum=18) == {"Tiny": 18}
    assert included_families(grouped, "method", minimum=19) == {}


def test_absent_family_produces_no_row():
    corpus = {
        "1.0": [method("1.0", "Big-Core", "A", f"selector{i}") for i in range(30)],
        "1.1": [method("1.1", "Other-Core", "A", f"message{i}") for i in range(30)],
    }
    metric_rows, summary_rows = analyze_package_families(corpus)

    observed = {(row["package_family"], row["version"]) for row in metric_rows}
    assert observed == {("Big", "1.0"), ("Other", "1.1")}
    assert {row["versions_observed"] for row in summary_rows} == {1}


def test_trend_of_a_straight_line():
    trend = describe_trend([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
    assert math.isclose(trend["spearman_rho"], 1.0)
    assert math.isclose(trend["linear_slope"], 0.1)
    assert math.isclose(trend["theil_sen_slope"], 0.1)
    assert math.isclose(trend["absolute_change"], 0.3)
    assert math.isclose(trend["relative_change_percent"], 300.0)


# The toy corpus has no class names, so their series is constant and Spearman
# warns that its correlation is undefined; only the selector series is checked.
@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
def test_trend_rows_follow_the_historical_order():
    corpus = {
        version: [method(version, "Kernel", "A", name) for name in names]
        for version, names in (
            ("1.0", ["abc", "abd", "xyz"]),
            ("2", ["abc", "abd", "abe"]),
            ("14", ["abc", "abd", "abe", "abf"]),
        )
    }
    rows = trend_rows(analyze_global(corpus))
    pooled = next(
        row for row in rows
        if row["identifier_kind"] == "method"
        and row["weighting"] == "vocabulary"
        and row["aggregation"] == POOLED
        and row["metric"] == "cohesion"
    )

    assert (pooled["first_version"], pooled["last_version"]) == ("1.0", "14")
    assert pooled["n_versions"] == 3


def test_frequent_prefixes_report_shares_of_the_vocabulary():
    names = ["testa", "testb", "testc", "init", "x"]
    assert frequent_prefixes(names, 4, 1) == [("test", 60.0)]


def test_family_trends_are_counted_per_family():
    def rows(family, values):
        return [
            {
                "identifier_kind": "method", "weighting": "vocabulary",
                "aggregation": POOLED, "package_family": family,
                "release_index": index, "cohesion": value,
            }
            for index, value in enumerate(values, start=1)
        ]

    metric_rows = (
        rows("Up", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        + rows("Down", [0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        + rows("Flat", [0.5] * 6)
        + rows("Short", [0.1, 0.9])
    )
    summary_rows = [
        {"identifier_kind": "method", "median_identifiers_per_version": size,
         "weighted_mean_cohesion": cohesion}
        for size, cohesion in ((10, 0.9), (100, 0.6), (1000, 0.4))
    ]
    counts = family_trend_summary(metric_rows, summary_rows, "method")

    assert (counts["families"], counts["increasing"], counts["decreasing"], counts["no_trend"]) == (3, 1, 1, 1)
    assert math.isclose(counts["size_correlation"], -1.0)
