"""Robustness of the whole-system trends.

    python -m pharo_naming.analysis.robustness_analysis

Four questions are asked of the pooled prefix cohesion (prefix lengths 1-4):

1. Prefix length.  Does the trend hold at each prefix length separately?
2. Vocabulary size.  Does it hold when every release contributes a random
   sample of the same number of distinct identifiers?
3. Composition.  Does it hold when only the package families present in
   every release are kept?
4. Test code.  Does it hold without the identifiers that follow the unit-test
   naming convention (``testSomething`` selectors, ``SomethingTest`` classes)?
5. Measurement.  Does it hold when capitalization is kept, when every
   declaration counts instead of every distinct name, and when the entropy
   measure is replaced by the probability of the most frequent next character?
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from ..corpus import IdentifierRecord, read_corpus
from ..paths import CORPUS_DIR, RESULTS_DIR
from ..plotting import BLUE, INK, MUTED, ORANGE, TEXT_WIDTH, pyplot, save_figure, version_axis
from ..releases import release_index
from ..reporting import HEADER, fmt, signed, write_text
from ..vocabulary import (
    EXACT,
    IDENTIFIER_KINDS,
    POOLED,
    POOLED_LENGTHS,
    PRIMARY_WEIGHTING,
    identifiers,
    package_family,
    prefix_metric_rows,
)
from .trend_analysis import describe_trend


KIND_LABELS = {
    "method": "Selectors",
    "class": "Class names",
    "method_without_tests": "Selectors without test names",
}
PANEL_TITLES = {"method": "Method selectors", "class": "Class names"}

# Size-controlled resampling.  Each sample size is below the smallest release
# vocabulary of its kind (28,888 selectors and 2,982 class names, Pharo 1.0).
SAMPLE_SIZES = {"method": 20_000, "class": 1_500}
REPETITIONS = 500
SEED = 2027
INTERVAL = (2.5, 97.5)

# Normalized (lowercase) identifiers that follow the SUnit naming convention.
TEST_CONVENTION = {
    "method": lambda name: name.startswith("test"),
    "class": lambda name: name.endswith(("test", "tests")),
}

_CONTINUATION_BITS = 21  # enough for any Unicode code point

Corpus = dict[str, list[IdentifierRecord]]


# -- pooled cohesion on integer-coded vocabularies ---------------------------

def encode_vocabulary(values: Sequence[str]) -> list[np.ndarray]:
    """Code every (prefix, next character) observation as one integer.

    Returns one array per pooled prefix length, aligned with ``values``; an
    identifier too short to continue after the prefix is coded -1.
    """
    encoded = []
    for length in POOLED_LENGTHS:
        prefix_codes: dict[str, int] = {}
        keys = np.full(len(values), -1, dtype=np.int64)
        for position, value in enumerate(values):
            if len(value) > length:
                prefix = prefix_codes.setdefault(value[:length], len(prefix_codes))
                keys[position] = (prefix << _CONTINUATION_BITS) | ord(value[length])
        encoded.append(keys)
    return encoded


def pooled_cohesion(encoded: list[np.ndarray], sample: np.ndarray | None = None) -> float:
    """Support-weighted cohesion pooled over prefix lengths 1-4.

    Computes the same quantity as :func:`metrics.aggregate_prefix_metrics`,
    on the coded form, optionally for the identifiers selected by ``sample``.
    """
    weighted_cohesion = 0.0
    total_support = 0

    for keys in encoded:
        if sample is not None:
            keys = keys[sample]
        keys = keys[keys >= 0]
        if not keys.size:
            continue

        observations, counts = np.unique(keys, return_counts=True)
        prefixes = observations >> _CONTINUATION_BITS
        starts = np.concatenate(([0], np.flatnonzero(np.diff(prefixes)) + 1))

        support = np.add.reduceat(counts, starts)
        branches = np.diff(np.concatenate((starts, [counts.size])))
        entropy = np.log2(support) - np.add.reduceat(counts * np.log2(counts), starts) / support
        maximum = np.log2(np.maximum(branches, 2))
        cohesion = np.where(branches > 1, 1.0 - entropy / maximum, 1.0)

        weighted_cohesion += float(np.sum(support * cohesion))
        total_support += int(support.sum())

    return weighted_cohesion / total_support if total_support else float("nan")


# -- 1. prefix length --------------------------------------------------------

def prefix_length_series(
    corpus: Corpus,
    without_tests: bool = False,
) -> dict[str, dict[object, list[float]]]:
    """Cohesion per release for each prefix length and for the pooled score."""
    series: dict[str, dict[object, list[float]]] = {}
    for identifier_kind in IDENTIFIER_KINDS:
        by_length: dict[object, list[float]] = {}
        for records in corpus.values():
            names = identifiers(records, identifier_kind, PRIMARY_WEIGHTING)
            if without_tests:
                names = [name for name in names if not TEST_CONVENTION[identifier_kind](name)]
            rows = prefix_metric_rows(names)
            for row in rows:
                if row["aggregation"] == POOLED or (
                    row["aggregation"] == EXACT and row["prefix_length"] in POOLED_LENGTHS
                ):
                    by_length.setdefault(row["prefix_length"], []).append(row["cohesion"])
        series[identifier_kind] = by_length
    return series


# -- 2. vocabulary size ------------------------------------------------------

def size_controlled_samples(
    corpus: Corpus,
    identifier_kind: str,
    sample_size: int,
    repetitions: int = REPETITIONS,
    seed: int = SEED,
) -> np.ndarray:
    """Pooled cohesion of equal-size random samples: releases x repetitions."""
    generator = np.random.default_rng(seed)
    samples = np.empty((len(corpus), repetitions))

    for row, records in enumerate(corpus.values()):
        vocabulary = identifiers(records, identifier_kind, PRIMARY_WEIGHTING)
        if len(vocabulary) < sample_size:
            raise ValueError(
                f"sample size {sample_size} exceeds a release vocabulary of {len(vocabulary)}"
            )
        encoded = encode_vocabulary(vocabulary)
        for repetition in range(repetitions):
            chosen = generator.choice(len(vocabulary), size=sample_size, replace=False)
            samples[row, repetition] = pooled_cohesion(encoded, chosen)

    return samples


def slopes(positions: Sequence[float], samples: np.ndarray) -> np.ndarray:
    """Least-squares slope of every column of ``samples`` along ``positions``."""
    centered = np.asarray(positions, dtype=float)
    centered = centered - centered.mean()
    return centered @ (samples - samples.mean(axis=0)) / (centered @ centered)


# -- 3. composition ----------------------------------------------------------

def persistent_families(corpus: Corpus, identifier_kind: str) -> set[str]:
    """Families that contribute identifiers of this kind to every release."""
    present = [
        {
            package_family(record.package)
            for record in records
            if record.kind == identifier_kind
        }
        for records in corpus.values()
    ]
    return set.intersection(*present)


def persistent_family_series(corpus: Corpus, identifier_kind: str) -> dict:
    """Cohesion and vocabulary coverage of the persistent families per release."""
    families = persistent_families(corpus, identifier_kind)
    cohesion, coverage = [], []

    for records in corpus.values():
        whole = identifiers(records, identifier_kind, PRIMARY_WEIGHTING)
        kept = identifiers(
            [record for record in records if package_family(record.package) in families],
            identifier_kind,
            PRIMARY_WEIGHTING,
        )
        cohesion.append(pooled_cohesion(encode_vocabulary(kept)))
        coverage.append(len(kept) / len(whole))

    return {"families": sorted(families), "cohesion": cohesion, "coverage": coverage}


# -- 4. test code ------------------------------------------------------------

def without_test_identifiers(corpus: Corpus, identifier_kind: str) -> dict:
    """Cohesion per release after removing test-convention identifiers."""
    is_test = TEST_CONVENTION[identifier_kind]
    cohesion, share = [], []

    for records in corpus.values():
        whole = identifiers(records, identifier_kind, PRIMARY_WEIGHTING)
        kept = [name for name in whole if not is_test(name)]
        cohesion.append(pooled_cohesion(encode_vocabulary(kept)))
        share.append(1 - len(kept) / len(whole))

    return {"cohesion": cohesion, "test_share": share}


# -- 5. measurement choices --------------------------------------------------

def measurement_variants(corpus: Corpus, identifier_kind: str) -> dict[str, list[float]]:
    """Per-release series under three alternative ways of measuring."""
    variants: dict[str, list[float]] = {
        "Case preserved": [],
        "Occurrence-weighted": [],
        "Top-1 probability": [],
    }

    for records in corpus.values():
        if identifier_kind == "method":
            names = {record.selector for record in records if record.kind == "method"}
        else:
            names = {record.class_name for record in records if record.kind == "class"}
        variants["Case preserved"].append(pooled_cohesion(encode_vocabulary(sorted(names))))

        occurrences = identifiers(records, identifier_kind, "declaration")
        variants["Occurrence-weighted"].append(pooled_cohesion(encode_vocabulary(occurrences)))

        rows = prefix_metric_rows(identifiers(records, identifier_kind, PRIMARY_WEIGHTING))
        pooled = next(row for row in rows if row["aggregation"] == POOLED)
        variants["Top-1 probability"].append(pooled["top1_probability"])

    return variants


# -- reports -----------------------------------------------------------------

def print_prefix_lengths(positions: list[int], series: dict) -> None:
    """Trend of the cohesion at each prefix length and of the pooled score."""
    for identifier_kind, by_length in series.items():
        for length, values in by_length.items():
            trend = describe_trend(positions, values)
            print(
                f"{KIND_LABELS[identifier_kind]}, prefix length {length}: "
                f"{fmt(trend['first_value'])} to {fmt(trend['last_value'])} "
                f"({trend['absolute_change']:+.3f}), Spearman rho {trend['spearman_rho']:+.2f}, "
                f"slope {trend['linear_slope']:+.4f}"
            )


# Analyses shown in the paper's table, in its order (the labels of some rows
# carry a sample size in parentheses, so they are matched on their start).
PAPER_ANALYSES = ("Whole system", "Equal-size samples", "Without test names", "Persistent families")


def write_paper_robustness_table(path: Path, rows: list[dict]) -> None:
    """The table of the paper: the change between the first and the last
    release, for four of the analyses, selectors and class names side by side."""
    def change(identifier_kind: str, analysis: str) -> str:
        matches = [
            row for row in rows
            if row["identifier_kind"] == identifier_kind and row["analysis"].startswith(analysis)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one {analysis!r} row for {identifier_kind}, found {len(matches)}.")
        return signed(matches[0]["last"] - matches[0]["first"], 3)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Change in prefix cohesion between \ph 1.0 and 14 under different conditions.}",
        r"\label{tab:robustness}",
        r"\footnotesize",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        HEADER + r"Analysis & Selectors & Class names\\",
        r"\midrule",
    ]
    for analysis in PAPER_ANALYSES:
        lines.append(f"{analysis} & {change('method', analysis)} & {change('class', analysis)}\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(path, lines)


def latex_rho(value: float) -> str:
    """A Spearman coefficient with an explicit sign, as LaTeX."""
    return f"{'+' if value >= 0 else '$-$'}{abs(value):.2f}"


def write_robustness_table(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pooled prefix cohesion (lengths 1--4) under the robustness analyses: first and last release, change, Spearman $\rho$ with the release order, and slope per release (green: increase, red: decrease). Class names are declared once each, so counting declarations leaves their series unchanged.}",
        r"\label{tab:robustness-full}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        HEADER + r"Identifiers & Analysis & First & Last & Change & $\rho$ & Slope\\",
        r"\midrule",
    ]
    previous = None
    for row in rows:
        label = KIND_LABELS[row["identifier_kind"]]
        if previous and label != previous:
            lines.append(r"\midrule")
        lines.append(
            f"{'' if label == previous else label} & {row['analysis']} & "
            f"{fmt(row['first'])} & {fmt(row['last'])} & "
            f"{signed(row['last'] - row['first'], 3)} & {latex_rho(row['rho'])} & "
            f"{signed(row['slope'], 4)}\\\\"
        )
        previous = label
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(path, lines)


# Events of the history that the decomposition figure points at:
# (identifier kind, release after the event, label, side of the line).
EVENTS = (
    ("class", "2", "OmniBrowser\n(OB) leaves", "right"),
    ("class", "8", "Spec 2, Calypso, Clap,\nand others enter", "right"),
)


def write_decomposition_figure(
    path: Path,
    versions: list[str],
    whole: dict[str, list[float]],
    decomposed: dict[str, tuple[str, list[float]]],
) -> None:
    """The main finding in one figure: each whole-system series next to the
    series that removes its driver (test selectors; families that come and go).

    Both panels share the y scale, so the two trends are compared at the same
    magnitude; both panels show its tick labels.
    """
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.25), sharey=True, constrained_layout=True)
    positions = np.arange(len(versions))
    colors = {"method": BLUE, "class": ORANGE}

    for ax, identifier_kind in zip(axes, IDENTIFIER_KINDS):
        label, values = decomposed[identifier_kind]
        overall = whole[identifier_kind]
        color = colors[identifier_kind]
        ax.fill_between(positions, overall, values, color=color, alpha=0.12, linewidth=0)
        ax.plot(positions, overall, color=INK, marker="o", markersize=2.5)
        ax.plot(positions, values, color=color, linestyle="--", marker="s", markersize=2.5)
        ax.set_title(PANEL_TITLES[identifier_kind], loc="left")
        version_axis(ax, versions, rotation=90)

        for text, series, offset in (("Whole system", overall, 0), (label, values, 0)):
            ax.annotate(text, xy=(positions[-1], series[-1]), xytext=(4, offset),
                        textcoords="offset points", va="center", fontsize=7, color=INK,
                        annotation_clip=False)

        for kind, release, text, side in EVENTS:
            if kind != identifier_kind:
                continue
            x = versions.index(release) - 0.5
            ax.axvline(x, color=MUTED, linewidth=0.6, linestyle=":")
            ax.annotate(text, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3 if side == "right" else -3, -2), textcoords="offset points",
                        ha="left" if side == "right" else "right", va="top", fontsize=6.5,
                        color=MUTED)

    middle = versions.index("10")
    axes[0].annotate("test selectors", xy=(middle, (whole["method"][middle] + decomposed["method"][1][middle]) / 2),
                     ha="center", va="center", fontsize=6.5, color=INK, style="italic")
    axes[0].set_ylabel("Prefix cohesion")
    # Shared scale, but each panel keeps its own tick labels.
    axes[1].tick_params(axis="y", labelleft=True)
    save_figure(fig, path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    corpus = read_corpus(args.corpus_dir)
    versions = list(corpus)
    positions = [release_index(version) for version in versions]
    outputs = []

    # 1. Prefix length.
    by_length = prefix_length_series(corpus)
    original = {
        kind: next(values for key, values in lengths.items() if key not in POOLED_LENGTHS)
        for kind, lengths in by_length.items()
    }
    table_series = dict(by_length)
    table_series["method_without_tests"] = prefix_length_series(corpus, without_tests=True)["method"]
    print_prefix_lengths(positions, table_series)

    rows = []
    persistent, untested = {}, {}

    for identifier_kind in IDENTIFIER_KINDS:
        whole = describe_trend(positions, original[identifier_kind])
        rows.append({
            "identifier_kind": identifier_kind,
            "analysis": "Whole system",
            "first": whole["first_value"],
            "last": whole["last_value"],
            "slope": whole["linear_slope"],
            "rho": whole["spearman_rho"],
        })

        # 2. Vocabulary size.
        size = SAMPLE_SIZES[identifier_kind]
        samples = size_controlled_samples(
            corpus, identifier_kind, size, args.repetitions, args.seed
        )
        means = samples.mean(axis=1)
        sample_slopes = slopes(positions, samples)
        low, high = np.percentile(sample_slopes, INTERVAL)
        print(
            f"{KIND_LABELS[identifier_kind]}: {args.repetitions} equal-size samples of {size:,}; "
            f"{INTERVAL[0]}th to {INTERVAL[1]}th percentile of the slope [{low:+.4f}, {high:+.4f}]; "
            f"{int((sample_slopes > 0).sum())} of {len(sample_slopes)} slopes positive, "
            f"{int((sample_slopes < 0).sum())} negative"
        )
        rows.append({
            "identifier_kind": identifier_kind,
            "analysis": f"Equal-size samples ({size:,})",
            "first": float(means[0]),
            "last": float(means[-1]),
            "slope": float(sample_slopes.mean()),
            "rho": describe_trend(positions, list(means))["spearman_rho"],
        })

        # 3. Composition.
        kept = persistent_family_series(corpus, identifier_kind)
        persistent[identifier_kind] = kept["cohesion"]
        trend = describe_trend(positions, kept["cohesion"])
        rows.append({
            "identifier_kind": identifier_kind,
            "analysis": f"Persistent families ({len(kept['families'])})",
            "first": trend["first_value"],
            "last": trend["last_value"],
            "slope": trend["linear_slope"],
            "rho": trend["spearman_rho"],
        })
        print(
            f"{KIND_LABELS[identifier_kind]}: {len(kept['families'])} persistent families "
            f"cover {min(kept['coverage']):.0%} to {max(kept['coverage']):.0%} of the "
            f"release vocabulary ({kept['coverage'][0]:.0%} in Pharo {versions[0]}, "
            f"{kept['coverage'][-1]:.0%} in Pharo {versions[-1]}); Spearman rho of their "
            f"series {trend['spearman_rho']:+.2f}"
        )
        print(f"  families: {', '.join(kept['families'])}")

        # 4. Test code.
        tests = without_test_identifiers(corpus, identifier_kind)
        untested[identifier_kind] = tests["cohesion"]
        trend = describe_trend(positions, tests["cohesion"])
        rows.append({
            "identifier_kind": identifier_kind,
            "analysis": "Without test names",
            "first": trend["first_value"],
            "last": trend["last_value"],
            "slope": trend["linear_slope"],
            "rho": trend["spearman_rho"],
        })
        print(
            f"{KIND_LABELS[identifier_kind]}: test identifiers are "
            f"{tests['test_share'][0]:.0%} of the vocabulary in Pharo {versions[0]} and "
            f"{tests['test_share'][-1]:.0%} in Pharo {versions[-1]}; without them "
            f"Spearman rho is {trend['spearman_rho']:+.2f}"
        )
        print(
            f"  share of test identifiers per release: "
            f"{', '.join(f'{v} {s:.1%}' for v, s in zip(versions, tests['test_share']))} "
            f"(from {min(tests['test_share']):.1%} to {max(tests['test_share']):.1%})"
        )
        print(
            f"  cohesion without them per release: "
            f"{', '.join(f'{v} {c:.4f}' for v, c in zip(versions, tests['cohesion']))} "
            f"(highest in Pharo {versions[int(np.argmax(tests['cohesion']))]})"
        )

        # 5. Measurement choices.
        for label, values in measurement_variants(corpus, identifier_kind).items():
            trend = describe_trend(positions, values)
            rows.append({
                "identifier_kind": identifier_kind,
                "analysis": label,
                "first": trend["first_value"],
                "last": trend["last_value"],
                "slope": trend["linear_slope"],
                "rho": trend["spearman_rho"],
            })

    outputs.append(args.results_dir / "robustness_trends.tex")
    write_paper_robustness_table(outputs[-1], rows)
    outputs.append(args.results_dir / "robustness_full.tex")
    write_robustness_table(outputs[-1], rows)
    outputs.append(args.results_dir / "cohesion_decomposition.tex")
    write_decomposition_figure(
        outputs[-1],
        versions,
        original,
        {
            "method": ("Without test\nselectors", untested["method"]),
            "class": ("Persistent\nfamilies", persistent["class"]),
        },
    )

    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
