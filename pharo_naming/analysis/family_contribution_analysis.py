"""Which package families drive the whole-system trends.

    python -m pharo_naming.analysis.family_contribution_analysis

The contribution of a family is measured by leaving it out: the whole-system
series is recomputed without the declarations of the family, and the
contribution is the difference between the original change and the change
without the family.  A family whose removal makes the class-name decline
smaller contributes to that decline.  Contributions are reported for the
change between the first and the last release and for the slope over all
releases.  They interact, so the contributions of several families do not add
up.

The same is done for groups of families: those that are absent from the first
release (they entered later) and those that are absent from the last release
(they left).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..corpus import IdentifierRecord, read_corpus
from ..paths import CORPUS_DIR, RESULTS_DIR
from ..releases import release_index
from ..plotting import GREEN, INK, MUTED, PURPLE, FIGURE_HEIGHT, TEXT_WIDTH, pyplot, save_figure
from ..vocabulary import IDENTIFIER_KINDS, PRIMARY_WEIGHTING, identifiers, package_family
from .robustness_analysis import encode_vocabulary, pooled_cohesion, slopes


KIND_LABELS = {"method": "Method selectors", "class": "Class names"}
EN_DASH = "\u2013"

# Families shown in the figure: those that push the whole-system series most
# in the direction of its trend, then those that push it most against it.
WITH_TREND = {"method": 4, "class": 6}
AGAINST_TREND = {"method": 3, "class": 3}

Corpus = dict[str, list[IdentifierRecord]]
NameFamilies = dict[str, frozenset[str]]  # identifier -> families that declare it


def name_families(records: list[IdentifierRecord], identifier_kind: str) -> NameFamilies:
    """Every distinct identifier of a release with the families that declare it."""
    families: dict[str, set[str]] = {}
    for record in records:
        for name in identifiers([record], identifier_kind, PRIMARY_WEIGHTING):
            families.setdefault(name, set()).add(package_family(record.package))
    return {name: frozenset(families[name]) for name in sorted(families)}


def cohesion_series(
    releases: list[NameFamilies],
    encoded: list[list[np.ndarray]],
    excluded: set[str],
) -> np.ndarray:
    """Pooled cohesion per release without the families in ``excluded``.

    An identifier is kept while a family that is not excluded declares it.
    """
    series = []
    for release, codes in zip(releases, encoded):
        kept = np.flatnonzero([bool(declaring - excluded) for declaring in release.values()])
        series.append(pooled_cohesion(codes, kept))
    return np.array(series)


def family_presence(versions: list[str], releases: list[NameFamilies]) -> dict[str, list[str]]:
    """Releases in which each family contributes identifiers."""
    presence: dict[str, list[str]] = {}
    for version, release in zip(versions, releases):
        for family in set().union(*release.values()):
            presence.setdefault(family, []).append(version)
    return presence


def family_size(release: NameFamilies, family: str) -> int:
    return sum(1 for declaring in release.values() if family in declaring)


def contributions(corpus: Corpus, identifier_kind: str) -> dict:
    """Leave-one-family-out contribution of every family, and of two groups."""
    versions = list(corpus)
    positions = [release_index(version) for version in versions]
    releases = [name_families(records, identifier_kind) for records in corpus.values()]
    encoded = [encode_vocabulary(list(release)) for release in releases]

    def change_and_slope(series: np.ndarray) -> tuple[float, float]:
        return float(series[-1] - series[0]), float(slopes(positions, series[:, None])[0])

    change, slope = change_and_slope(cohesion_series(releases, encoded, set()))
    presence = family_presence(versions, releases)

    rows = []
    for family, present in presence.items():
        without_change, without_slope = change_and_slope(cohesion_series(releases, encoded, {family}))
        rows.append({
            "family": family,
            "present": present,
            "first_size": family_size(releases[0], family),
            "last_size": family_size(releases[-1], family),
            "change": change - without_change,
            "slope_share": (slope - without_slope) / slope,
        })
    # Largest share of the trend first: those families push in its direction.
    rows.sort(key=lambda row: row["slope_share"], reverse=True)

    groups = {
        "entered": {f for f, present in presence.items() if versions[0] not in present},
        "left": {f for f, present in presence.items() if versions[-1] not in present},
    }
    group_rows = {}
    for name, families in groups.items():
        without_change, without_slope = change_and_slope(cohesion_series(releases, encoded, families))
        group_rows[name] = {
            "families": len(families),
            "change_without": without_change,
            "slope_without": without_slope,
        }

    return {"change": change, "slope": slope, "families": rows, "groups": group_rows}


def _presence_label(present: list[str], versions: list[str]) -> str:
    if len(present) == len(versions):
        return "all"
    return f"{present[0]}--{present[-1]}" if len(present) > 1 else present[0]


def selected_families(result: dict, identifier_kind: str) -> list[dict]:
    """The families that push most with the trend, then most against it."""
    rows = result["families"]
    return rows[:WITH_TREND[identifier_kind]] + rows[-AGAINST_TREND[identifier_kind]:]


def write_contribution_figure(path: Path, versions: list[str], results: dict[str, dict]) -> None:
    """Diverging bars: the share of the slope of each trend due to one family.

    A positive share pushes the series in the direction of its trend (green), a
    negative one against it (purple); the bars therefore also differ by side.
    """
    plt = pyplot()
    # Same order as Figures 1 and 3: method selectors left, class names right.
    kinds = ("method", "class")
    counts = [len(selected_families(results[kind], kind)) for kind in kinds]
    # Full text width, the two panels side by side like Figure 1; the shorter
    # panel is padded so that bars have the same height in both.
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, FIGURE_HEIGHT), sharex=True, constrained_layout=True)

    for ax, identifier_kind in zip(axes, kinds):
        result = results[identifier_kind]
        rows = selected_families(result, identifier_kind)
        shares = np.array([100 * row["slope_share"] for row in rows])
        positions = np.arange(len(rows))
        colors = [GREEN if share > 0 else PURPLE for share in shares]
        ax.barh(positions, shares, height=0.62, color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color=MUTED, linewidth=0.6)
        ax.set_yticks(
            positions,
            [f"{row['family']}  ({_presence_label(row['present'], versions).replace('--', EN_DASH).replace('1.0', '1')})"
             for row in rows],
            fontsize=7,
        )
        ax.set_ylim(-0.6, max(counts) - 0.4)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        for position, share in zip(positions, shares):
            ax.annotate(
                f"{share:+.0f}%".replace("-", "\u2212"), xy=(share, position),
                xytext=(3 if share > 0 else -3, 0), textcoords="offset points",
                ha="left" if share > 0 else "right", va="center", fontsize=6.5, color=INK,
            )
        ax.set_title(KIND_LABELS[identifier_kind], loc="left")
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)

    limit = max(abs(100 * row["slope_share"]) for kind in kinds for row in selected_families(results[kind], kind))
    axes[-1].set_xlim(-limit * 1.45, limit * 1.45)
    for ax in axes:
        ax.set_xlabel("Share of the slope due to the family (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GREEN), plt.Rectangle((0, 0), 1, 1, color=PURPLE)]
    fig.legend(handles, ["Strengthens the trend", "Weakens the trend"], loc="outside upper center",
               ncols=2, frameon=False, fontsize=7, handlelength=1.2)
    save_figure(fig, path)
    plt.close(fig)


# Project prefixes cited in the text: (release, family, prefixes).
PROJECT_PREFIXES = (
    ("1.0", "OB", ("OB", "OR")),
    ("14", "Calypso", ("Cly",)),
    ("14", "Metacello", ("Metacello",)),
    ("14", "Athens", ("Athens",)),
)


def print_largest_drops(corpus: Corpus, count: int = 3) -> None:
    """The largest release-to-release decreases of class-name cohesion, and
    what is left of each without the families that enter in that release."""
    versions = list(corpus)
    releases = [name_families(records, "class") for records in corpus.values()]
    encoded = [encode_vocabulary(list(release)) for release in releases]
    series = cohesion_series(releases, encoded, set())
    presence = family_presence(versions, releases)
    steps = np.diff(series)
    print(
        f"Class names, all {len(steps)} consecutive-release changes: "
        + ", ".join(f"{before} -> {after} {step:+.4f}" for before, after, step in zip(versions, versions[1:], steps))
    )
    for index in np.argsort(steps)[:count]:
        version = versions[index + 1]
        entering = {family for family, present in presence.items() if present[0] == version}
        without = cohesion_series(releases, encoded, entering)
        step_without = without[index + 1] - without[index]
        print(
            f"Class names, {versions[index]} -> {version}: change {steps[index]:+.4f}; without the "
            f"{len(entering)} families entering in {version}: {step_without:+.4f} "
            f"({1 - step_without / steps[index]:.0%} of the decrease removed)"
        )


def print_project_prefixes(corpus: Corpus) -> None:
    for version, family, prefixes in PROJECT_PREFIXES:
        names = [
            record.class_name for record in corpus[version]
            if record.kind == "class" and package_family(record.package) == family
        ]
        matching = sum(name.startswith(prefixes) for name in names)
        print(f"{family} in Pharo {version}: {matching} of {len(names)} classes start with {' or '.join(prefixes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    corpus = read_corpus(args.corpus_dir)
    versions = list(corpus)
    results = {kind: contributions(corpus, kind) for kind in IDENTIFIER_KINDS}

    for identifier_kind, result in results.items():
        print(
            f"{KIND_LABELS[identifier_kind]}: change {result['change']:+.4f}, "
            f"slope {result['slope']:+.4f}"
        )
        for name, group in result["groups"].items():
            print(
                f"  without the {group['families']} families that {name}: "
                f"change {group['change_without']:+.4f}, slope {group['slope_without']:+.4f}"
            )
        for row in result["families"][:10]:
            print(
                f"  {row['family']:<24} {_presence_label(row['present'], versions):<8} "
                f"change {row['change']:+.4f}  share of slope {row['slope_share']:+.0%}"
            )

    for identifier_kind, result in results.items():
        print(f"{KIND_LABELS[identifier_kind]}: families shown in the figure")
        for row in selected_families(result, identifier_kind):
            print(
                f"  {row['family']:<24} {_presence_label(row['present'], versions):<8} "
                f"n {row['first_size']:,} -> {row['last_size']:,}  change {row['change']:+.4f}  "
                f"share of slope {row['slope_share']:+.0%}"
            )

    print_largest_drops(corpus)
    print_project_prefixes(corpus)

    output = args.results_dir / "family_contributions.tex"
    write_contribution_figure(output, versions, results)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
