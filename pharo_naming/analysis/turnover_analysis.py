"""How many names each release keeps, adds, relocates, and removes.

A table also lists the most frequent prefixes of the first and last release
(:func:`global_analysis.frequent_prefixes`).

    python -m pharo_naming.analysis.turnover_analysis

A name is *kept* by a release when the previous release already had it,
*added* when it did not, and *removed* when the previous release had it and
this one does not.  A kept name is *relocated* when none of the package
families that declare it in this release declared it in the previous one:
the name survives, but in another part of the system.  Names are compared as
strings, so a renamed class counts as one removal and one addition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..corpus import IdentifierRecord, read_corpus
from ..paths import CORPUS_DIR, RESULTS_DIR
from ..plotting import BLUE, COLUMN_WIDTH, GRID, INK, MUTED, ORANGE, SEQUENTIAL_RAMP, pyplot, save_figure
from ..reporting import HEADER, signed, write_text
from ..vocabulary import IDENTIFIER_KINDS, PRIMARY_WEIGHTING, identifiers
from .family_contribution_analysis import NameFamilies, name_families
from .global_analysis import PREFIX_TABLE_LENGTHS, frequent_prefixes


KIND_LABELS = {"method": "Selectors", "class": "Class names"}
Corpus = dict[str, list[IdentifierRecord]]


def transition(before: NameFamilies, after: NameFamilies) -> dict:
    kept = before.keys() & after.keys()
    return {
        "kept": len(kept),
        "added": len(after.keys() - before.keys()),
        "removed": len(before.keys() - after.keys()),
        "relocated": sum(1 for name in kept if not before[name] & after[name]),
    }


def turnover(corpus: Corpus, identifier_kind: str) -> dict:
    """Release-to-release transitions, and the balance of the whole history."""
    versions = list(corpus)
    releases = [name_families(records, identifier_kind) for records in corpus.values()]
    steps = [
        {"from": versions[i], "to": versions[i + 1], **transition(releases[i], releases[i + 1])}
        for i in range(len(releases) - 1)
    ]

    first, last = releases[0], releases[-1]
    ever = set().union(*releases)
    return {
        "steps": steps,
        "first": len(first),
        "last": len(last),
        "ever": len(ever),
        "overall": transition(first, last),
        # Names that appeared after the first release and were gone by the last.
        "transient": len(ever - first.keys() - last.keys()),
    }


# Segments of the turnover figure: (key, legend label, color, text color).
SEGMENTS = (
    ("same", "kept, same family", BLUE, "white"),
    ("relocated", "kept, other family", SEQUENTIAL_RAMP[1], INK),
    ("removed", "removed", "#b4b2ad", INK),
    ("added", "added", ORANGE, INK),
)


# Vertical position of the group brackets, in axes fraction (below the release labels).
BRACKET_Y = -0.1


def write_turnover_figure(path: Path, versions: list[str], results: dict[str, dict]) -> None:
    """For each vocabulary and each of the first and last releases, one column
    per part of the vocabulary: names kept (in the same family or another one),
    and names removed (first release) or added (last release).

    Both vocabularies share one logarithmic axis in numbers of names, so that
    the small class-name counts stay readable next to the selector counts.  The
    columns are not stacked, since stacked lengths are meaningless on a log scale."""
    plt = pyplot()
    first, last = versions[0], versions[-1]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.6), constrained_layout=True)
    styles = {key: (color, text_color) for key, _, color, text_color in SEGMENTS}
    columns = {0: ("same", "relocated", "removed"), 1: ("same", "relocated", "added")}
    width, step = 0.8, 1.0  # column width and distance between two columns
    floor = 100  # start of the logarithmic axis
    labels, ticks, position = [], [], 0.0
    below = ax.get_xaxis_transform()  # x in data, y in axes fraction
    for kind in ("method", "class"):
        overall = results[kind]["overall"]
        counts = {
            "same": overall["kept"] - overall["relocated"],
            "relocated": overall["relocated"],
            "removed": overall["removed"],
            "added": overall["added"],
        }
        group_left = position
        for offset, keys in columns.items():
            release_left = position
            for key in keys:
                color, text_color = styles[key]
                ax.bar(position, counts[key] - floor, bottom=floor, width=width, color=color,
                       edgecolor="white", linewidth=0.6)
                ax.text(position, counts[key] / 1.12, f"{counts[key]:,}", ha="center", va="top",
                        rotation=90, fontsize=5.5, color=text_color)
                position += step
            ticks.append((release_left + position - step) / 2)
            labels.append(f"Pharo {first if offset == 0 else last}")
            position += 0.5 * step
        # Group label under a bracket spanning its columns.
        left, right = group_left - width / 2, position - 1.5 * step + width / 2
        ax.plot([left, left, right, right], [BRACKET_Y + 0.025, BRACKET_Y, BRACKET_Y, BRACKET_Y + 0.025],
                transform=below, color=INK, linewidth=0.7, clip_on=False)
        ax.text((left + right) / 2, BRACKET_Y - 0.03, KIND_LABELS[kind], transform=below,
                ha="center", va="top", fontsize=7, color=INK)
        position += 0.7 * step
    ax.set_yscale("log")
    ax.set_ylim(floor, 60_000)
    ax.set_yticks([100, 1_000, 10_000], ["100", "1,000", "10,000"])
    ax.set_xticks(ticks, labels, fontsize=6.5)
    ax.set_xlim(-0.6, position - 1.2 * step + 0.6)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=6)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Number of names (log scale)", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, _, color, _ in SEGMENTS]
    fig.legend(handles, [label for _, label, _, _ in SEGMENTS], loc="outside upper center",
               ncols=4, frameon=False, fontsize=6.5, handlelength=1.2, columnspacing=1.0)
    save_figure(fig, path)
    plt.close(fig)


# Most frequent prefixes of each release that enter the table.
PREFIX_TABLE_TOP = 5


def prefix_shares(names: list[str], length: int) -> dict[str, float]:
    """Share of the vocabulary (in percent) of every prefix of ``length`` characters."""
    return dict(frequent_prefixes(names, length, None))


def prefix_changes(corpus: Corpus, kind: str) -> list[tuple[str, float, float]]:
    """The prefixes among the most frequent of the first or last release, with
    their share in both, from the largest gain to the largest loss."""
    versions = list(corpus)
    length = PREFIX_TABLE_LENGTHS[kind]
    first, last = (
        prefix_shares(identifiers(corpus[version], kind, PRIMARY_WEIGHTING), length)
        for version in (versions[0], versions[-1])
    )
    top = {
        prefix
        for shares in (first, last)
        for prefix in sorted(shares, key=lambda prefix: -shares[prefix])[:PREFIX_TABLE_TOP]
    }
    rows = [(prefix, first.get(prefix, 0.0), last.get(prefix, 0.0)) for prefix in top]
    return sorted(rows, key=lambda row: (row[1] - row[2], row[0]))


def write_prefix_table(path: Path, corpus: Corpus) -> None:
    """The most frequent prefixes of the first and last release, with their
    share in both and the change, green for a gain and red for a loss."""
    versions = list(corpus)
    first, last = versions[0], versions[-1]
    columns = [prefix_changes(corpus, kind) for kind in IDENTIFIER_KINDS]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Most frequent prefixes in \textsc{{Pharo}} {first} and {last} "
        rf"({PREFIX_TABLE_LENGTHS['method']} characters for method names, {PREFIX_TABLE_LENGTHS['class']} "
        r"for class names): share of the vocabulary (\%) and change.}",
        r"\label{tab:prefixes}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        # A vertical rule separates the two vocabularies.  No @{} at the edges:
        # colortbl would paint the gray header past the horizontal rules.
        r"\begin{tabular}{lrrr|lrrr}",
        r"\toprule",
        HEADER + r"\multicolumn{4}{c|}{Method names} & \multicolumn{4}{c}{Class names}\\",
        HEADER + rf"Prefix & {first} & {last} & Change & Prefix & {first} & {last} & Change\\",
        r"\midrule",
    ]
    for index in range(max(len(column) for column in columns)):
        cells = []
        for column in columns:
            if index < len(column):
                prefix, before, after = column[index]
                cells.append(rf"\texttt{{{prefix}}} & {before:.1f} & {after:.1f} & {signed(after - before, 1)}")
            else:
                cells.append(" & & & ")
        lines.append(" & ".join(cells) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write_text(path, lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    corpus = read_corpus(args.corpus_dir)
    results = {kind: turnover(corpus, kind) for kind in IDENTIFIER_KINDS}

    for identifier_kind, result in results.items():
        overall = result["overall"]
        print(
            f"{KIND_LABELS[identifier_kind]}: {result['first']:,} -> {result['last']:,}; "
            f"kept {overall['kept']:,} (relocated {overall['relocated']:,}), "
            f"removed {overall['removed']:,}, added {overall['added']:,}, "
            f"transient {result['transient']:,}, ever {result['ever']:,}"
        )
        first = result["first"]
        print(
            f"  relative to Pharo {next(iter(corpus))}: kept {overall['kept'] / first:.1%}, "
            f"kept in none of the original families {overall['relocated'] / first:.1%}, "
            f"added {overall['added'] / first:.1%} "
            f"({overall['added'] / overall['kept']:.1f} added per kept name), "
            f"transient {result['transient'] / result['last']:.1%} of the Pharo "
            f"{list(corpus)[-1]} vocabulary"
        )
        for step in result["steps"]:
            size = step["kept"] + step["removed"]
            print(
                f"  {step['from']:>4} -> {step['to']:<3} added {step['added'] / size:6.1%}  "
                f"removed {step['removed'] / size:6.1%}  relocated {step['relocated']:,}"
            )

    figure = args.results_dir / "vocabulary_turnover.tex"
    write_turnover_figure(figure, list(corpus), results)
    table = args.results_dir / "frequent_prefixes.tex"
    write_prefix_table(table, corpus)
    print(f"Saved {figure} and {table}")


if __name__ == "__main__":
    main()
