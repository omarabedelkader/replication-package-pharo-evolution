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


def write_turnover_figure(path: Path, versions: list[str], results: dict[str, dict]) -> None:
    """For each vocabulary, one bar per release: the first release split into
    names kept and removed, the last one into names kept and added.

    Lengths are relative to the size of the first release, so both vocabularies
    share one axis and their turnover can be compared directly; the labels
    give the shares and the total counts.  A band links the names kept by both releases."""
    plt = pyplot()
    first, last = versions[0], versions[-1]
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 2.25), sharex=True, constrained_layout=True)
    styles = {key: (color, text_color) for key, _, color, text_color in SEGMENTS}
    bars = {0: ("same", "relocated", "removed"), 1: ("same", "relocated", "added")}
    height = 0.56
    for ax, kind in zip(axes, ("method", "class")):
        result = results[kind]
        overall = result["overall"]
        counts = {
            "same": overall["kept"] - overall["relocated"],
            "relocated": overall["relocated"],
            "removed": overall["removed"],
            "added": overall["added"],
        }
        scale = 100 / result["first"]
        kept = overall["kept"] * scale
        ax.fill_between([0, kept], 0 + height / 2, 1 - height / 2, color=GRID, alpha=0.6, linewidth=0)
        for position, keys in bars.items():
            left = 0.0
            for key in keys:
                width = counts[key] * scale
                color, text_color = styles[key]
                ax.barh(position, width, left=left, height=height, color=color,
                        edgecolor="white", linewidth=0.8)
                ax.text(left + width / 2, position, f"{width:.0f}", ha="center", va="center",
                        fontsize=6, color=text_color)
                left += width
            total = result["first"] if position == 0 else result["last"]
            ax.text(left + 3, position, f"{total:,} names", ha="left", va="center", fontsize=6, color=INK)
        ax.axvline(100, color=MUTED, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
        ax.set_yticks([0, 1], [f"Pharo {first}", f"Pharo {last}"], fontsize=6.5)
        ax.set_ylim(1.5, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(KIND_LABELS[kind], loc="left", fontsize=7, pad=2)
    axes[-1].set_xlim(0, 300)
    axes[-1].set_xticks([0, 50, 100, 150, 200, 250], ["0", "50", "100", "150", "200", "250"])
    axes[-1].set_xlabel(f"Names, in % of the Pharo {first} vocabulary", fontsize=7)
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
