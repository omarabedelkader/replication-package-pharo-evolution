"""Conceptual cohesion of the package families, and its relation to prefix cohesion.

    python -m pharo_naming.analysis.conceptual_cohesion_analysis

For every release, each class becomes a document made of the words of its
name and of the selectors of the methods that its own family defines on it
(extension methods belong to the family that defines them).  The conceptual
cohesion of a family (CCP, Santos et al. 2014) is the mean cosine similarity
of the TF-IDF vectors of its pairs of classes.  Three controls are reported:

* the *baseline*, the mean similarity of all pairs of classes of the release,
  which is the expected CCP of a random group of classes of any size; the
  *adjusted* CCP is the difference to it, and a family is above random when
  fewer than 5% of 500 size-matched random groups reach its CCP;
* *without prefix words*: the words that start at least half of the class
  names of a family (project prefixes such as ``Athens`` or ``Cly``) are
  removed from its documents before weighting, so that a shared prefix does
  not count as a shared concept;
* *without test classes*: classes named by the SUnit convention are left out.

The CCP of a family is then compared with the prefix cohesion of its class
names and of its selectors, over the families of each release, with Spearman
correlations; the partial correlation removes the effect of family size, with
which both measures decrease.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from ..concepts import mean_pairwise_similarity, prefix_words, random_group_similarities, terms, tfidf_matrix
from ..corpus import IdentifierRecord, read_corpus
from ..paths import CORPUS_DIR, RESULTS_DIR
from ..plotting import BLUE, COLUMN_WIDTH, GRID, INK, MUTED, ORANGE, SEQUENTIAL_RAMP, TEXT_WIDTH, FIGURE_HEIGHT, pyplot, save_figure
from ..vocabulary import PRIMARY_WEIGHTING, identifiers, package_family
from .robustness_analysis import REPETITIONS, SEED, TEST_CONVENTION, encode_vocabulary, pooled_cohesion


# A family is analyzed in a release when it has at least this many classes
# with a non-empty document; below that, a mean over pairs is too unstable.
MIN_CLASSES = 10
SIGNIFICANCE = 0.05
# Families ranked over time: analyzed in at least this many releases.
MIN_RANKED_RELEASES = 12
LABELLED_FAMILIES = {
    "Athens", "Metacello", "Morphic", "Kernel", "System", "Zinc", "Calypso",
    "Fuel", "SystemCommands", "Refactoring", "Hermes", "Epicea", "Clap",
}

# Label positions (points) for the families whose names would collide.
LABEL_OFFSETS: dict[str, tuple[float, float]] = {
    "Hermes": (-30, 1), "Epicea": (-26, -2), "Kernel": (-25, -1), "Calypso": (4, -5),
    "Refactoring": (4, -6), "Clap": (2, 3), "Morphic": (4, -3),
}

Corpus = dict[str, list[IdentifierRecord]]


# -- documents ---------------------------------------------------------------

def class_documents(records: list[IdentifierRecord]) -> dict[str, tuple[str, Counter[str]]]:
    """Family and bag of words of every class of a release."""
    family_of = {
        record.class_name: package_family(record.package)
        for record in records if record.kind == "class"
    }
    documents = {name: Counter(terms(name)) for name in family_of}
    for record in records:
        if (
            record.kind == "method"
            and family_of.get(record.class_name) == package_family(record.package)
        ):
            documents[record.class_name].update(terms(record.selector))
    return {name: (family_of[name], documents[name]) for name in sorted(documents)}


def without_words(
    documents: dict[str, tuple[str, Counter[str]]],
    removed: dict[str, set[str]],
) -> dict[str, tuple[str, Counter[str]]]:
    """The documents with the words ``removed[family]`` taken out of each family."""
    return {
        name: (family, Counter({
            word: count for word, count in document.items()
            if word not in removed.get(family, ())
        }))
        for name, (family, document) in documents.items()
    }


def family_ccp(documents: dict[str, tuple[str, Counter[str]]]) -> tuple[dict[str, float], float, dict]:
    """CCP of every family large enough, the release baseline, and the data
    needed for the random groups (vectors and row indices of each family)."""
    names = [name for name, (_, document) in documents.items() if document]
    vectors = tfidf_matrix([documents[name][1] for name in names])
    rows: dict[str, list[int]] = defaultdict(list)
    for row, name in enumerate(names):
        rows[documents[name][0]].append(row)
    rows = {family: indices for family, indices in rows.items() if len(indices) >= MIN_CLASSES}
    ccp = {family: mean_pairwise_similarity(vectors[indices]) for family, indices in rows.items()}
    return ccp, mean_pairwise_similarity(vectors), {"vectors": vectors, "rows": rows}


# -- one release -------------------------------------------------------------

def analyze_release(
    version: str,
    records: list[IdentifierRecord],
    generator: np.random.Generator,
    repetitions: int = REPETITIONS,
) -> tuple[list[dict], dict]:
    """One row per analyzed family of the release, and the release summary."""
    documents = class_documents(records)
    ccp, baseline, data = family_ccp(documents)

    members: dict[str, list[str]] = defaultdict(list)
    for name, (family, _) in documents.items():
        members[family].append(name)
    prefixes = {family: prefix_words(members[family]) for family in ccp}
    ccp_unprefixed, _, _ = family_ccp(without_words(documents, prefixes))

    is_test = TEST_CONVENTION["class"]
    untested = {name: entry for name, entry in documents.items() if not is_test(name.lower())}
    ccp_untested, _, _ = family_ccp(untested)

    by_family: dict[str, list[IdentifierRecord]] = defaultdict(list)
    for record in records:
        by_family[package_family(record.package)].append(record)

    rows = []
    for family in sorted(ccp):
        size = len(data["rows"][family])
        random = random_group_similarities(data["vectors"], size, repetitions, generator)
        class_names = identifiers(by_family[family], "class", PRIMARY_WEIGHTING)
        selectors = identifiers(by_family[family], "method", PRIMARY_WEIGHTING)
        rows.append({
            "version": version,
            "family": family,
            "classes": size,
            "selectors": len(selectors),
            "class_cohesion": pooled_cohesion(encode_vocabulary(class_names)),
            "selector_cohesion": pooled_cohesion(encode_vocabulary(selectors)) if selectors else np.nan,
            "ccp": ccp[family],
            "adjusted": ccp[family] - baseline,
            "p_random": (1 + int(np.sum(random >= ccp[family]))) / (1 + repetitions),
            "ccp_unprefixed": ccp_unprefixed.get(family, np.nan),
            "prefix_words": sorted(prefixes[family]),
            "ccp_untested": ccp_untested.get(family, np.nan),
        })

    summary = {
        "version": version,
        "baseline": baseline,
        "families": len(rows),
        "above_random": sum(row["p_random"] < SIGNIFICANCE for row in rows),
        **release_correlations(rows),
    }
    return rows, summary


def partial_spearman(x, y, z) -> float:
    """Spearman correlation of ``x`` and ``y`` with the ranks of ``z`` partialled out."""
    keep = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    rx, ry, rz = (rankdata(values[keep]) for values in (x, y, z))
    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]
    return float((r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2)))


def release_correlations(rows: list[dict]) -> dict:
    def column(key):
        return np.array([row[key] for row in rows], dtype=float)

    ccp, size = column("ccp"), column("classes")
    class_pc, selector_pc = column("class_cohesion"), column("selector_cohesion")
    unprefixed, untested = column("ccp_unprefixed"), column("ccp_untested")

    def rho(x, y):
        keep = ~(np.isnan(x) | np.isnan(y))
        return float(spearmanr(x[keep], y[keep]).statistic)

    return {
        "class_pc_ccp": rho(class_pc, ccp),
        "class_pc_ccp_partial": partial_spearman(class_pc, ccp, size),
        "class_pc_ccp_unprefixed": rho(class_pc, unprefixed),
        "class_pc_ccp_untested": rho(class_pc, untested),
        "selector_pc_ccp": rho(selector_pc, ccp),
        "selector_pc_ccp_partial": partial_spearman(selector_pc, ccp, size),
        "size_ccp": rho(size, ccp),
        "size_class_pc": rho(size, class_pc),
    }


def analyze(corpus: Corpus, repetitions: int = REPETITIONS, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    generator = np.random.default_rng(seed)
    rows, summaries = [], []
    for version, records in corpus.items():
        release_rows, summary = analyze_release(version, records, generator, repetitions)
        rows.extend(release_rows)
        summaries.append(summary)
    return rows, summaries


# -- rankings over time --------------------------------------------------------

def percentile_ranks(rows: list[dict], key: str) -> dict[tuple[str, str], float]:
    """Rank of each family within its release, from 0 (lowest) to 1 (highest)."""
    by_version: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_version[row["version"]].append(row)
    ranks = {}
    for version, release_rows in by_version.items():
        values = rankdata([row[key] for row in release_rows])
        for row, value in zip(release_rows, values):
            ranks[(version, row["family"])] = (value - 1) / max(len(release_rows) - 1, 1)
    return ranks


def rank_stability(rows: list[dict], versions: list[str], key: str) -> list[float]:
    """Spearman correlation of ``key`` between consecutive releases, over the
    families analyzed in both."""
    values = {(row["version"], row["family"]): row[key] for row in rows}
    stability = []
    for before, after in zip(versions, versions[1:]):
        common = sorted(
            family for version, family in values
            if version == before and (after, family) in values
        )
        stability.append(float(spearmanr(
            [values[(before, family)] for family in common],
            [values[(after, family)] for family in common],
        ).statistic))
    return stability


# Type of a family, by its position with respect to the medians of the release:
# (prefix cohesion of class names at least the median, CCP at least the median).
TYPES = {
    (True, True): "aligned",
    (True, False): "prefix only",
    (False, True): "concept only",
    (False, False): "diffuse",
}


def quadrant(row: dict, median_pc: float, median_ccp: float) -> str:
    return TYPES[(row["class_cohesion"] >= median_pc, row["ccp"] >= median_ccp)]


def family_types(rows: list[dict]) -> dict[tuple[str, str], str]:
    """Type of every analyzed family in every release."""
    by_version: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_version[row["version"]].append(row)
    types = {}
    for version, release in by_version.items():
        median_pc = float(np.median([row["class_cohesion"] for row in release]))
        median_ccp = float(np.median([row["ccp"] for row in release]))
        for row in release:
            types[(version, row["family"])] = quadrant(row, median_pc, median_ccp)
    return types


def ranked_families(rows: list[dict], versions: list[str]) -> list[str]:
    """Families analyzed in at least ``MIN_RANKED_RELEASES`` releases, by mean rank."""
    ranks = percentile_ranks(rows, "ccp")
    observed: dict[str, list[float]] = defaultdict(list)
    for (version, family), rank in ranks.items():
        observed[family].append(rank)
    eligible = [family for family, values in observed.items() if len(values) >= MIN_RANKED_RELEASES]
    return sorted(eligible, key=lambda family: -np.mean(observed[family]))


# -- reports -----------------------------------------------------------------

# Rows of the correlation figure: (label, key in the release summaries).
FAMILY_CORRELATIONS = (
    ("Class PC \u2013 CCP", "class_pc_ccp"),
    ("Class PC \u2013 CCP, size out", "class_pc_ccp_partial"),
    ("Class PC \u2013 CCP w/o prefix words", "class_pc_ccp_unprefixed"),
    ("Class PC \u2013 CCP w/o test classes", "class_pc_ccp_untested"),
    ("Selector PC \u2013 CCP", "selector_pc_ccp"),
    ("Selector PC \u2013 CCP, size out", "selector_pc_ccp_partial"),
    ("Size \u2013 CCP", "size_ccp"),
    ("Size \u2013 class PC", "size_class_pc"),
)
STABILITY = (("CCP, next snapshot", "ccp"), ("Class PC, next snapshot", "class_cohesion"))
SYSTEM_CORRELATIONS = (
    ("Class PC \u2013 system CCP", "class_pc_ccp"),
    ("Selector PC \u2013 system CCP", "selector_pc_ccp"),
    ("Families \u2013 system CCP", "families_ccp"),
    ("Families \u2013 class PC", "families_class_pc"),
    ("Order \u2013 system CCP", "order_ccp"),
)
SYSTEM_VIEWS = (
    ("whole", "Whole system", {"color": INK, "marker": "o"}),
    ("persistent", "Persistent families", {"color": BLUE, "marker": "s", "markerfacecolor": "white"}),
    ("changes", "Changes", {"color": ORANGE, "marker": "^"}),
)


def correlation_ranges(summaries: list[dict], stability: dict[str, list[float]]) -> list[tuple[str, float, float, float]]:
    """(label, median, minimum, maximum) of every family-level correlation over the releases."""
    ranges = []
    for label, key in FAMILY_CORRELATIONS:
        values = [summary[key] for summary in summaries]
        ranges.append((label, float(np.median(values)), min(values), max(values)))
    for label, key in STABILITY:
        values = stability[key]
        ranges.append((label, float(np.median(values)), min(values), max(values)))
    return ranges


# Panels of the correlation figure, side by side and in the order of Figure 1
# (selectors left, class names right):
# [(group title, color key, [(row label, key)])] per panel.
# Colors follow Figure 1: orange for class names, blue for selectors.
CORRELATION_PANELS = (
    (
        ("Method selectors", "method", (
            ("PC vs. CCP", "selector_pc_ccp"),
            ("PC vs. CCP, size-adjusted", "selector_pc_ccp_partial"),
        )),
        ("Family size", "size", (
            ("Size vs. CCP", "size_ccp"),
            ("Size vs. class-name PC", "size_class_pc"),
        )),
    ),
    (
        ("Class names", "class", (
            ("PC vs. CCP", "class_pc_ccp"),
            ("PC vs. CCP, size-adjusted", "class_pc_ccp_partial"),
            ("PC vs. CCP, no prefix words", "class_pc_ccp_unprefixed"),
            ("PC vs. CCP, no test classes", "class_pc_ccp_untested"),
        )),
    ),
)


def write_correlation_figure(
    path: Path,
    summaries: list[dict],
    stability: dict[str, list[float]],
    system: dict[str, dict[str, float]],
) -> None:
    """Dot plot of the Spearman correlations across the families of each release
    (median and range over the releases), in two panels at full text width.
    The whole-system correlations and the rank stability are printed in
    run.log (see :mod:`system_ccp_analysis`)."""
    plt = pyplot()
    colors = {"class": ORANGE, "method": BLUE, "size": INK}
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, FIGURE_HEIGHT), constrained_layout=True)

    def layout(groups):
        # Every group starts with a header row, so both panels align.
        position, rows, headers = 0.0, [], []
        for title, kind, entries in groups:
            headers.append((position, title, colors[kind]))
            position += 1
            for label, key in entries:
                rows.append((position, label, key, kind))
                position += 1
        return position, rows, headers

    layouts = [layout(groups) for groups in CORRELATION_PANELS]
    slots = max(size for size, _, _ in layouts)
    for ax, groups, (_, rows, headers) in zip(axes, CORRELATION_PANELS, layouts):
        for position, label, key, kind in rows:
            values = [summary[key] for summary in summaries]
            median, low, high = float(np.median(values)), min(values), max(values)
            ax.plot([low, high], [position, position], color=colors[kind], alpha=0.35,
                    linewidth=3, solid_capstyle="round")
            ax.plot([median], [position], color=colors[kind], marker="o", markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.6, linestyle="none")
            ax.annotate(f"{median:+.2f}".replace("-", "\u2212"), xy=(high, position),
                        xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=6.5, color=INK)
        ax.set_yticks([row[0] for row in rows], [row[1] for row in rows], fontsize=7)
        for y, title, color in headers:
            ax.annotate(title, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(-4, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=8, fontweight="bold", color=color)
        ax.set_ylim(slots - 0.5, -0.6)
        ax.axvline(0, color=MUTED, linewidth=0.7)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
        ax.set_xlim(-1, 1)
        ax.set_xlabel(r"Spearman correlation $\rho$ across families")
    save_figure(fig, path)
    plt.close(fig)


def write_scatter(path: Path, rows: list[dict], version: str, baseline: float) -> None:
    """Prefix cohesion of the class names against CCP, one point per family.

    Every family is above the random baseline, so the points share one color;
    the families named in the text are emphasized and labelled.
    """
    plt = pyplot()
    release = [row for row in rows if row["version"] == version]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.7))

    pc = np.array([row["class_cohesion"] for row in release])
    ccp = np.array([row["ccp"] for row in release])
    size = np.array([row["classes"] for row in release])
    named = np.array([row["family"] in LABELLED_FAMILIES for row in release])
    area = 8 + 70 * np.sqrt(size / size.max())

    ax.axvline(np.median(pc), color=GRID, linewidth=0.8, zorder=0)
    ax.axhline(np.median(ccp), color=GRID, linewidth=0.8, zorder=0)
    ax.axhline(baseline, color=MUTED, linewidth=0.6, linestyle=":", zorder=0)
    ax.annotate("mean of random groups of classes", xy=(0.5, baseline), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize=6,
                color=MUTED, xycoords=("axes fraction", "data"))

    ax.scatter(pc[~named], ccp[~named], s=area[~named], color=GRID,
               edgecolors="white", linewidths=0.5, zorder=2)
    ax.scatter(pc[named], ccp[named], s=area[named], color=BLUE,
               edgecolors="white", linewidths=0.6, zorder=3)
    for row in release:
        if row["family"] in LABELLED_FAMILIES:
            ax.annotate(row["family"], xy=(row["class_cohesion"], row["ccp"]),
                        xytext=LABEL_OFFSETS.get(row["family"], (4, 1)),
                        textcoords="offset points", fontsize=6, color=INK, zorder=4)

    corner = dict(fontsize=6.5, color=MUTED, transform=ax.transAxes, style="italic")
    ax.text(0.99, 0.97, TYPES[(True, True)], ha="right", va="top", **corner)
    ax.text(0.99, 0.02, TYPES[(True, False)], ha="right", va="bottom", **corner)
    ax.text(0.01, 0.97, TYPES[(False, True)], ha="left", va="top", **corner)
    ax.text(0.01, 0.02, TYPES[(False, False)], ha="left", va="bottom", **corner)

    ax.set_yscale("log")
    ax.set_ylim(baseline * 0.75, ccp.max() * 1.6)
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5], ["0.02", "0.05", "0.1", "0.2", "0.5"])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(f"Prefix cohesion of the class names (Pharo {version})")
    ax.set_ylabel("Conceptual cohesion, CCP (log scale)")
    ax.grid(True, axis="y", which="major")
    ax.set_axisbelow(True)
    save_figure(fig, path)
    plt.close(fig)


def write_rank_heatmap(path: Path, rows: list[dict], versions: list[str]) -> None:
    """Rank of the long-lived families by CCP and by class-name prefix cohesion."""
    plt = pyplot()
    from matplotlib.colors import LinearSegmentedColormap

    families = ranked_families(rows, versions)
    colormap = LinearSegmentedColormap.from_list("rank", SEQUENTIAL_RAMP)
    colormap.set_bad("white")
    height = 0.072 * len(families) + 0.75
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, height), constrained_layout=True)
    image = None

    for ax, (key, title) in zip(axes, (
        ("ccp", "Conceptual cohesion (CCP)"),
        ("class_cohesion", "Prefix cohesion of class names"),
    )):
        ranks = percentile_ranks(rows, key)
        matrix = np.array([
            [ranks.get((version, family), np.nan) for version in versions]
            for family in families
        ])
        # Vector cells (an imshow image would be written as a separate PNG).
        image = ax.pcolormesh(
            np.arange(-0.5, len(versions)), np.arange(-0.5, len(families)),
            np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=colormap,
        )
        ax.set_ylim(len(families) - 0.5, -0.5)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Pharo version")
        ax.set_xticks(range(len(versions)), versions, fontsize=6, rotation=90)
        ax.set_yticks(range(len(families)), families, fontsize=6)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(versions)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(families)), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)

    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.01)
    bar.set_label("Rank in the snapshot (0 lowest, 1 highest)")
    bar.outline.set_visible(False)
    bar.solids.set_rasterized(False)
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
    rows, summaries = analyze(corpus, args.repetitions, args.seed)
    stability = {key: rank_stability(rows, versions, key) for key in ("ccp", "class_cohesion")}

    for summary in summaries:
        print(
            f"Pharo {summary['version']:>4}: {summary['families']:3} families, "
            f"{summary['above_random']:3} above random, baseline {summary['baseline']:.4f}, "
            f"rho(PC, CCP) {summary['class_pc_ccp']:+.2f} "
            f"(partial {summary['class_pc_ccp_partial']:+.2f}, "
            f"no prefix {summary['class_pc_ccp_unprefixed']:+.2f}, "
            f"no tests {summary['class_pc_ccp_untested']:+.2f}), "
            f"rho(selector PC, CCP) {summary['selector_pc_ccp']:+.2f} "
            f"(partial {summary['selector_pc_ccp_partial']:+.2f}), "
            f"rho(size, CCP) {summary['size_ccp']:+.2f}"
        )
    for key, values in stability.items():
        print(f"Rank stability of {key}: median {np.median(values):+.2f}, "
              f"range [{min(values):+.2f}, {max(values):+.2f}]")

    last = versions[-1]
    release = [row for row in rows if row["version"] == last]
    median_pc = float(np.median([row["class_cohesion"] for row in release]))
    median_ccp = float(np.median([row["ccp"] for row in release]))
    types = Counter(quadrant(row, median_pc, median_ccp) for row in release)
    print(f"Pharo {last} family types: {dict(types)}")
    pc_rank = {
        row["family"]: position
        for position, row in enumerate(sorted(release, key=lambda row: -row["class_cohesion"]), start=1)
    }
    for ccp_rank, row in enumerate(sorted(release, key=lambda row: -row["ccp"]), start=1):
        print(
            f"  {ccp_rank:3}/{len(release)} {row['family']:<26} n={row['classes']:4} "
            f"PC={row['class_cohesion']:.3f} (rank {pc_rank[row['family']]}) "
            f"CCP={row['ccp']:.3f} CCP-={row['ccp_unprefixed']:.3f} p={row['p_random']:.3f} "
            f"{quadrant(row, median_pc, median_ccp):<24} {row['prefix_words']}"
        )
    history: dict[str, list[str]] = defaultdict(list)
    for (_, family), kind in family_types(rows).items():
        history[family].append(kind)
    usual = [max(Counter(kinds).values()) / len(kinds) for kinds in history.values()]
    print(
        f"Type stability over {len(usual)} families: median share of snapshots with the "
        f"usual type {np.median(usual):.0%}; {np.mean(np.array(usual) >= 0.75):.0%} of the "
        "families keep their usual type in at least 75% of their snapshots"
    )
    families = ranked_families(rows, versions)
    print(f"{len(families)} families ranked in at least {MIN_RANKED_RELEASES} releases: {families}")

    for label, median, low, high in correlation_ranges(summaries, stability):
        print(f"{label.strip():<32} median {median:+.2f}  range [{low:+.2f}, {high:+.2f}]")

    # The system level reuses this module, hence the import here.
    from .system_ccp_analysis import report_system_level
    system = report_system_level(corpus)

    outputs = {
        "ccp_correlations.tex": lambda path: write_correlation_figure(path, summaries, stability, system),
        "ccp_prefix_cohesion.tex": lambda path: write_scatter(path, rows, last, summaries[-1]["baseline"]),
        "ccp_family_ranks.tex": lambda path: write_rank_heatmap(path, rows, versions),
    }
    for name, write in outputs.items():
        write(args.results_dir / name)
        print(f"Saved {args.results_dir / name}")


if __name__ == "__main__":
    main()
