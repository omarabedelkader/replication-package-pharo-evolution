"""Vocabulary shared between package families, and why the most similar ones are similar.

    python -m pharo_naming.analysis.family_similarity_analysis

CCP measures how alike the classes of one family are.  The same TF-IDF vectors
give the similarity *between* two families: the mean cosine similarity over
all pairs made of one class of each.  For every analyzed family we compare its
most similar other family with its own CCP; a ratio of one or more means that
the family boundary does not separate the two vocabularies.

The ten most similar pairs of every release were inspected by hand, with the
words that contribute most to their similarity and the packages of each
family, and assigned to one of the categories of ``CATEGORIES``.
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..concepts import between_group_similarity
from ..corpus import IdentifierRecord, read_corpus
from ..paths import CORPUS_DIR
from .conceptual_cohesion_analysis import class_documents, family_ccp

Corpus = dict[str, list[IdentifierRecord]]

# Most similar pairs inspected in every release.
TOP_PAIRS = 10
SHARED_WORDS = 5

CATEGORIES = {
    "split": "One project, several families",
    "successor": "A family and its successor",
    "related": "Related subsystems",
    "tests": "Test families (setUp, tearDown)",
}

# Manual classification of the pairs that are among the TOP_PAIRS most similar
# of at least one release.  *split*: the packages of one project do not share
# the part of their name before the first hyphen.  *successor*: one family
# replaces the other, or receives its deprecated classes.  *related*: one
# family builds on the other, or both implement the same domain (class
# definitions, files, commands).  *tests*: test families of Pharo 1.0 to 2 whose only shared
# words are those of the SUnit protocol.
PAIR_CATEGORIES: dict[frozenset[str], str] = {
    frozenset(pair): category
    for category, pairs in {
        "split": [
            ("Monticello", "MonticelloFileTree"), ("Monticello", "MonticelloGUI"),
            ("Monticello", "MonticelloMocks"), ("MonticelloFileTree", "MonticelloRemoteRepositories"),
            ("MonticelloMocks", "MonticelloRemoteRepositories"), ("Epicea", "EpiceaBrowsers"),
            ("Epicea", "EpiceaTests"), ("EpiceaBrowsers", "EpiceaTests"), ("Fuel", "FuelTests"),
            ("FuelPlatform", "FuelTests"), ("Ocompletion", "OcompletionTests"),
            ("SmartSuggestions", "SmartSuggestionsTests"), ("Slot", "SlotTests"),
            ("Ombu", "OmbuTests"), ("Sound", "SoundMorphicUserInterface"), ("OB", "OmniBrowser"),
            ("GeneralRules", "Renraku"), ("General", "Renraku"),
        ],
        "successor": [
            ("Refactoring", "Refactoring2"), ("BlueInk", "EnlumineurFormatter"),
            ("Compiler", "OpalCompiler"), ("FFI", "UnifiedFFI"), ("FileSystem", "Files"),
            ("Deprecated50", "UnifiedFFI"),
        ],
        "related": [
            ("Commander", "SystemCommands"), ("Commander2", "SystemCommands"),
            ("Files", "MonticelloFileTree"), ("FileSystem", "MonticelloFileTree"),
            ("Compression", "Files"), ("MonticelloGUI", "Tools"), ("MonticelloGUI", "ToolBuilder"),
            ("MonticelloGUI", "Spec"), ("Gofer", "Metacello"), ("Metacello", "MonticelloGUI"),
            ("NautilusRefactoring", "Refactoring"), ("GeneralRules", "Refactoring"),
            ("RecentSubmissions", "Tools"), ("MonticelloGUI", "RecentSubmissions"),
            ("PreferenceBrowser", "Services"), ("DebuggerActions", "SmartSuggestions"),
            ("MorphExamplesAndDemos", "SoundMorphicUserInterface"), ("EmbeddedFreeType", "Fonts"),
            ("Slot", "VariablesLibrary"), ("MonticelloFileTree", "Ring"), ("Ring", "Traits"),
            ("RPackage", "Ring"), ("ClassParser", "TraitsV2"), ("ClassParser", "Traits"),
            ("ClassDefinitionPrinters", "ClassParser"), ("FluidClassBuilder", "TraitsV2"),
            ("ClassParser", "FluidClassBuilder"), ("Tests", "TraitsV2"),
            ("MonticelloFileTree", "RPackage"), ("MonticelloFileTree", "Traits"),
        ],
        "tests": [
            ("MorphicTests", "OcompletionTests"), ("MorphicTests", "NetworkTests"),
            ("KernelTests", "MorphicTests"), ("KernelTests", "OcompletionTests"),
            ("CollectionsTests", "MorphicTests"), ("MorphicTests", "Tests"),
            ("CollectionsTests", "KernelTests"), ("KernelTests", "NetworkTests"),
        ],
    }.items()
    for pair in pairs
}


def analyze_release(version: str, records: list[IdentifierRecord]) -> tuple[list[dict], list[float]]:
    """Every pair of analyzed families with its similarity, and the ratio of
    each family's nearest-family similarity to its own CCP."""
    documents = class_documents(records)
    ccp, _, data = family_ccp(documents)
    vectors, rows = data["vectors"], data["rows"]

    names = [name for name, (_, document) in documents.items() if document]
    vocabulary: dict[str, int] = {}
    for name in names:
        for word in documents[name][1]:
            vocabulary.setdefault(word, len(vocabulary))
    words = np.array(sorted(vocabulary, key=vocabulary.get))
    sums = {family: np.asarray(vectors[indices].sum(axis=0)).ravel() for family, indices in rows.items()}

    pairs = []
    for first, second in itertools.combinations(sorted(rows), 2):
        similarity = between_group_similarity(vectors[rows[first]], vectors[rows[second]])
        contribution = sums[first] * sums[second]
        pairs.append({
            "version": version,
            "first": first,
            "second": second,
            "similarity": similarity,
            "ratio": similarity / min(ccp[first], ccp[second]),
            "words": [str(word) for word in words[np.argsort(-contribution)[:SHARED_WORDS]]],
        })
    nearest = defaultdict(float)
    for pair in pairs:
        for family in (pair["first"], pair["second"]):
            nearest[family] = max(nearest[family], pair["similarity"])
    return pairs, [nearest[family] / ccp[family] for family in rows]


def analyze(corpus: Corpus) -> tuple[list[dict], np.ndarray]:
    top, ratios = [], []
    for version, records in corpus.items():
        pairs, release_ratios = analyze_release(version, records)
        top.extend(sorted(pairs, key=lambda pair: -pair["similarity"])[:TOP_PAIRS])
        ratios.extend(release_ratios)
    return top, np.array(ratios)


def category(pair: dict) -> str:
    return PAIR_CATEGORIES.get(frozenset((pair["first"], pair["second"])), "unclassified")


def distinct_pairs(top: list[dict]) -> dict[frozenset[str], list[dict]]:
    pairs: dict[frozenset[str], list[dict]] = defaultdict(list)
    for pair in top:
        pairs[frozenset((pair["first"], pair["second"]))].append(pair)
    return pairs


def print_examples(top: list[dict]) -> None:
    """The two pairs of each category that are most often among the most similar."""
    by_category: dict[str, list[list[dict]]] = defaultdict(list)
    for occurrences in distinct_pairs(top).values():
        by_category[category(occurrences[0])].append(occurrences)
    for key, label in CATEGORIES.items():
        members = sorted(by_category.get(key, []), key=lambda occurrences: -len(occurrences))
        examples = "; ".join(
            f"{occurrences[-1]['first']}--{occurrences[-1]['second']} ({', '.join(occurrences[-1]['words'][:3])})"
            for occurrences in members[:2]
        )
        print(f"{label}: {len(members)} pairs, e.g. {examples}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    args = parser.parse_args()

    corpus = read_corpus(args.corpus_dir)
    top, ratios = analyze(corpus)
    print(
        f"Nearest family / own CCP over {len(ratios)} family-releases: median {np.median(ratios):.2f}, "
        f"90th percentile {np.percentile(ratios, 90):.2f}, at least one in {np.mean(ratios >= 1):.1%}"
    )
    pairs = distinct_pairs(top)
    counts = Counter(category(occurrences[0]) for occurrences in pairs.values())
    print(f"{len(pairs)} distinct pairs among the {TOP_PAIRS} most similar of each release: {dict(counts)}")
    for key, occurrences in sorted(pairs.items(), key=lambda item: -len(item[1])):
        last = occurrences[-1]
        print(f"  {category(last):<12} {last['first']}--{last['second']}: {len(occurrences)} releases, "
              f"last {last['version']} similarity {last['similarity']:.3f} "
              f"(ratio {last['ratio']:.2f}) {last['words']}")
    unclassified = [key for key, occurrences in pairs.items() if category(occurrences[0]) == "unclassified"]
    if unclassified:
        print(f"WARNING: {len(unclassified)} pairs need a manual category: {unclassified}")
    print_examples(top)


if __name__ == "__main__":
    main()
