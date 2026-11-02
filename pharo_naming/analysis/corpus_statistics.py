"""Size of the corpus per release.

    python -m pharo_naming.analysis.corpus_statistics
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..corpus import IdentifierRecord, read_corpus
from ..paths import CORPUS_DIR
from ..releases import find_release, release_index
from ..vocabulary import identifiers, package_family


def analyze_records(version: str, records: list[IdentifierRecord]) -> dict:
    """Count the entities of one release.

    ``unique_class_names`` and ``unique_selectors`` are the sizes of the
    normalized vocabularies that the cohesion analysis works on.
    """
    if not records:
        raise ValueError("Cannot analyze an empty record sequence.")

    packages = {record.package for record in records}

    return {
        "version": version,
        "release_index": release_index(version),
        "artifact_kind": find_release(version).kind,
        "packages": len(packages),
        "package_families": len({package_family(package) for package in packages}),
        "class_declarations": sum(record.kind == "class" for record in records),
        "unique_class_names": len(identifiers(records, "class", "vocabulary")),
        "method_declarations": sum(record.kind == "method" for record in records),
        "unique_selectors": len(identifiers(records, "method", "vocabulary")),
    }


def print_table(rows: list[dict]) -> None:
    print(f"{'Pharo':<6}{'Packages':>9}{'Classes':>9}{'Class names':>12}{'Methods':>9}{'Selectors':>10}")
    for row in rows:
        print(
            f"{row['version']:<6}{row['packages']:>9,}{row['class_declarations']:>9,}"
            f"{row['unique_class_names']:>12,}{row['method_declarations']:>9,}{row['unique_selectors']:>10,}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    args = parser.parse_args()

    rows = [
        analyze_records(version, records)
        for version, records in read_corpus(args.corpus_dir).items()
    ]

    print_table(rows)


if __name__ == "__main__":
    main()
