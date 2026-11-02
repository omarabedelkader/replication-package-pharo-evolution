"""Check the corpus before it is analyzed.

    python -m pharo_naming.analysis.validate_corpus

Fails when a release is missing or when a corpus file is internally
inconsistent, so that no figure is ever produced from a broken extraction.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ..corpus import IdentifierRecord, available_versions, corpus_path, read_records
from ..paths import CORPUS_DIR
from ..releases import RELEASE_LABELS, find_release


def validate_records(version: str, records: list[IdentifierRecord]) -> dict:
    """Validate the records of one release and return its summary row."""
    if not records:
        raise ValueError(f"Pharo {version}: the corpus file is empty.")

    seen: set[IdentifierRecord] = set()
    for record in records:
        if record.version != version:
            raise ValueError(
                f"Pharo {version}: record belongs to version {record.version!r}"
            )
        if not record.package or not record.class_name:
            raise ValueError(f"Pharo {version}: incomplete record {record}")
        if record in seen:
            raise ValueError(f"Pharo {version}: duplicate declaration {record}")
        seen.add(record)

    classes = [record for record in records if record.kind == "class"]
    methods = [record for record in records if record.kind == "method"]
    if not classes or not methods:
        raise ValueError(f"Pharo {version}: no class or no method declaration.")

    declared = {record.class_name for record in classes}
    homes = Counter(record.class_name for record in classes)

    return {
        "version": version,
        "artifact_kind": find_release(version).kind,
        "records": len(records),
        "packages": len({record.package for record in records}),
        "class_declarations": len(classes),
        "method_declarations": len(methods),
        # Extensions of classes that live outside the release (rare).
        "methods_in_undeclared_classes": sum(
            record.class_name not in declared for record in methods
        ),
        "classes_declared_in_several_packages": sum(
            count > 1 for count in homes.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    args = parser.parse_args()

    missing = sorted(set(RELEASE_LABELS) - set(available_versions(args.corpus_dir)))
    if missing:
        raise SystemExit(
            f"Missing corpus files for Pharo {', '.join(missing)} in {args.corpus_dir}."
        )

    for version in RELEASE_LABELS:
        row = validate_records(
            version,
            list(read_records(corpus_path(args.corpus_dir, version))),
        )
        print(
            f"Pharo {version:<4} {row['artifact_kind']:<9}"
            f"{row['packages']:>5,} packages "
            f"{row['class_declarations']:>6,} classes "
            f"{row['method_declarations']:>8,} methods"
        )
    print(f"Validated {len(RELEASE_LABELS)} releases.")


if __name__ == "__main__":
    main()
