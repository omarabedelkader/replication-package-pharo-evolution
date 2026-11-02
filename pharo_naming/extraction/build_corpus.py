"""Build the declaration corpus: one file per Pharo release.

    python -m pharo_naming.extraction.build_corpus [--releases 1.0 1.1 ...]

Each release is downloaded once into the cache, read with the extractor that
matches its artifact, and written as sorted, de-duplicated JSON Lines.  The
exact artifact behind every release is pinned in ``pharo_naming/releases.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..corpus import IdentifierRecord, corpus_path, write_records
from ..paths import CACHE_DIR, CORPUS_DIR
from ..releases import RELEASE_LABELS, RELEASES, Release, find_release
from .download import fetch, read_image, read_tarball
from .filetree import read_filetree_declarations
from .image import read_image_declarations
from .model import Declaration
from .tonel import read_tonel_declarations


def extract(release: Release, archive: Path) -> list[Declaration]:
    """Read the declarations of ``release`` out of its downloaded archive."""
    if release.kind == "image":
        return read_image_declarations(read_image(archive))
    if release.kind == "filetree":
        return read_filetree_declarations(read_tarball(archive))
    if release.kind == "tonel":
        return read_tonel_declarations(read_tarball(archive, "src"))
    raise ValueError(f"unknown artifact kind {release.kind!r}")


def to_records(release: Release, declarations: list[Declaration]) -> list[IdentifierRecord]:
    """Drop excluded packages and duplicates, and order the records."""
    unique = {
        declaration for declaration in declarations
        if declaration.package not in release.excluded_packages
    }
    ordered = sorted(
        unique,
        key=lambda item: (
            item.package,
            item.class_name,
            item.kind,
            item.side or "",
            item.selector or "",
        ),
    )
    return [
        IdentifierRecord(
            version=release.label,
            package=item.package,
            class_name=item.class_name,
            kind=item.kind,
            side=item.side,
            selector=item.selector,
        )
        for item in ordered
    ]


def build(releases: list[Release], corpus_dir: Path, cache_dir: Path) -> None:
    for release in releases:
        print(f"{release.name}: fetching {release.url}", flush=True)
        archive = fetch(release, cache_dir)
        records = to_records(release, extract(release, archive))
        path = corpus_path(corpus_dir, release.label)
        write_records(path, records)
        print(
            f"{release.name}: "
            f"{len({record.package for record in records}):,} packages, "
            f"{sum(record.kind == 'class' for record in records):,} classes, "
            f"{sum(record.kind == 'method' for record in records):,} methods "
            f"-> {path.name}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument(
        "--releases",
        nargs="+",
        choices=RELEASE_LABELS,
        default=list(RELEASE_LABELS),
        metavar="VERSION",
        help="releases to build (default: all)",
    )
    args = parser.parse_args()

    build(
        [find_release(label) for label in args.releases],
        args.corpus_dir,
        args.cache_dir,
    )


if __name__ == "__main__":
    main()
