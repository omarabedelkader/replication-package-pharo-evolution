"""The declaration corpus: one compressed JSON Lines file per release."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator

from .releases import RELEASE_LABELS


CORPUS_PREFIX = "pharo-"
CORPUS_SUFFIX = ".jsonl.gz"
KINDS = ("class", "method")
SIDES = ("instance", "class")


@dataclass(frozen=True)
class IdentifierRecord:
    """A class declaration or a method declaration of one release."""

    version: str
    package: str
    class_name: str
    kind: str
    side: str | None = None
    selector: str | None = None


def corpus_path(corpus_dir: Path, version: str) -> Path:
    return corpus_dir / f"{CORPUS_PREFIX}{version}{CORPUS_SUFFIX}"


def write_records(path: Path, records: Iterable[IdentifierRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # No name and no timestamp: the archive is byte-identical between runs.
    with open(path, "wb") as raw, gzip.GzipFile("", "wb", fileobj=raw, mtime=0) as stream:
        for record in records:
            row = {
                "version": record.version,
                "kind": record.kind,
                "package": record.package,
                "class": record.class_name,
            }
            if record.kind == "method":
                row["side"] = record.side
                row["selector"] = record.selector
            stream.write(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n")


def read_records(path: Path) -> Iterator[IdentifierRecord]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                yield _parse_record(path, line_number, line)


def _parse_record(path: Path, line_number: int, line: str) -> IdentifierRecord:
    where = f"{path.name}:{line_number}"
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"{where}: invalid JSON") from error

    kind = raw.get("kind")
    if kind not in KINDS:
        raise ValueError(f"{where}: invalid kind {kind!r}")

    side = raw.get("side")
    selector = raw.get("selector")
    if kind == "method":
        if side not in SIDES:
            raise ValueError(f"{where}: invalid side {side!r}")
        if not selector:
            raise ValueError(f"{where}: method record has no selector")

    return IdentifierRecord(
        version=str(raw["version"]),
        package=str(raw["package"]),
        class_name=str(raw["class"]),
        kind=kind,
        side=side,
        selector=selector,
    )


def available_versions(corpus_dir: Path) -> list[str]:
    """Releases that have a corpus file, in historical order."""
    return [
        version for version in RELEASE_LABELS
        if corpus_path(corpus_dir, version).exists()
    ]


def read_corpus(corpus_dir: Path) -> dict[str, list[IdentifierRecord]]:
    """Load every available release, keyed by version in historical order."""
    versions = available_versions(corpus_dir)
    if not versions:
        raise FileNotFoundError(
            f"no corpus file in {corpus_dir}; build it first with "
            "`python -m pharo_naming.extraction.build_corpus`"
        )
    return {
        version: list(read_records(corpus_path(corpus_dir, version)))
        for version in versions
    }
