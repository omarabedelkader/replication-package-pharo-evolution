"""The neutral record every extractor produces."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Declaration:
    """A class (or trait) declaration, or a method declaration."""

    kind: str  # "class" or "method"
    package: str
    class_name: str
    side: str | None = None  # "instance" or "class", methods only
    selector: str | None = None  # methods only
