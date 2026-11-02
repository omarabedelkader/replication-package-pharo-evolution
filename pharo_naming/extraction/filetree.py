"""Read declarations from a FileTree export (the ``pharo-core`` repository).

Pharo 2 to 6 were mirrored, one package per directory, in
``github.com/pharo-project/pharo-core``.  Two layouts exist:

* Pharo 2 (FileTree with metadata)::

      Kernel.package/Object.class/properties.json
      Kernel.package/Object.class/instance/printString.st
      AST-Core.package/Behavior.extension/instance/parseTreeFor..st

  A method file starts with one line holding its protocol.

* Pharo 3 to 6 (metadata-less)::

      Kernel.package/Object.class/definition.st
      Kernel.package/Object.class/instance/printing/printString.st
      AST-Core.package/extension/Behavior/instance/parseTreeFor_.st

In both the package is the directory itself, so no inference is needed.
"""

from __future__ import annotations

from typing import Mapping

from .model import Declaration
from .selectors import selector_from_source


PACKAGE_SUFFIX = ".package"
DECLARING_SUFFIXES = (".class", ".trait")
EXTENSION_SUFFIX = ".extension"
EXTENSION_DIRECTORY = "extension"
SIDES = ("instance", "class")


class FileTreeError(ValueError):
    """Raised when an export does not have the expected structure."""


def read_filetree_declarations(files: Mapping[str, bytes]) -> list[Declaration]:
    """Extract every class and method declaration from an export.

    ``files`` maps each path, relative to the repository root, to its content.
    """
    has_protocol_line = any(path.endswith("/properties.json") for path in files)

    # Some exports list a method twice: under its protocol and under the
    # virtual "-- all --" protocol.  A dict keeps one copy, in reading order.
    declarations: dict[Declaration, None] = {}

    for path in sorted(files):
        parts = path.split("/")
        if len(parts) < 3 or not parts[0].endswith(PACKAGE_SUFFIX):
            continue
        package = parts[0][: -len(PACKAGE_SUFFIX)]

        if parts[1] == EXTENSION_DIRECTORY:
            class_name, inner, declares = parts[2], parts[3:], False
        else:
            class_name, dot, suffix = parts[1].rpartition(".")
            inner = parts[2:]
            if dot + suffix in DECLARING_SUFFIXES:
                declares = True
            elif dot + suffix == EXTENSION_SUFFIX:
                declares = False
            else:
                continue

        if declares:
            declarations[Declaration("class", package, class_name)] = None

        if len(inner) < 2 or inner[0] not in SIDES or not path.endswith(".st"):
            continue  # definition.st, properties.json, README.md, ...

        source = _decode(files[path])
        if has_protocol_line:
            source = source.partition("\n")[2]
        selector = selector_from_source(source)
        if selector is None:
            raise FileTreeError(f"cannot read the selector of {path}")

        declarations[
            Declaration("method", package, class_name, inner[0], selector)
        ] = None

    if not declarations:
        raise FileTreeError("no *.package directory found in the export")
    return list(declarations)


def _decode(data: bytes) -> str:
    """Decode a method file; wide-string sources were exported as raw UTF-32."""
    if data[:3] == b"\0\0\0" and len(data) % 4 == 0:
        return data.decode("utf-32-be", errors="replace")
    return data.decode("utf-8", errors="replace")
