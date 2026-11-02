"""Read declarations from a Tonel source tree (Pharo 7 and later).

Since Pharo 7 the system is developed in ``github.com/pharo-project/pharo``
using Tonel: one directory per package under ``src/`` and one file per class,
trait or set of extension methods::

    src/Kernel/Object.class.st
    src/Kernel/TComparable.trait.st
    src/AST-Core/Behavior.extension.st

A file holds an optional comment, a ``Class { ... }`` description and then
the methods, each written as ``Name >> pattern [ body ]``.  Bodies are skipped
with a real scanner (strings, comments, character literals and nested
brackets) so that source text quoted inside a method can never be mistaken
for a declaration.  The few bodies whose brackets do not balance, such as a
literal array ``#( ] )``, are closed at the next ``]`` that starts a line and
is followed by another method or the end of the file.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from .model import Declaration
from .selectors import selector_from_source


DECLARING_SUFFIXES = (".class.st", ".trait.st")
EXTENSION_SUFFIX = ".extension.st"

_BLANK = re.compile(r"\s*")
_COMMENT = re.compile(r'"(?:[^"]|"")*"')
_TYPE = re.compile(r"(Class|Trait|Extension)\s*(?=\{)")
_NAME = re.compile(r"#name\s*:\s*#?'?([A-Za-z_][A-Za-z0-9_]*)'?")
_METHOD = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)(?:[ \t]+(class|classTrait|classSide))?[ \t]*>>"
)
_BOUNDARY = re.compile(
    r"[ \t]*(?:\n\s*(?:\{|" + _METHOD.pattern + r")|\s*\Z)"
)
_CLOSING_LINE = re.compile(r"\n\]")
_BRACE_TOKEN = re.compile(r"'(?:[^']|'')*'|[{}]")
_BODY_TOKEN = re.compile(r"""'(?:[^']|'')*'|"[^"]*"|\$.|[\[\]]""", re.DOTALL)


class TonelError(ValueError):
    """Raised when a Tonel file cannot be parsed."""


@dataclass(frozen=True)
class TonelMethod:
    class_name: str
    side: str
    selector: str


@dataclass(frozen=True)
class TonelFile:
    kind: str  # "Class", "Trait" or "Extension"
    name: str
    methods: tuple[TonelMethod, ...]


def parse_tonel(text: str) -> TonelFile:
    """Parse the content of one ``.st`` Tonel file."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    position = _BLANK.match(text).end()

    comment = _COMMENT.match(text, position)
    if comment:
        position = _BLANK.match(text, comment.end()).end()

    kind = _TYPE.match(text, position)
    if not kind:
        raise TonelError("missing Class/Trait/Extension description")
    description_end = _skip_balanced(text, kind.end(), _BRACE_TOKEN, "{", "}")
    name = _NAME.search(text, kind.end(), description_end)
    if not name:
        raise TonelError("description has no #name")

    methods: list[TonelMethod] = []
    position = _BLANK.match(text, description_end).end()

    while position < len(text):
        if text[position] == "{":  # method metadata, e.g. { #category : 'x' }
            position = _skip_balanced(text, position, _BRACE_TOKEN, "{", "}")
            position = _BLANK.match(text, position).end()
            continue

        header = _METHOD.match(text, position)
        if not header:
            raise TonelError(f"unexpected content at offset {position}")
        body_start = text.find("[", header.end())
        if body_start < 0:
            raise TonelError(f"method without body at offset {position}")

        pattern = text[header.end():body_start]
        selector = selector_from_source(pattern)
        if selector is None:
            raise TonelError(f"cannot read selector from {pattern.strip()!r}")

        methods.append(
            TonelMethod(
                class_name=header.group(1),
                side="class" if header.group(2) else "instance",
                selector=selector,
            )
        )
        position = _BLANK.match(text, _method_end(text, body_start)).end()

    return TonelFile(kind=kind.group(1), name=name.group(1), methods=tuple(methods))


def _method_end(text: str, body_start: int) -> int:
    """Return the offset just after the ``]`` closing the body at ``body_start``."""
    try:
        end = _skip_balanced(text, body_start, _BODY_TOKEN, "[", "]")
    except TonelError:
        end = -1
    if end > 0 and _ends_method(text, end):
        return end

    for closing in _CLOSING_LINE.finditer(text, body_start):
        if _ends_method(text, closing.end()):
            return closing.end()
    raise TonelError(f"unterminated method body at offset {body_start}")


def _ends_method(text: str, end: int) -> bool:
    """True when the ``]`` before ``end`` starts a line and a method or EOF follows."""
    return text[end - 2:end - 1] == "\n" and _BOUNDARY.match(text, end) is not None


def _skip_balanced(
    text: str,
    start: int,
    token: re.Pattern[str],
    opening: str,
    closing: str,
) -> int:
    """Return the offset just after the bracket closing the one at ``start``."""
    depth = 0
    for match in token.finditer(text, start):
        found = match.group()
        if found == opening:
            depth += 1
        elif found == closing:
            depth -= 1
            if depth == 0:
                return match.end()
    raise TonelError(f"unbalanced {opening!r} at offset {start}")


def read_tonel_declarations(files: Mapping[str, bytes]) -> list[Declaration]:
    """Extract every class and method declaration from a Tonel source tree.

    ``files`` maps ``<package>/<file>.st`` paths, relative to the ``src``
    directory, to their content.
    """
    declarations: list[Declaration] = []

    for path in sorted(files):
        package, _, file_name = path.rpartition("/")
        declares = file_name.endswith(DECLARING_SUFFIXES)
        if not package or "/" in package:
            continue
        if not declares and not file_name.endswith(EXTENSION_SUFFIX):
            continue  # package.st only describes the package

        try:
            parsed = parse_tonel(files[path].decode("utf-8", errors="replace"))
        except TonelError as error:
            raise TonelError(f"{path}: {error}") from error

        if declares:
            declarations.append(Declaration("class", package, parsed.name))
        declarations.extend(
            Declaration("method", package, method.class_name, method.side, method.selector)
            for method in parsed.methods
        )

    if not declarations:
        raise TonelError("no Tonel source file found")
    return declarations
