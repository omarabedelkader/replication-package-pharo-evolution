"""Recover a selector from the first tokens of a Smalltalk method definition."""

from __future__ import annotations

import re


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BINARY = re.compile(r"[-+*/\\<>=~@%&?!,|]+")
_BLANK = re.compile(r'(?:\s+|"[^"]*")*')


def selector_from_source(source: str) -> str | None:
    """Return the selector declared by the method pattern that starts ``source``.

    Handles the three Smalltalk message forms: unary (``size``), binary
    (``+ aNumber``) and keyword (``at: index put: value``), whose parts may be
    spread over several lines.  Returns ``None`` when no pattern is found.
    """
    position = _BLANK.match(source).end()

    binary = _BINARY.match(source, position)
    if binary:
        return binary.group()

    first = _IDENTIFIER.match(source, position)
    if not first:
        return None
    if not _is_keyword(source, first.end()):
        return first.group()

    parts: list[str] = []
    keyword = first
    while keyword and _is_keyword(source, keyword.end()):
        parts.append(keyword.group() + ":")
        position = _BLANK.match(source, keyword.end() + 1).end()
        argument = _IDENTIFIER.match(source, position)
        if not argument:
            return None
        position = _BLANK.match(source, argument.end()).end()
        keyword = _IDENTIFIER.match(source, position)

    return "".join(parts)


def _is_keyword(source: str, end: int) -> bool:
    """True when the identifier ending at ``end`` is followed by ``:`` but not ``:=``."""
    return source[end:end + 1] == ":" and source[end + 1:end + 2] != "="
