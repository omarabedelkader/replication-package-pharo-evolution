"""Shared helpers for writing LaTeX fragments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


# Starts a column-header row of a table; the macro is defined in main.tex
# (gray background).
HEADER = r"\headerrow "


def latex_escape(text: object) -> str:
    return "".join(_LATEX_ESCAPES.get(character, character) for character in str(text))


def fmt(value: object, digits: int = 3) -> str:
    """Format a number for a LaTeX table; missing values become a dash."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def signed(value: float, digits: int) -> str:
    """A signed number for a LaTeX table, green when positive and red when
    negative (black when it rounds to zero)."""
    if round(value, digits) == 0:
        return f"{0:.{digits}f}"
    color = "green!50!black" if value > 0 else "red!75!black"
    sign = "+" if value > 0 else "$-$"
    return rf"\textcolor{{{color}}}{{{sign}{abs(value):.{digits}f}}}"


def write_text(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
