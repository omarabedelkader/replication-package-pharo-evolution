"""Table of the Pharo naming conventions that shape the two vocabularies.

    python -m pharo_naming.analysis.naming_conventions

The table describes the language and its community, not the corpus: it is
written here so that every file of ``results`` comes from ``run.sh``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..paths import RESULTS_DIR
from ..reporting import HEADER, write_text

# (convention, applies to, status, examples), in LaTeX.
CONVENTIONS = (
    ("One keyword per argument", "selectors", "language syntax", r"\textsf{at:put:}"),
    ("Recurring behavioral forms", "selectors", r"style guide~\cite{Duca22b}",
     r"\textsf{isEmpty}, \textsf{asString}, \textsf{printOn:}"),
    (r"\textsf{test} prefix", "selectors", "enforced by SUnit", r"\textsf{testAdd}"),
    (r"\textsf{Test(s)} suffix", "class names", "convention", r"\textsf{SetTest}"),
    ("Project prefix", "class names", "convention (global namespace)",
     r"\textsf{OBBrowser}, \textsf{ClyBrowser}"),
    (r"\textsf{Family-Part}", "package names", "convention", r"\textsf{Kernel-Objects}"),
)


def write_table(path: Path) -> None:
    column = r">{\raggedright\arraybackslash}p{%s}"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Naming conventions of \ph that shape the two vocabularies.}",
        r"\label{tab:conventions}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{" + "".join(column % width for width in ("2.4cm", "1.25cm", "1.7cm", "2.45cm")) + "}",
        r"\toprule",
        HEADER + r"Convention & Applies to & Status & Examples\\",
        r"\midrule",
        *(" & ".join(row) + r"\\" for row in CONVENTIONS),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    write_text(path, lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    path = args.results_dir / "naming_conventions.tex"
    write_table(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
