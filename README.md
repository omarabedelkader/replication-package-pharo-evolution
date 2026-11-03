# Replication package: the evolution of identifier names in Pharo

This package reproduces every table, figure and number of the paper
from the 18 Pharo releases (1.0 to 1.4, and 2 to 14).

## Run

```sh
./run.sh
```

Requirements: Python 3.12 or newer, `pdflatex`, and an Internet
connection for the first run (about 400 MB of downloads, kept in
`.cache/`). The first run takes about 15 minutes.

`./run.sh --skip-download` reuses the corpus in `results/corpus/`.

## Output

Everything is written to `results/`:

| File | In the paper |
| --- | --- |
| `naming_conventions.tex` | Table I |
| `robustness_trends.tex` | Table II (the paper shows its *Change* column for four of the analyses; this file has all of them) |
| `cohesion_decomposition.tex` | Figure 1 |
| `vocabulary_turnover.tex` | Figure 2 |
| `family_contributions.tex` | Figure 3 |
| `ccp_correlations.tex` | Figure 4 |
| `ccp_prefix_cohesion.tex` | Figure 5 |
| `frequent_prefixes.tex` | additional result: most frequent prefixes (Section V-A) |
| `ccp_family_ranks.tex` | additional result: family rankings over time (Section V-D) |
| `numbers.txt` | every number quoted in the text, under the section it belongs to |
| `corpus/` | the extracted classes and methods, one file per release |

Tables and figures are LaTeX files (figures are PGF code); include them
with `\input{results/...}`.

## Contents

| Path | Role |
| --- | --- |
| `run.sh` | the single entry point |
| `pharo_naming/releases.py` | the 18 releases, each pinned to a checksum or a Git commit |
| `pharo_naming/extraction/` | download and the three readers (image, FileTree, Tonel) |
| `pharo_naming/analysis/` | one module per step of `run.sh` |
| `tests/` | unit tests, run by `run.sh` |

The random samples use the fixed seed 2027, so every run gives the same
results. The family pairs of RQ4 were classified by hand; the
classification is in `PAIR_CATEGORIES` of
`pharo_naming/analysis/family_similarity_analysis.py`.
