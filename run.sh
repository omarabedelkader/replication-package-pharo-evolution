#!/usr/bin/env bash
#
# Reproduces every table, figure and number of the paper:
#
#   1. creates a Python environment (.venv) with the pinned dependencies;
#   2. runs the unit tests;
#   3. downloads the 14 Pharo releases (cached in .cache) and extracts
#      their classes and methods into results/corpus;
#   4. writes the tables and figures of the paper to results/ as .tex files.
#
# The terminal shows only the steps; everything the scripts print (including
# every number quoted in the paper) goes to run.log.
#
# Usage: ./run.sh [--skip-download] [--skip-tests]
#   --skip-download  reuse the corpus already in results/corpus
#   --skip-tests     do not run the unit tests

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$ROOT_DIR/results"
VENV_DIR="$ROOT_DIR/.venv"
LOG="$ROOT_DIR/run.log"

RUN_TESTS=1
BUILD_CORPUS=1
for argument in "$@"; do
    case "$argument" in
        --skip-download) BUILD_CORPUS=0 ;;
        --skip-tests)    RUN_TESTS=0 ;;
        -h|--help)       sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)               echo "Unknown option: $argument (see ./run.sh --help)" >&2; exit 2 ;;
    esac
done

step() {
    printf '==> %s\n' "$1"
    printf '\n==> %s\n' "$1" >> "$LOG"
}

# On failure, point to the log instead of failing silently.
trap 'status=$?; [[ $status -ne 0 ]] && echo "Failed (exit $status): see run.log" >&2' EXIT

find_python() {
    local candidate
    for candidate in "${PYTHON:-}" python3 python; do
        if [[ -n "$candidate" ]] && command -v "$candidate" >/dev/null 2>&1 \
            && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
            echo "$candidate"
            return
        fi
    done
    echo "Python 3.12 or newer is required (or set PYTHON=/path/to/python)." >&2
    exit 1
}

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "pdflatex is required: the figures are written as LaTeX (PGF) code." >&2
    exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$RESULTS_DIR"
: > "$LOG"
rm -f "$RESULTS_DIR/numbers.txt"

step "Preparing the Python environment"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$(find_python)" -m venv "$VENV_DIR"
fi
PYTHON_BIN="$VENV_DIR/bin/python"
"$PYTHON_BIN" -m pip install --quiet --disable-pip-version-check -r requirements.txt >> "$LOG" 2>&1
"$PYTHON_BIN" --version >> "$LOG" 2>&1

export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$MPLCONFIGDIR"

# Runs one module; its output goes to run.log.
# Paths are printed relative to the package, so the log names no local folder.
run() {
    "$PYTHON_BIN" -m "$@" 2>&1 | sed "s|$ROOT_DIR/||g" >> "$LOG"
}

if [[ "$RUN_TESTS" -eq 1 ]]; then
    step "Unit tests"
    run pytest -q -p no:cacheprovider tests
fi

if [[ "$BUILD_CORPUS" -eq 1 ]]; then
    step "Corpus: results/corpus/ (download and extraction of the 14 releases)"
    run pharo_naming.extraction.build_corpus
    step "Check of the image reader on Pharo 2"
    run pharo_naming.analysis.validate_image_reader
fi

step "Check of the corpus";                                         run pharo_naming.analysis.validate_corpus
step "Table I: results/naming_conventions.tex";                     run pharo_naming.analysis.naming_conventions
step "Table II: results/corpus_size.tex";                           run pharo_naming.analysis.corpus_statistics
step "Whole-system prefix cohesion (numbers in run.log)";           run pharo_naming.analysis.global_analysis
step "Vocabulary turnover: results/vocabulary_turnover.tex, results/frequent_prefixes.tex"
run pharo_naming.analysis.turnover_analysis
step "Package families (numbers in run.log)";                       run pharo_naming.analysis.package_family_analysis
step "Table III and Figure 1: results/robustness_trends.tex, results/cohesion_decomposition.tex, results/robustness_full.tex"
run pharo_naming.analysis.robustness_analysis
step "Figure 2: results/family_contributions.tex";                  run pharo_naming.analysis.family_contribution_analysis
step "Figure 3: results/ccp_correlations.tex (also results/ccp_prefix_cohesion.tex, results/ccp_family_ranks.tex)"
run pharo_naming.analysis.conceptual_cohesion_analysis
step "Similarity between families (numbers in run.log)";            run pharo_naming.analysis.family_similarity_analysis

step "Done: tables and figures in results/, log in run.log"
