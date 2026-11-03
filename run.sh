#!/usr/bin/env bash
#
# Reproduces every table, figure and number of the paper:
#
#   1. creates a Python environment (.venv) with the pinned dependencies;
#   2. runs the unit tests;
#   3. downloads the 18 Pharo releases (cached in .cache) and extracts
#      their classes and methods into results/corpus;
#   4. writes the tables and figures to results/ as .tex files, and every
#      number printed along the way to results/numbers.txt.
#
# Usage: ./run.sh [--skip-download] [--skip-tests]
#   --skip-download  reuse the corpus already in results/corpus
#   --skip-tests     do not run the unit tests

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$ROOT_DIR/results"
VENV_DIR="$ROOT_DIR/.venv"
NUMBERS="$RESULTS_DIR/numbers.txt"

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
    printf '\n==> %s\n' "$1" | tee -a "$NUMBERS"
}

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
: > "$NUMBERS"

step "Preparing the Python environment"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$(find_python)" -m venv "$VENV_DIR"
fi
PYTHON_BIN="$VENV_DIR/bin/python"
"$PYTHON_BIN" -m pip install --quiet --disable-pip-version-check -r requirements.txt
"$PYTHON_BIN" --version

export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$MPLCONFIGDIR"

# Runs one module; its output goes to the terminal and to results/numbers.txt.
# Paths are printed relative to the package, so the output names no local folder.
run() {
    "$PYTHON_BIN" -m "$@" 2>&1 | sed "s|$ROOT_DIR/||g" | tee -a "$NUMBERS"
}

if [[ "$RUN_TESTS" -eq 1 ]]; then
    step "Unit tests"
    "$PYTHON_BIN" -m pytest -q -p no:cacheprovider tests
fi

if [[ "$BUILD_CORPUS" -eq 1 ]]; then
    step "Corpus: downloading the releases and extracting their declarations"
    run pharo_naming.extraction.build_corpus
    step "Section IV: check of the image reader on Pharo 2"
    run pharo_naming.analysis.validate_image_reader
fi

step "Corpus validation";                             run pharo_naming.analysis.validate_corpus
step "Section IV: Table II (corpus size)";          run pharo_naming.analysis.corpus_statistics
step "Table I: naming conventions";                  run pharo_naming.analysis.naming_conventions
step "RQ1: whole-system prefix cohesion";            run pharo_naming.analysis.global_analysis
step "RQ1: Table II and Figure 2 (prefixes, turnover)"; run pharo_naming.analysis.turnover_analysis
step "RQ2: package families";                        run pharo_naming.analysis.package_family_analysis
step "RQ1 and RQ3: Table III and Figure 1";          run pharo_naming.analysis.robustness_analysis
step "RQ3: Figure 3 (family contributions)";         run pharo_naming.analysis.family_contribution_analysis
step "RQ4: Figures 4 to 6 (conceptual cohesion)";    run pharo_naming.analysis.conceptual_cohesion_analysis
step "RQ4: similarity between families";             run pharo_naming.analysis.family_similarity_analysis

step "Done: tables and figures in results/, numbers in results/numbers.txt"
