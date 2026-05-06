#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="$PROJECT_ROOT/latex_report"
REPORT_TEX="$REPORT_DIR/pj1_report.tex"
TINYTEX_BIN="/Users/bytedance/Library/TinyTeX/bin/universal-darwin"

if [[ ! -f "$REPORT_TEX" ]]; then
  echo "Error: report source not found at $REPORT_TEX" >&2
  exit 1
fi

if [[ -d "$TINYTEX_BIN" ]]; then
  export PATH="$PATH:$TINYTEX_BIN"
fi

if ! command -v latexmk >/dev/null 2>&1; then
  echo "Error: latexmk not found. Please install TinyTeX or another TeX distribution first." >&2
  exit 1
fi

latexmk \
  -xelatex \
  -interaction=nonstopmode \
  -halt-on-error \
  -outdir="$REPORT_DIR" \
  -auxdir="$REPORT_DIR" \
  "$REPORT_TEX"

echo "Built: $REPORT_DIR/pj1_report.pdf"
