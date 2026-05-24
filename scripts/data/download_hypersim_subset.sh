#!/usr/bin/env bash
set -euo pipefail

# Usage example:
# bash scripts/data/download_hypersim_subset.sh \
#   --scenes ai_001_001 ai_001_002 \
#   --include-rgb --include-depth --include-metadata --force-extract

python scripts/data/download_hypersim_subset.py "$@"
