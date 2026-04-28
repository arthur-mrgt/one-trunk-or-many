#!/usr/bin/env bash
set -euo pipefail

# Usage example:
# bash scripts/download_hypersim_subset.sh \
#   --scenes ai_001_001 ai_001_002 \
#   --include-rgb --include-depth --include-metadata

python scripts/download_hypersim_subset.py "$@"
