#!/usr/bin/env bash
# Shared environment for repository launchers. Source this file; do not execute it.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/third_party/unilm:$REPO_ROOT/models:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/outputs}"
