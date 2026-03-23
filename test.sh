#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$ROOT_DIR"

"$ROOT_DIR/.venv/bin/ruff" format "$ROOT_DIR"
"$ROOT_DIR/.venv/bin/ruff" check "$ROOT_DIR"
"$ROOT_DIR/.venv/bin/mypy" "$ROOT_DIR"
"$ROOT_DIR/.venv/bin/pytest" "$ROOT_DIR/tests"
