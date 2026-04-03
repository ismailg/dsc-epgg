#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="${PYTHON_BIN}"
elif command -v python3.10 >/dev/null 2>&1; then
  PY=python3.10
else
  PY=python3
fi

PY_MINOR="$("${PY}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ "${PY_MINOR}" != "3.10" ]]; then
  echo "dsc-epgg-vectorized currently expects Python 3.10.x; got ${PY_MINOR} from ${PY}" >&2
  exit 1
fi

REQ_FILE="${REQ_FILE:-}"
if [[ -z "${REQ_FILE}" ]]; then
  if [[ -f requirements_locked.txt ]]; then
    REQ_FILE=requirements_locked.txt
  else
    REQ_FILE=requirements.txt
  fi
fi

$PY -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

# Project install
if [ -f "${REQ_FILE}" ]; then
  pip install -r "${REQ_FILE}"
fi
pip install --no-build-isolation -e .

# Lock exact environment for reproducibility
pip freeze > requirements_locked.txt
