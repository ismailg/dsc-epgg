#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv}"
REQ_FILE="${REQ_FILE:-requirements_locked.txt}"
TORCH_CPU_INDEX_URL="${TORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
BACKUP_EXISTING_VENV="${BACKUP_EXISTING_VENV:-0}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

PY_MINOR="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ "${PY_MINOR}" != "3.10" ]]; then
  echo "Hetzner CPU bootstrap expects Python 3.10.x; got ${PY_MINOR} from ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "Requirements file not found: ${REQ_FILE}" >&2
  exit 1
fi

TORCH_LINE="$(grep -E '^torch==[^[:space:]]+$' "${REQ_FILE}" | head -n 1 || true)"
if [[ -z "${TORCH_LINE}" ]]; then
  echo "Expected an exact torch pin in ${REQ_FILE}" >&2
  exit 1
fi
TORCH_VERSION="${TORCH_LINE#torch==}"

TMP_REQ="$(mktemp)"
trap 'rm -f "${TMP_REQ}"' EXIT
grep -v -E '^torch==[^[:space:]]+$' "${REQ_FILE}" > "${TMP_REQ}"

if [[ -d "${VENV_DIR}" ]]; then
  if [[ "${BACKUP_EXISTING_VENV}" == "1" ]]; then
    BACKUP_DIR="${VENV_DIR}.bak-$(date +%Y%m%d%H%M%S)"
    mv "${VENV_DIR}" "${BACKUP_DIR}"
    echo "Backed up existing virtualenv to ${BACKUP_DIR}"
  else
    rm -rf "${VENV_DIR}"
    echo "Removed existing virtualenv at ${VENV_DIR}"
  fi
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r "${TMP_REQ}"
"${VENV_DIR}/bin/python" -m pip install --index-url "${TORCH_CPU_INDEX_URL}" "torch==${TORCH_VERSION}"
"${VENV_DIR}/bin/python" -m pip install --no-build-isolation -e .

"${VENV_DIR}/bin/python" - <<'PY'
import pettingzoo
import sys
import torch

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("pettingzoo", pettingzoo.__version__)
PY
