#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/llm_poc/requirements_runtime.txt" ]]; then
  ROOT_DIR="${SCRIPT_DIR}"
  REQUIREMENTS_FILE="${ROOT_DIR}/llm_poc/requirements_runtime.txt"
elif [[ -f "${SCRIPT_DIR}/runtime_requirements.txt" && -d "${SCRIPT_DIR}/../llm_poc" ]]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
  REQUIREMENTS_FILE="${SCRIPT_DIR}/runtime_requirements.txt"
else
  echo "Could not find runtime files."
  echo "Run this from dist/bib-assistant-runtime/start_mac.sh, or rebuild with packaging/make_runtime_bundle.sh."
  exit 1
fi

VENV_DIR="${ROOT_DIR}/.venv"
MODEL_PATH="${ROOT_DIR}/models/bib-llama-3.1-8b.Q4_K_M.gguf"

cd "${ROOT_DIR}"
export BIB_RUNTIME_LOCKED=1

find_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi
  for candidate in python3.11 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      echo "${candidate}"
      return
    fi
  done
  echo ""
}

check_python_version() {
  local python_cmd="$1"
  "${python_cmd}" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(1)
PY
}

PYTHON_BIN="$(find_python)"
if [[ -z "${PYTHON_BIN}" ]] || ! check_python_version "${PYTHON_BIN}"; then
  echo "Python 3.11 or newer is required."
  echo "Install Python 3.11+ and rerun this script."
  echo "On macOS with Homebrew: brew install python@3.11"
  echo "Alternatively set PYTHON_BIN=/path/to/python3.11 before running."
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating local Python environment..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python - <<'PY' || {
import sys
if sys.version_info < (3, 11):
    raise SystemExit(1)
PY
  echo "The existing .venv was created with an older Python."
  echo "Delete ${VENV_DIR}, install Python 3.11+, then rerun this script."
  exit 1
}
python -m pip install --upgrade pip
python -m pip install -r "${REQUIREMENTS_FILE}"

python - <<'PY' >/dev/null 2>&1 || {
import llama_cpp
PY
  echo "Installing llama-cpp-python with Apple Metal support..."
  CMAKE_ARGS="-DGGML_METAL=on" python -m pip install -U llama-cpp-python --no-cache-dir
}

if [[ ! -f "${ROOT_DIR}/llm_poc/.chroma_db/chroma.sqlite3" ]]; then
  echo "Missing Chroma DB at llm_poc/.chroma_db/chroma.sqlite3"
  echo "This runtime bundle expects a prebuilt index."
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Missing GGUF model at ${MODEL_PATH}"
  exit 1
fi

echo "Starting Born in Bradford assistant..."
echo "Open http://127.0.0.1:5050/assistant"

exec python "${ROOT_DIR}/llm_poc/server.py" \
  --llm-backend llama_cpp \
  --gguf-model-path "${MODEL_PATH}" \
  --llama-n-ctx "${LLAMA_N_CTX:-4096}" \
  --llama-n-gpu-layers "${LLAMA_N_GPU_LAYERS:--1}" \
  --rag-n-results "${RAG_N_RESULTS:-3}" \
  --rag-context-max-chars "${RAG_CONTEXT_MAX_CHARS:-3500}" \
  "$@"
