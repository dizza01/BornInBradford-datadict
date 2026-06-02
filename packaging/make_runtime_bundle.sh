#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${1:-${REPO_DIR}/dist/bib-assistant-runtime}"
MODEL_FILE="${MODEL_FILE:-${REPO_DIR}/models/bib-llama-3.1-8b.Q4_K_M.gguf}"
RCADS_FILE="${RCADS_FILE:-${REPO_DIR}/papers/RCADS25-Youth-English-2018.pdf}"

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "Missing required path: $1" >&2
    exit 1
  fi
}

require_path "${REPO_DIR}/llm_poc/server.py"
require_path "${REPO_DIR}/llm_poc/bib_research_assistant.py"
require_path "${REPO_DIR}/llm_poc/.chroma_db/chroma.sqlite3"
require_path "${REPO_DIR}/docs"
require_path "${MODEL_FILE}"
require_path "${RCADS_FILE}"

rm -rf "${OUT_DIR}"
mkdir -p "${OUT_DIR}/llm_poc" "${OUT_DIR}/models" "${OUT_DIR}/papers"

echo "Copying runtime code..."
cp "${REPO_DIR}/llm_poc/server.py" "${OUT_DIR}/llm_poc/server.py"
cp "${REPO_DIR}/llm_poc/bib_research_assistant.py" "${OUT_DIR}/llm_poc/bib_research_assistant.py"
cp "${REPO_DIR}/llm_poc/bib_utils.py" "${OUT_DIR}/llm_poc/bib_utils.py"
cp "${REPO_DIR}/llm_poc/production_data_dictionary_reference.md" "${OUT_DIR}/llm_poc/production_data_dictionary_reference.md"
cp "${SCRIPT_DIR}/runtime_requirements.txt" "${OUT_DIR}/llm_poc/requirements_runtime.txt"
cp "${SCRIPT_DIR}/runtime.env" "${OUT_DIR}/llm_poc/.env"

echo "Copying static UI..."
cp -R "${REPO_DIR}/llm_poc/static" "${OUT_DIR}/llm_poc/static"

echo "Copying prebuilt Chroma DB..."
cp -R "${REPO_DIR}/llm_poc/.chroma_db" "${OUT_DIR}/llm_poc/.chroma_db"

echo "Copying data dictionary docs..."
cp -R "${REPO_DIR}/docs" "${OUT_DIR}/docs"

echo "Copying required source PDFs..."
cp "${RCADS_FILE}" "${OUT_DIR}/papers/$(basename "${RCADS_FILE}")"

echo "Copying quantized GGUF model..."
cp "${MODEL_FILE}" "${OUT_DIR}/models/$(basename "${MODEL_FILE}")"

echo "Copying launchers and runtime README..."
cp "${SCRIPT_DIR}/start_mac.sh" "${OUT_DIR}/start_mac.sh"
cp "${SCRIPT_DIR}/start_windows.ps1" "${OUT_DIR}/start_windows.ps1"
cp "${SCRIPT_DIR}/README_RUNTIME.md" "${OUT_DIR}/README_RUNTIME.md"
chmod +x "${OUT_DIR}/start_mac.sh"

cat <<EOF

Runtime bundle created:
  ${OUT_DIR}

Approximate size:
  $(du -sh "${OUT_DIR}" | awk '{print $1}')

To test on this Mac:
  cd "${OUT_DIR}"
  ./start_mac.sh

EOF
