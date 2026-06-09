#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT="${PROJECT:-bib-assistant}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-bib-assistant-gpu-primary}"
REMOTE_DIR="${REMOTE_DIR:-/home/dawud_izza_york_ac_uk/bib-assistant-runtime}"
REMOTE_ARCHIVE="${REMOTE_ARCHIVE:-/tmp/bib-assistant-code-update.tar.gz}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"
INCLUDE_DOCS="${INCLUDE_DOCS:-0}"
USE_SUDO="${USE_SUDO:-1}"
REMOTE_OWNER="${REMOTE_OWNER:-dawud_izza_york_ac_uk:dawud_izza_york_ac_uk}"
REMOTE_RUN_USER="${REMOTE_OWNER%%:*}"

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "Missing required path: $1" >&2
    exit 1
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_cmd gcloud
require_path "${REPO_DIR}/llm_poc/server.py"
require_path "${REPO_DIR}/llm_poc/bib_research_assistant.py"
require_path "${REPO_DIR}/llm_poc/bib_utils.py"
require_path "${REPO_DIR}/llm_poc/static"
require_path "${REPO_DIR}/llm_poc/production_data_dictionary_reference.md"
require_path "${SCRIPT_DIR}/runtime_requirements.txt"
require_path "${SCRIPT_DIR}/runtime.env"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

UPDATE_DIR="${TMP_DIR}/bib-assistant-code-update"
mkdir -p "${UPDATE_DIR}/llm_poc"

echo "Preparing code-only update package..."
cp "${REPO_DIR}/llm_poc/server.py" "${UPDATE_DIR}/llm_poc/server.py"
cp "${REPO_DIR}/llm_poc/bib_research_assistant.py" "${UPDATE_DIR}/llm_poc/bib_research_assistant.py"
cp "${REPO_DIR}/llm_poc/bib_utils.py" "${UPDATE_DIR}/llm_poc/bib_utils.py"
cp "${REPO_DIR}/llm_poc/production_data_dictionary_reference.md" "${UPDATE_DIR}/llm_poc/production_data_dictionary_reference.md"
cp "${SCRIPT_DIR}/runtime_requirements.txt" "${UPDATE_DIR}/llm_poc/requirements_runtime.txt"
cp "${SCRIPT_DIR}/runtime.env" "${UPDATE_DIR}/llm_poc/.env"
cp -R "${REPO_DIR}/llm_poc/static" "${UPDATE_DIR}/llm_poc/static"

if [[ "${INCLUDE_DOCS}" == "1" ]]; then
  require_path "${REPO_DIR}/docs"
  echo "Including docs/ in update package..."
  cp -R "${REPO_DIR}/docs" "${UPDATE_DIR}/docs"
fi

ARCHIVE="${TMP_DIR}/bib-assistant-code-update.tar.gz"
tar -czf "${ARCHIVE}" -C "${UPDATE_DIR}" .

echo "Uploading update package to ${VM_NAME}..."
gcloud compute scp "${ARCHIVE}" "${VM_NAME}:${REMOTE_ARCHIVE}" \
  --zone="${ZONE}" \
  --project="${PROJECT}"

echo "Applying update and restarting service..."
if [[ "${USE_SUDO}" == "1" ]]; then
  gcloud compute ssh "${VM_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --command="set -euo pipefail
sudo mkdir -p '${REMOTE_DIR}'
sudo tar -xzf '${REMOTE_ARCHIVE}' -C '${REMOTE_DIR}'
if [[ '${REMOTE_OWNER}' != '' ]]; then
  sudo chown -R '${REMOTE_OWNER}' '${REMOTE_DIR}'
fi
if [[ '${INSTALL_REQUIREMENTS}' == '1' ]]; then
  sudo -u '${REMOTE_RUN_USER}' bash -lc \"cd '${REMOTE_DIR}' && . .venv/bin/activate && python -m pip install -r llm_poc/requirements_runtime.txt\"
fi
sudo systemctl restart bib-assistant
sleep 5
sudo systemctl status bib-assistant --no-pager --lines=30"
else
  gcloud compute ssh "${VM_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT}" \
    --command="set -euo pipefail
mkdir -p '${REMOTE_DIR}'
tar -xzf '${REMOTE_ARCHIVE}' -C '${REMOTE_DIR}'
cd '${REMOTE_DIR}'
if [[ '${INSTALL_REQUIREMENTS}' == '1' ]]; then
  . .venv/bin/activate
  python -m pip install -r llm_poc/requirements_runtime.txt
fi
sudo systemctl restart bib-assistant
sleep 5
sudo systemctl status bib-assistant --no-pager --lines=30"
fi

cat <<EOF

Code update deployed.

Project: ${PROJECT}
VM:      ${VM_NAME}
Zone:    ${ZONE}
URL:     https://34-134-9-82.sslip.io/assistant

EOF
