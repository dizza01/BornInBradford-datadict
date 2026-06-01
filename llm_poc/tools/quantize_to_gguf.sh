#!/usr/bin/env bash
set -euo pipefail

# Convert the fine-tuned Hugging Face model to GGUF and quantize it with llama.cpp.
#
# Usage:
#   llm_poc/tools/quantize_to_gguf.sh
#   QUANT_TYPE=Q5_K_M llm_poc/tools/quantize_to_gguf.sh
#
# Optional environment variables:
#   MODEL_ID        Hugging Face repo id to download
#   QUANT_TYPE      llama.cpp quantization type, e.g. Q4_K_M or Q5_K_M
#   MODELS_DIR      Output/cache directory for HF and GGUF files
#   LLAMA_CPP_DIR   Existing llama.cpp checkout to reuse
#   F16_GGUF        Intermediate F16 GGUF path
#   OUT_GGUF        Final quantized GGUF path

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_POC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${LLM_POC_DIR}/.." && pwd)"

MODEL_ID="${MODEL_ID:-dizza01/llama-3.1-8b-bib-grounded-sft-merged}"
MODEL_SLUG="${MODEL_ID##*/}"
QUANT_TYPE="${QUANT_TYPE:-Q4_K_M}"
MODELS_DIR="${MODELS_DIR:-${REPO_DIR}/models}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${REPO_DIR}/.external/llama.cpp}"
HF_MODEL_DIR="${HF_MODEL_DIR:-${MODELS_DIR}/${MODEL_SLUG}}"
F16_GGUF="${F16_GGUF:-${MODELS_DIR}/bib-llama-3.1-8b.f16.gguf}"
OUT_GGUF="${OUT_GGUF:-${MODELS_DIR}/bib-llama-3.1-8b.${QUANT_TYPE}.gguf}"

mkdir -p "${MODELS_DIR}" "$(dirname "${LLAMA_CPP_DIR}")"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

find_quantize_bin() {
  for candidate in \
    "${LLAMA_CPP_DIR}/build/bin/llama-quantize" \
    "${LLAMA_CPP_DIR}/build/bin/Release/llama-quantize" \
    "${LLAMA_CPP_DIR}/quantize"; do
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

require_cmd git
require_cmd cmake
require_cmd python3
HF_CLI=""
if command -v hf >/dev/null 2>&1; then
  HF_CLI="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI="huggingface-cli"
else
  echo "Missing required command: hf" >&2
  echo "Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

if [[ ! -d "${LLAMA_CPP_DIR}/.git" ]]; then
  echo "Cloning llama.cpp into ${LLAMA_CPP_DIR}"
  git clone https://github.com/ggml-org/llama.cpp "${LLAMA_CPP_DIR}"
fi

echo "Building llama.cpp"
cmake -S "${LLAMA_CPP_DIR}" -B "${LLAMA_CPP_DIR}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${LLAMA_CPP_DIR}/build" --config Release -j

QUANTIZE_BIN="$(find_quantize_bin)"

echo "Installing llama.cpp conversion Python requirements"
python3 -m pip install -r "${LLAMA_CPP_DIR}/requirements.txt"

echo "Downloading Hugging Face model: ${MODEL_ID}"
"${HF_CLI}" download "${MODEL_ID}" --local-dir "${HF_MODEL_DIR}"

echo "Converting HF model to F16 GGUF: ${F16_GGUF}"
python3 "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
  "${HF_MODEL_DIR}" \
  --outfile "${F16_GGUF}" \
  --outtype f16

echo "Quantizing ${F16_GGUF} -> ${OUT_GGUF} (${QUANT_TYPE})"
"${QUANTIZE_BIN}" "${F16_GGUF}" "${OUT_GGUF}" "${QUANT_TYPE}"

cat <<EOF

Done.

Final GGUF:
  ${OUT_GGUF}

Run the assistant with:
  cd ${REPO_DIR}
  ../.venv/bin/python llm_poc/server.py --llm-backend llama_cpp --gguf-model-path "${OUT_GGUF}"

EOF
