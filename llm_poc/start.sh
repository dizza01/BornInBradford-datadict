#!/usr/bin/env bash
# ── BiB Research Assistant — Start Server ─────────────────────────────────────
# Run from anywhere: bash BornInBradford-datadict/llm_poc/start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../../.venv"

# Find Python — prefer the project venv
if [[ -x "$VENV/bin/python" ]]; then
    PY="$VENV/bin/python"
elif command -v python3 &>/dev/null; then
    PY="python3"
else
    echo "❌ Python not found. Activate your venv or install Python 3."
    exit 1
fi

echo "Using Python: $PY"

# Ensure Flask is installed
$PY -c "import flask" 2>/dev/null || {
    echo "Installing Flask..."
    $PY -m pip install flask -q
}

cd "$SCRIPT_DIR"

# Check index exists
if [[ ! -f ".chroma_db/chroma.sqlite3" ]]; then
    echo ""
    echo "⚠️  No index found. Building now (takes ~2-5 mins on first run)..."
    echo ""
    $PY bib_research_assistant.py --build
fi

echo ""
echo "Starting server..."
echo ""
exec $PY server.py "$@"
