#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# BiB Research Assistant — Index Builder
#
# Run this script whenever source data changes:
#   • New data release (CSVs updated)
#   • New papers added to bib_papers_metadata.json
#   • New data dictionary HTML pages added to docs/
#
# Usage:
#   chmod +x build_index.sh   (first time only)
#   ./build_index.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../../.venv/bin/python"

# ── Find Python ───────────────────────────────────────────────────────────────
if [[ -x "$VENV" ]]; then
    PYTHON="$VENV"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "❌ No Python found. Expected venv at: $VENV"
    exit 1
fi

echo "Using Python: $PYTHON"
echo ""

# ── Check required packages ───────────────────────────────────────────────────
if ! "$PYTHON" -c "import chromadb, pandas, openai" 2>/dev/null; then
    echo "📦 Installing required packages..."
    "$PYTHON" -m pip install chromadb openai pandas python-dotenv
    echo ""
fi

# ── Wipe old index (optional — comment out to do incremental updates) ─────────
CHROMA_DIR="$SCRIPT_DIR/.chroma_db"
if [[ -d "$CHROMA_DIR" ]]; then
    echo "🗑  Removing old index at: $CHROMA_DIR"
    rm -rf "$CHROMA_DIR"
fi

# ── Run the build ─────────────────────────────────────────────────────────────
echo "🔨 Building index..."
cd "$SCRIPT_DIR"
"$PYTHON" bib_research_assistant.py --build

echo ""
echo "✅ Done. To query the index run:"
echo "   export OPENAI_API_KEY='sk-...'"
echo "   $PYTHON bib_research_assistant.py --chat"
