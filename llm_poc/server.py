"""
Born in Bradford — Research Assistant Web Server
=================================================
Serves the BiB Data Dictionary website with an embedded AI chat widget,
plus a full-screen /assistant page.

Usage:
  cd BornInBradford-datadict/llm_poc
  python server.py

  # Custom port:
  python server.py --port 8080

  # Use a different HF model:
  python server.py --model "HuggingFaceH4/zephyr-7b-beta"

Then open: http://localhost:5050
"""

import os
import sys
import json
import argparse
import threading
from pathlib import Path
from typing import Any, Optional

# ── Load .env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
DATADICT_DIR = SCRIPT_DIR.parent          # BornInBradford-datadict/
DOCS_DIR     = DATADICT_DIR / "docs"
STATIC_DIR   = SCRIPT_DIR / "static"

# ── Import RAG engine from bib_research_assistant.py ──────────────────────────
sys.path.insert(0, str(SCRIPT_DIR))
from bib_research_assistant import (
    retrieve_context,
    query as rag_query,
    get_chroma_client,
    _get_hf_client,
    DEFAULT_MODEL,
    _check_index,
)

# ── Flask setup ────────────────────────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify, send_from_directory, Response
except ImportError:
    print("❌ Flask not installed. Run: pip install flask")
    sys.exit(1)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/widget-static")

# ── Global state (initialised once at startup) ─────────────────────────────────
chroma_client: Any = None
llm_client:    Any = None
current_model: str = DEFAULT_MODEL
_init_lock = threading.Lock()


def _ensure_clients():
    global chroma_client, llm_client
    with _init_lock:
        if chroma_client is None:
            chroma_client = get_chroma_client()
        if llm_client is None:
            llm_client = _get_hf_client(current_model)


# ── Chat widget snippet injected before </body> ────────────────────────────────
WIDGET_SNIPPET = """
<!-- BiB Research Assistant Widget -->
<link rel="stylesheet" href="/widget-static/chat-widget.css">
<script src="/widget-static/chat-widget.js"></script>
"""


def inject_widget(html_bytes: bytes) -> bytes:
    """Inject the chat widget into HTML responses."""
    html = html_bytes.decode("utf-8", errors="replace")
    if "</body>" in html and "chat-widget.js" not in html:
        html = html.replace("</body>", WIDGET_SNIPPET + "\n</body>", 1)
    return html.encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
#  API Route — /api/chat
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """
    POST /api/chat
    Body: {"question": "...", "show_context": false}
    Returns: {"answer": "...", "context": "..." (optional)}
    """
    _ensure_clients()

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    show_ctx  = bool(data.get("show_context", False))

    if not question:
        return jsonify({"error": "question is required"}), 400

    if not llm_client:
        return jsonify({"error": "LLM client not available — check HF_TOKEN in .env"}), 503

    if not chroma_client:
        return jsonify({"error": "Vector database not initialised — run --build first"}), 503

    try:
        context = retrieve_context(question, chroma_client)
        answer  = rag_query(
            question, chroma_client, llm_client,
            model=current_model, show_context=False
        )
        result: dict = {"answer": answer}
        if show_ctx:
            result["context"] = context
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  Full-screen assistant page — /assistant
# ══════════════════════════════════════════════════════════════════════════════

ASSISTANT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BiB Research Assistant</title>
<style>
  :root {
    --bib-blue: #1a4e8c;
    --bib-light: #e8f0fb;
    --bib-accent: #2e7d32;
    --radius: 12px;
    --shadow: 0 4px 24px rgba(0,0,0,.12);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f4f6fa;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: var(--bib-blue);
    color: #fff;
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,.2);
  }
  header .logo { font-size: 1.5rem; }
  header h1 { font-size: 1.1rem; font-weight: 600; }
  header .sub { font-size: .78rem; opacity: .8; margin-top: 1px; }
  header a.back-link {
    margin-left: auto;
    color: rgba(255,255,255,.85);
    text-decoration: none;
    font-size: .85rem;
    border: 1px solid rgba(255,255,255,.4);
    padding: 5px 12px;
    border-radius: 20px;
    transition: background .2s;
  }
  header a.back-link:hover { background: rgba(255,255,255,.15); }

  #chat-history {
    flex: 1;
    overflow-y: auto;
    padding: 24px 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .msg {
    max-width: 800px;
    width: 100%;
    padding: 14px 18px;
    border-radius: var(--radius);
    line-height: 1.6;
    font-size: .93rem;
  }
  .msg.user {
    align-self: flex-end;
    background: var(--bib-blue);
    color: #fff;
    border-bottom-right-radius: 4px;
  }
  .msg.assistant {
    align-self: flex-start;
    background: #fff;
    box-shadow: var(--shadow);
    border-bottom-left-radius: 4px;
    white-space: pre-wrap;
  }
  .msg.assistant code {
    background: #f0f4ff;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: .88em;
    font-family: "SF Mono", "Fira Code", monospace;
  }
  .msg.thinking {
    align-self: flex-start;
    background: #fff;
    color: #888;
    box-shadow: var(--shadow);
    border-bottom-left-radius: 4px;
    font-style: italic;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .dot-bounce span {
    display: inline-block;
    width: 7px; height: 7px;
    background: #aaa;
    border-radius: 50%;
    animation: bounce 1.2s infinite;
  }
  .dot-bounce span:nth-child(2) { animation-delay: .2s; }
  .dot-bounce span:nth-child(3) { animation-delay: .4s; }
  @keyframes bounce {
    0%,80%,100% { transform: translateY(0); }
    40% { transform: translateY(-8px); }
  }
  .welcome {
    text-align: center;
    color: #7a8499;
    margin: auto;
    max-width: 540px;
    padding: 40px 20px;
  }
  .welcome .icon { font-size: 3rem; margin-bottom: 12px; }
  .welcome h2 { font-size: 1.2rem; color: var(--bib-blue); margin-bottom: 8px; }
  .welcome p { font-size: .9rem; line-height: 1.6; }
  .suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 20px;
  }
  .suggestion-btn {
    background: var(--bib-light);
    color: var(--bib-blue);
    border: 1px solid #c3d4f0;
    padding: 8px 14px;
    border-radius: 20px;
    font-size: .82rem;
    cursor: pointer;
    transition: background .15s, border-color .15s;
  }
  .suggestion-btn:hover { background: #d0e0f8; border-color: var(--bib-blue); }

  #input-bar {
    background: #fff;
    border-top: 1px solid #e0e4ef;
    padding: 14px 20px;
    display: flex;
    gap: 10px;
    align-items: flex-end;
    box-shadow: 0 -2px 8px rgba(0,0,0,.06);
  }
  #input-bar textarea {
    flex: 1;
    border: 1.5px solid #d0d8e8;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: .93rem;
    resize: none;
    outline: none;
    transition: border-color .2s;
    font-family: inherit;
    max-height: 120px;
    overflow-y: auto;
    line-height: 1.5;
  }
  #input-bar textarea:focus { border-color: var(--bib-blue); }
  #send-btn {
    background: var(--bib-blue);
    color: #fff;
    border: none;
    border-radius: 10px;
    width: 44px; height: 44px;
    font-size: 1.2rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background .2s, transform .1s;
    flex-shrink: 0;
  }
  #send-btn:hover { background: #1560b0; }
  #send-btn:active { transform: scale(.95); }
  #send-btn:disabled { background: #b0bdd6; cursor: not-allowed; }

  .error-msg {
    color: #c62828;
    background: #fff3f3;
    border: 1px solid #f5c6c6;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: .88rem;
  }
  .meta {
    font-size: .75rem;
    color: #aaa;
    margin-top: 6px;
    text-align: right;
  }
</style>
</head>
<body>
<header>
  <span class="logo">🔬</span>
  <div>
    <h1>BiB Research Assistant</h1>
    <div class="sub">Born in Bradford · AI-powered dataset explorer</div>
  </div>
  <a class="back-link" href="/">← Data Dictionary</a>
</header>

<div id="chat-history">
  <div class="welcome" id="welcome-msg">
    <div class="icon">🧬</div>
    <h2>What would you like to explore?</h2>
    <p>Ask about variables, tables, cohort methodology, published papers, or analysis approaches using the Born in Bradford dataset.</p>
    <div class="suggestions">
      <button class="suggestion-btn" onclick="sendSuggestion(this.textContent)">What anxiety variables exist in Age of Wonder?</button>
      <button class="suggestion-btn" onclick="sendSuggestion(this.textContent)">How do I link BiB1000 data to school records?</button>
      <button class="suggestion-btn" onclick="sendSuggestion(this.textContent)">What has been published on childhood obesity?</button>
      <button class="suggestion-btn" onclick="sendSuggestion(this.textContent)">Which tables contain genetic/omics data?</button>
      <button class="suggestion-btn" onclick="sendSuggestion(this.textContent)">What covariates are used in mental health analyses?</button>
      <button class="suggestion-btn" onclick="sendSuggestion(this.textContent)">Describe the BiB_Baseline maternal survey variables</button>
    </div>
  </div>
</div>

<div id="input-bar">
  <textarea id="q-input" rows="1" placeholder="Ask about the BiB dataset, variables, papers, or analysis plans…"
            onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
  <button id="send-btn" onclick="sendMessage()" title="Send">&#9658;</button>
</div>

<script>
const history = document.getElementById('chat-history');
const input   = document.getElementById('q-input');
const sendBtn = document.getElementById('send-btn');
let thinking  = null;

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function sendSuggestion(text) {
  input.value = text;
  sendMessage();
}

function appendMsg(cls, html) {
  const welcome = document.getElementById('welcome-msg');
  if (welcome) welcome.remove();
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.innerHTML = html;
  history.appendChild(div);
  history.scrollTop = history.scrollHeight;
  return div;
}

function showThinking() {
  thinking = appendMsg('thinking',
    'Searching knowledge base… <span class="dot-bounce"><span></span><span></span><span></span></span>');
}

function removeThinking() {
  if (thinking) { thinking.remove(); thinking = null; }
}

function escHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function formatAnswer(text) {
  // Simple markdown-ish: **bold**, `code`, preserve line breaks
  return escHtml(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

async function sendMessage() {
  const q = input.value.trim();
  if (!q) return;

  appendMsg('user', escHtml(q));
  input.value = '';
  autoResize(input);
  sendBtn.disabled = true;
  showThinking();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    });
    const data = await res.json();
    removeThinking();

    if (data.error) {
      appendMsg('assistant error-msg', '⚠ ' + escHtml(data.error));
    } else {
      appendMsg('assistant', formatAnswer(data.answer));
    }
  } catch (err) {
    removeThinking();
    appendMsg('assistant error-msg', '⚠ Network error: ' + escHtml(err.message));
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

input.focus();
</script>
</body>
</html>
"""


@app.route("/assistant")
def assistant_page():
    return Response(ASSISTANT_HTML, mimetype="text/html")


# ══════════════════════════════════════════════════════════════════════════════
#  Static docs — serve with widget injection
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def serve_docs(path):
    """Serve docs/ files; inject chat widget into HTML responses."""
    file_path = DOCS_DIR / path
    if not file_path.exists():
        # Try 404 page
        p404 = DOCS_DIR / "404.html"
        if p404.exists():
            content = inject_widget(p404.read_bytes())
            return Response(content, status=404, mimetype="text/html")
        return "Not found", 404

    # Determine MIME type
    suffix = file_path.suffix.lower()
    mime_map = {
        ".html": "text/html",
        ".css":  "text/css",
        ".js":   "application/javascript",
        ".json": "application/json",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".svg":  "image/svg+xml",
        ".ico":  "image/x-icon",
        ".woff": "font/woff",
        ".woff2":"font/woff2",
        ".ttf":  "font/ttf",
    }
    mime = mime_map.get(suffix, "application/octet-stream")

    content = file_path.read_bytes()
    if suffix == ".html":
        content = inject_widget(content)

    return Response(content, mimetype=mime)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global current_model

    parser = argparse.ArgumentParser(description="BiB Research Assistant Web Server")
    parser.add_argument("--port",  type=int, default=5050, help="Port to listen on (default: 5050)")
    parser.add_argument("--host",  type=str, default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"HuggingFace model (default: {DEFAULT_MODEL})")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    current_model = args.model

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Born in Bradford — Research Assistant Server        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Docs dir  : {DOCS_DIR}")
    print(f"  ChromaDB  : {SCRIPT_DIR / '.chroma_db'}")
    print(f"  LLM model : {current_model}")
    print()

    # Pre-initialise clients
    _ensure_clients()

    # Index health check
    if chroma_client:
        _check_index(chroma_client)

    if not llm_client:
        print("⚠️  Starting without LLM — /api/chat will return 503")
        print("   Set HF_TOKEN in llm_poc/.env to enable chat")

    print(f"\n🌐 Server running at: http://{args.host}:{args.port}")
    print(f"   Data dictionary  : http://{args.host}:{args.port}/")
    print(f"   Full assistant   : http://{args.host}:{args.port}/assistant")
    print(f"   Chat API         : POST http://{args.host}:{args.port}/api/chat")
    print("\n   Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
