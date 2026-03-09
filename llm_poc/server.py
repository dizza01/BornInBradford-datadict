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
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

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
    query_stream as rag_query_stream,
    get_chroma_client,
  parse_html_sections,
    _get_hf_client,
    DEFAULT_MODEL,
    _check_index,
    _strip_filler,
)

# ── Flask setup ────────────────────────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
except ImportError:
    print("❌ Flask not installed. Run: pip install flask")
    sys.exit(1)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/widget-static")

# ── Global state (initialised once at startup) ─────────────────────────────────
chroma_client: Any = None
llm_client:    Any = None
current_model: str = DEFAULT_MODEL
_init_lock = threading.Lock()
_registry_lock = threading.Lock()
_registry_cache: Optional[dict[str, Any]] = None


def _ensure_clients():
    global chroma_client, llm_client
    with _init_lock:
        if chroma_client is None:
            chroma_client = get_chroma_client()
        if llm_client is None:
            llm_client = _get_hf_client(current_model)


def _clean_value(val: Any) -> str:
  if val is None:
    return ""
  try:
    if pd.isna(val):
      return ""
  except Exception:
    pass
  return str(val).strip()


_THEME_PATTERNS = [
  ("Mental Health", re.compile(r"\b(anxiety|anxious|depress|depression|epds|rcads|sdq|stress|wellbeing|well-being|mental|mood|emotion|psych)\b", re.I)),
  ("Pregnancy & Recruitment", re.compile(r"\b(pregnan|antenatal|booking|recruit|gestat|parity|labou?r|delivery|birth|maternal|postnatal|baby)\b", re.I)),
  ("Anthropometry & Growth", re.compile(r"\b(bmi|height|weight|waist|hip|skinfold|bioimpedance|anthrop|growth|adiposity|body mass)\b", re.I)),
  ("Lifestyle", re.compile(r"\b(smok|alcohol|drink|diet|nutrition|food|sleep|exercise|physical activity|activity level|breastfeed)\b", re.I)),
  ("Geography & Environment", re.compile(r"\b(postcode|lsoa|imd|geograph|neighbourhood|pollution|green space|environment|address|ward)\b", re.I)),
  ("Education & Development", re.compile(r"\b(school|educat|eyfs|ks1|ks2|ks4|gcse|language|speech|learning|cognitive|development)\b", re.I)),
  ("Biosamples & Omics", re.compile(r"\b(genet|genomic|dna|rna|methyl|metabol|proteom|omics|biosample|serum|plasma|sample)\b", re.I)),
  ("Health Records & Services", re.compile(r"\b(gp|hospital|admission|prescription|diagnos|clinic|record linkage|episode|nhs)\b", re.I)),
  ("Demographics & Family", re.compile(r"\b(ethnic|demograph|family|partner|mother|father|parent|child|household|country of birth|participant type)\b", re.I)),
  ("Administration & Identifiers", re.compile(r"\b(identifier|consent|admin|administration|participant id|personid|bibpersonid|legacy id|audit)\b", re.I)),
]

_TOPIC_THEME_MAP = {
  "administration": "Administration & Identifiers",
  "mental health": "Mental Health",
  "physical health": "Anthropometry & Growth",
  "nutrition": "Lifestyle",
  "lifestyle": "Lifestyle",
  "geographic": "Geography & Environment",
  "socioeconomic": "Demographics & Family",
  "cohort info": "Demographics & Family",
  "education": "Education & Development",
  "development": "Education & Development",
  "omics": "Biosamples & Omics",
}


def _titleise(text: str) -> str:
  clean = re.sub(r"[_/]+", " ", text).strip()
  clean = re.sub(r"\s+", " ", clean)
  return clean.title()


def _guess_html_stems(table_id: str, project: str, table_name: str) -> list[str]:
  stems: list[str] = []
  if table_id:
    stems.append(table_id.lower().replace(".", "_"))
  if project and table_name:
    proj = project.lower()
    if proj.startswith("bib_"):
      proj = proj[4:]
    proj = proj.replace("_", "")
    stems.append(f"bib_{proj}_{table_name}".lower())
  # preserve order but deduplicate
  seen = set()
  ordered = []
  for stem in stems:
    if stem not in seen:
      ordered.append(stem)
      seen.add(stem)
  return ordered


def _derive_theme(topic: str, section: str, label: str, description: str,
          table_id: str, table_display: str, project: str) -> str:
  for source in (topic, section):
    key = _clean_value(source).lower()
    if key in _TOPIC_THEME_MAP:
      return _TOPIC_THEME_MAP[key]

  blob = " ".join([
    _clean_value(topic), _clean_value(section), _clean_value(label),
    _clean_value(description), _clean_value(table_id),
    _clean_value(table_display), _clean_value(project),
  ])
  for theme, pattern in _THEME_PATTERNS:
    if pattern.search(blob):
      return theme

  if topic:
    return _titleise(topic)
  if section:
    return _titleise(section)
  return "Other"


def _derive_study_context(project: str, table_id: str, table_name: str,
              table_display: str, source_html: str) -> str:
  """Infer a human-readable study / wave label from metadata naming patterns."""
  project_l = _clean_value(project).lower()
  table_id_l = _clean_value(table_id).lower()
  table_name_l = _clean_value(table_name).lower()
  table_display_l = _clean_value(table_display).lower()
  source_html_l = _clean_value(source_html).lower()
  blob = " ".join([project_l, table_id_l, table_name_l, table_display_l, source_html_l])

  wave_match = re.search(r"(?:^|_)(6m|12m|18m|24m|36m)(?:_|$)", blob)
  wave = wave_match.group(1) if wave_match else ""

  if "bib_1000" in blob or "bib1000" in blob:
    return f"BiB 1000 ({wave})" if wave else "BiB 1000"
  if "ageofwonder" in blob:
    return "Age of Wonder"
  if "growingup" in blob:
    return "Growing Up"
  if "bibbs" in blob:
    return "BiBBS"
  if "startingschool" in blob:
    return "Starting School"
  if "primaryschoolyears" in blob or "primary_school" in blob:
    return "Primary School Years"
  if "baseline" in blob:
    return "Baseline"
  if "medall" in blob:
    return "MeDALL"
  if "all_in" in blob or "allin" in blob:
    return f"ALL IN ({wave})" if wave else "ALL IN"
  if "breathes" in blob:
    return "BREATHES"
  if "covid" in blob:
    return "COVID-19 Surveys"
  if "cohortinfo" in blob:
    return "Core Cohort"
  if "geographic" in blob:
    return "Geographic Linkage"
  if "biosamples" in blob or "biobank" in blob:
    return "Biosamples & Biobank"
  if "metabolomics" in blob:
    return "Metabolomics"
  if "proteomics" in blob:
    return "Proteomics"
  if "glycomics" in blob:
    return "Glycomics"
  if "genotyp" in blob or "methyl" in blob or "exome" in blob:
    return "Genetics & Omics"
  if "pregnancy" in blob or "maternity" in blob or "ultrasound" in blob:
    return "Pregnancy & Birth"
  if project:
    return _titleise(project.replace("BiB_", "").replace("BiB", "BiB "))
  return "Study not inferred"


def _build_variable_registry() -> dict[str, Any]:
  html_sections = parse_html_sections()
  vars_df = pd.read_csv(DOCS_DIR / "csv" / "all_variables_meta.csv")
  tables_df = pd.read_csv(DOCS_DIR / "csv" / "all_tables.csv")
  table_lookup = {
    _clean_value(row.get("table_id", "")): {
      "display_name": _clean_value(row.get("display_name", "")),
      "project_name": _clean_value(row.get("project_name", "")),
      "entity_type": _clean_value(row.get("entity_type", "")),
      "n_rows": _clean_value(row.get("n_rows", "")),
    }
    for _, row in tables_df.iterrows()
  }

  rows: list[dict[str, Any]] = []
  theme_counts: dict[str, int] = {}

  for _, row in vars_df.iterrows():
    table_id = _clean_value(row.get("table_id", ""))
    project = _clean_value(row.get("project", ""))
    table_name = _clean_value(row.get("table", ""))
    variable = _clean_value(row.get("variable", ""))
    label = _clean_value(row.get("label", ""))
    topic = _clean_value(row.get("topic", ""))
    value_type = _clean_value(row.get("value_type", ""))
    n_complete = _clean_value(row.get("n_complete", ""))
    n_entities_complete = _clean_value(row.get("n_entities_complete", ""))

    html_info: dict[str, str] = {}
    source_html = ""
    for stem in _guess_html_stems(table_id, project, table_name):
      if stem in html_sections:
        source_html = f"{stem}.html"
        html_info = html_sections.get(stem, {}).get(variable, {}) or {}
        if html_info:
          break

    table_meta = table_lookup.get(table_id, {})
    description = _clean_value(html_info.get("description", ""))
    section = _clean_value(html_info.get("section", ""))
    theme = _derive_theme(
      topic, section, label, description,
      table_id, table_meta.get("display_name", ""), project,
    )
    study_context = _derive_study_context(
      project, table_id, table_name, table_meta.get("display_name", ""), source_html,
    )

    entry = {
      "table": table_id,
      "table_name": table_name,
      "table_display": table_meta.get("display_name", ""),
      "project": project,
      "study_context": study_context,
      "variable": variable,
      "label": label,
      "description": description,
      "section": section,
      "topic": topic,
      "theme": theme,
      "type": value_type,
      "non_missing": n_complete,
      "entities_complete": n_entities_complete,
      "entity_type": table_meta.get("entity_type", ""),
      "source_html": source_html,
    }
    rows.append(entry)
    theme_counts[theme] = theme_counts.get(theme, 0) + 1

  rows.sort(key=lambda r: (r["theme"], r["table"], r["variable"]))
  themes = [{"name": name, "count": count} for name, count in sorted(
    theme_counts.items(), key=lambda item: (-item[1], item[0])
  )]

  return {
    "summary": {
      "variables": len(rows),
      "themes": len(themes),
      "tables": int(len(tables_df)),
      "html_files": int(len(html_sections)),
    },
    "themes": themes,
    "rows": rows,
  }


def _get_variable_registry() -> dict[str, Any]:
  global _registry_cache
  with _registry_lock:
    if _registry_cache is None:
      _registry_cache = _build_variable_registry()
    return _registry_cache


def _registry_score(row: dict[str, Any], query: str) -> int:
  q = query.lower().strip()
  if not q:
    return 0
  score = 0
  variable = row.get("variable", "").lower()
  label = row.get("label", "").lower()
  theme = row.get("theme", "").lower()
  table = row.get("table", "").lower()
  description = row.get("description", "").lower()
  if variable == q:
    score += 120
  if variable.startswith(q):
    score += 60
  if q in variable:
    score += 35
  if q in label:
    score += 25
  if q in theme:
    score += 15
  if q in table:
    score += 10
  if q in description:
    score += 8
  try:
    score += min(int(row.get("non_missing") or 0) // 1000, 15)
  except Exception:
    pass
  return score


# ── Chat widget snippet injected before </body> ────────────────────────────────
WIDGET_SNIPPET = """
<!-- BiB Research Assistant Widget + Nav links -->
<link rel="stylesheet" href="/widget-static/chat-widget.css">

<!-- Top-right nav pills -->
<style>
  #bib-nav-group {
    position: fixed;
    top: 14px;
    right: 18px;
    z-index: 9997;
    display: flex;
    gap: 10px;
    align-items: center;
  }
  #bib-assistant-nav,
  #bib-registry-nav {
    display: flex;
    align-items: center;
    gap: 7px;
    color: #fff;
    text-decoration: none;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: .8rem;
    font-weight: 600;
    padding: 7px 14px 7px 11px;
    border-radius: 24px;
    box-shadow: 0 3px 14px rgba(26,78,140,.38);
    transition: background .18s, box-shadow .18s, transform .12s;
    letter-spacing: .01em;
    white-space: nowrap;
  }
  #bib-assistant-nav {
    background: #1a4e8c;
    box-shadow: 0 3px 14px rgba(26,78,140,.38);
  }
  #bib-registry-nav {
    background: #375a7f;
    box-shadow: 0 3px 14px rgba(55,90,127,.32);
  }
  #bib-assistant-nav:hover {
    background: #1560b0;
    box-shadow: 0 5px 20px rgba(26,78,140,.52);
    transform: translateY(-1px);
  }
  #bib-registry-nav:hover {
    background: #466e98;
    box-shadow: 0 5px 20px rgba(55,90,127,.45);
    transform: translateY(-1px);
  }
  #bib-assistant-nav:active,
  #bib-registry-nav:active { transform: translateY(0); }
  #bib-assistant-nav .bib-nav-icon,
  #bib-registry-nav .bib-nav-icon { font-size: .95rem; }
  @media (max-width: 600px) {
    #bib-assistant-nav span.bib-nav-label,
    #bib-registry-nav span.bib-nav-label { display: none; }
    #bib-assistant-nav,
    #bib-registry-nav { padding: 8px 12px; }
    #bib-nav-group { gap: 8px; }
  }
</style>
<div id="bib-nav-group">
<a id="bib-assistant-nav" href="/assistant" title="Open BiB Research Assistant">
  <span class="bib-nav-icon">🔬</span>
  <span class="bib-nav-label">Research Assistant</span>
  <span>↗</span>
</a>
<a id="bib-registry-nav" href="/registry" title="Open canonical variable registry">
  <span class="bib-nav-icon">🗂</span>
  <span class="bib-nav-label">Variable Registry</span>
  <span>↗</span>
</a>
</div>

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
    # history: [{"role": "user"|"assistant", "content": "..."}] — prior turns
    history   = data.get("history") or []

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
            model=current_model, show_context=False,
            history=history,
        )
        result: dict = {"answer": answer}
        if show_ctx:
            result["context"] = context
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
#  Streaming API — /api/chat/stream  (Server-Sent Events)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/chat/stream", methods=["POST"])
def chat_stream_endpoint():
    """
    POST /api/chat/stream
    Body: {"question": "...", "history": [...]}
    Returns: text/event-stream with events:
      data: {"token": "..."}    — each generated token
      data: {"replace": "..."}  — footer was stripped; replace full message
      data: {"error": "..."}    — error occurred
      data: {"done": true}      — generation complete
    """
    _ensure_clients()

    data     = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    history  = data.get("history") or []

    if not question:
        return jsonify({"error": "question is required"}), 400
    if not llm_client:
        return jsonify({"error": "LLM client not available — check HF_TOKEN in .env"}), 503
    if not chroma_client:
        return jsonify({"error": "Vector database not initialised — run --build first"}), 503

    def generate():
        full_text = ""
        try:
            for token in rag_query_stream(
                question, chroma_client, llm_client,
                model=current_model, history=history,
            ):
                full_text += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            # Strip any footer boilerplate accumulated in the full response
            cleaned = _strip_filler(full_text)
            if cleaned != full_text:
                yield f"data: {json.dumps({'replace': cleaned})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        yield 'data: {"done": true}\n\n'

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
  header .nav-links {
    margin-left: auto;
    display: flex;
    gap: 10px;
    align-items: center;
  }
  header a.back-link {
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
  .md-table {
    border-collapse: collapse;
    width: 100%;
    font-size: .88rem;
    margin: 8px 0;
    overflow-x: auto;
    display: block;
  }
  .md-table th, .md-table td {
    border: 1px solid #d0d8e8;
    padding: 6px 11px;
    text-align: left;
    white-space: nowrap;
  }
  .md-table th {
    background: #e8f0fb;
    font-weight: 600;
    color: #1a4e8c;
  }
  .md-table tr:nth-child(even) td { background: #f7f9fe; }
  .md-h { margin: 10px 0 4px; line-height: 1.3; color: #14397a; }
  h3.md-h { font-size: 1rem; }
  h4.md-h { font-size: .93rem; }
  h5.md-h { font-size: .88rem; }
</style>
</head>
<body>
<header>
  <span class="logo">🔬</span>
  <div>
    <h1>BiB Research Assistant</h1>
    <div class="sub">Born in Bradford · AI-powered dataset explorer</div>
  </div>
  <div class="nav-links">
    <a class="back-link" href="/registry">🗂 Variable Registry</a>
    <a class="back-link" href="/">← Data Dictionary</a>
  </div>
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
const chatLog  = document.getElementById('chat-history');
const input    = document.getElementById('q-input');
const sendBtn  = document.getElementById('send-btn');
let thinking   = null;
const convHistory = [];  // tracks turns for multi-turn context

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}
function sendSuggestion(text) { input.value = text; sendMessage(); }

function appendMsg(cls, html) {
  const welcome = document.getElementById('welcome-msg');
  if (welcome) welcome.remove();
  const div = document.createElement('div');
  div.className = 'msg ' + cls;
  div.innerHTML = html;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}
function showThinking() {
  thinking = appendMsg('thinking',
    'Searching knowledge base… <span class="dot-bounce"><span></span><span></span><span></span></span>');
}
function removeThinking() { if (thinking) { thinking.remove(); thinking = null; } }

function escHtml(t) {
  return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function renderMdTable(lines) {
  const rows = lines.filter(l => !l.trim().match(/^\|[-: |]+\|$/));
  if (!rows.length) return '';
  let html = '<table class="md-table">';
  const headers = rows[0].trim().replace(/^\||\|$/g,'').split('|').map(c => c.trim());
  html += '<thead><tr>' + headers.map(h => `<th>${escHtml(h)}</th>`).join('') + '</tr></thead>';
  if (rows.length > 1) {
    html += '<tbody>';
    for (let i = 1; i < rows.length; i++) {
      const cells = rows[i].trim().replace(/^\||\|$/g,'').split('|').map(c => c.trim());
      html += '<tr>' + cells.map(c => `<td>${escHtml(c)}</td>`).join('') + '</tr>';
    }
    html += '</tbody>';
  }
  return html + '</table>';
}
function formatInline(l) {
  return escHtml(l)
    .replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')
    .replace(/`([^`]+)`/g,'<code>$1</code>');
}
function formatLine(l) { return formatInline(l); }
function formatAnswer(text) {
  const lines = text.split('\n');
  const segments = [];
  let textBuf = [];
  let i = 0;
  const flush = () => { if (textBuf.length) { segments.push({t:'text',l:textBuf.slice()}); textBuf=[]; } };
  while (i < lines.length) {
    const ln = lines[i];
    const hm = ln.match(/^(#{1,3}) (.+)/);
    if (hm) {
      flush();
      const tag = ['h3','h4','h5'][hm[1].length - 1];
      segments.push({t:'heading', tag, text: formatInline(hm[2])});
      i++;
    } else if (ln.trim().startsWith('|') && i+1 < lines.length && lines[i+1].trim().match(/^\|[-: |]+\|$/)) {
      flush();
      const tbl = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) { tbl.push(lines[i++]); }
      segments.push({t:'table',l:tbl});
    } else { textBuf.push(ln); i++; }
  }
  flush();
  return segments.map(s => {
    if (s.t === 'table') return renderMdTable(s.l);
    if (s.t === 'heading') return `<${s.tag} class="md-h">${s.text}</${s.tag}>`;
    return s.l.map(formatLine).join('<br>');
  }).join('');
}

async function sendMessage() {
  const q = input.value.trim();
  if (!q) return;

  appendMsg('user', escHtml(q));
  convHistory.push({ role: 'user', content: q });
  input.value = '';
  autoResize(input);
  sendBtn.disabled = true;
  showThinking();

  let msgEl    = null;
  let fullText = '';

  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, history: convHistory.slice(0, -1) }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Server error ' + res.status }));
      removeThinking();
      appendMsg('assistant error-msg', '⚠ ' + escHtml(err.error || 'Unknown error'));
      return;
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();  // keep the incomplete trailing line
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let evt;
        try { evt = JSON.parse(line.slice(6)); } catch { continue; }

        if (evt.token) {
          if (!msgEl) { removeThinking(); msgEl = appendMsg('assistant', ''); }
          fullText += evt.token;
          msgEl.textContent = fullText;  // plain text while streaming
          chatLog.scrollTop = chatLog.scrollHeight;
        }
        if (evt.replace) {
          fullText = evt.replace;
          if (msgEl) msgEl.innerHTML = formatAnswer(fullText);
          chatLog.scrollTop = chatLog.scrollHeight;
        }
        if (evt.error) {
          removeThinking();
          appendMsg('assistant error-msg', '⚠ ' + escHtml(evt.error));
        }
        if (evt.done) {
          if (msgEl) msgEl.innerHTML = formatAnswer(fullText);
          convHistory.push({ role: 'assistant', content: fullText });
          chatLog.scrollTop = chatLog.scrollHeight;
        }
      }
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


REGISTRY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BiB Canonical Variable Registry</title>
<style>
  :root {
    --bib-blue: #1a4e8c;
    --bib-blue-2: #375a7f;
    --bib-light: #e8f0fb;
    --bib-bg: #f4f6fa;
    --card: #ffffff;
    --text: #243041;
    --muted: #687487;
    --border: #d7deea;
    --shadow: 0 4px 24px rgba(0,0,0,.08);
    --radius: 14px;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bib-bg);
    color: var(--text);
  }
  header {
    background: linear-gradient(135deg, var(--bib-blue), var(--bib-blue-2));
    color: #fff;
    padding: 18px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 2px 10px rgba(0,0,0,.14);
  }
  header .logo { font-size: 1.55rem; }
  header h1 { margin: 0; font-size: 1.15rem; }
  header .sub { font-size: .82rem; opacity: .82; margin-top: 2px; }
  header .nav-links {
    margin-left: auto;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }
  header .nav-links a {
    color: rgba(255,255,255,.92);
    text-decoration: none;
    border: 1px solid rgba(255,255,255,.35);
    border-radius: 999px;
    padding: 7px 12px;
    font-size: .85rem;
  }
  header .nav-links a:hover { background: rgba(255,255,255,.12); }
  .page {
    max-width: 1480px;
    margin: 0 auto;
    padding: 22px;
  }
  .hero {
    display: grid;
    grid-template-columns: 1fr;
    gap: 18px;
    margin-bottom: 18px;
  }
  .hero-card, .stat-card, .step-card, .panel, .detail-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
  }
  .hero-card {
    padding: 22px 24px;
  }
  .hero-card h2 { margin: 0 0 10px; font-size: 1.35rem; color: var(--bib-blue); }
  .hero-card p { margin: 0; line-height: 1.65; color: #314154; }
  .stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 18px;
  }
  .stat-card { padding: 16px 18px; }
  .stat-card .k { font-size: .76rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
  .stat-card .v { margin-top: 8px; font-size: 1.4rem; font-weight: 700; color: var(--bib-blue); }
  .timeline-panel {
    margin-bottom: 18px;
    padding: 0;
    overflow: hidden;
  }
  .timeline-panel .timeline-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    padding: 16px 18px 12px;
    border-bottom: 1px solid var(--border);
    background: #fbfcff;
  }
  .timeline-panel .timeline-head h3 {
    margin: 0 0 6px;
    font-size: 1rem;
    color: var(--bib-blue);
  }
  .timeline-panel .timeline-head p {
    margin: 0;
    color: #516176;
    line-height: 1.55;
    font-size: .9rem;
    max-width: 760px;
  }
  .timeline-panel .timeline-range {
    color: var(--muted);
    font-size: .82rem;
    white-space: nowrap;
  }
  .timeline-grid {
    padding: 16px 18px 18px;
    display: grid;
    gap: 16px;
  }
  .timeline-section {
    display: grid;
    grid-template-columns: 150px 1fr;
    gap: 14px;
    align-items: start;
  }
  .timeline-section-name {
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .05em;
    text-transform: uppercase;
    color: var(--bib-blue);
    padding-top: 8px;
  }
  .timeline-items {
    display: grid;
    gap: 10px;
  }
  .timeline-item {
    display: grid;
    grid-template-columns: 180px minmax(0, 1fr) 92px;
    gap: 12px;
    align-items: center;
  }
  .timeline-item-label {
    font-size: .89rem;
    color: #314154;
  }
  .timeline-track {
    position: relative;
    height: 14px;
    border-radius: 999px;
    background: #edf2fa;
    overflow: hidden;
  }
  .timeline-bar {
    position: absolute;
    top: 0;
    bottom: 0;
    border-radius: 999px;
    background: linear-gradient(90deg, #6f97d0, #1a4e8c);
  }
  .timeline-item.pregnancy .timeline-bar { background: linear-gradient(90deg, #7d9bd0, #3c6fb4); }
  .timeline-item.early .timeline-bar { background: linear-gradient(90deg, #5aa8c8, #2b7a9b); }
  .timeline-item.school .timeline-bar { background: linear-gradient(90deg, #59a36f, #2e7d32); }
  .timeline-item.later .timeline-bar { background: linear-gradient(90deg, #d08a55, #b8661d); }
  .timeline-date {
    font-size: .82rem;
    color: var(--muted);
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .controls {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 14px;
  }
  .controls input {
    flex: 1;
    border: 1px solid var(--border);
    background: #fff;
    border-radius: 12px;
    padding: 12px 14px;
    font-size: .94rem;
    outline: none;
  }
  .controls input:focus { border-color: var(--bib-blue); }
  .controls .meta {
    color: var(--muted);
    font-size: .85rem;
    white-space: nowrap;
  }
  .registry-layout {
    display: grid;
    grid-template-columns: 270px minmax(0, 1fr) 360px;
    gap: 16px;
    align-items: start;
  }
  .panel { overflow: hidden; }
  .panel h3, .detail-card h3 {
    margin: 0;
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    font-size: .97rem;
    color: var(--bib-blue);
    background: #fbfcff;
  }
  .theme-list {
    max-height: 68vh;
    overflow: auto;
    padding: 8px;
  }
  .theme-btn {
    width: 100%;
    border: 0;
    background: transparent;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-radius: 10px;
    padding: 10px 11px;
    cursor: pointer;
    color: var(--text);
    text-align: left;
  }
  .theme-btn:hover { background: #f2f6fd; }
  .theme-btn.active {
    background: var(--bib-light);
    color: var(--bib-blue);
    font-weight: 600;
  }
  .theme-count {
    color: var(--muted);
    font-size: .82rem;
    font-variant-numeric: tabular-nums;
  }
  .results-wrap { overflow: hidden; }
  .results-table-wrap {
    overflow: auto;
    max-height: 68vh;
  }
  table.registry-table {
    width: 100%;
    border-collapse: collapse;
    font-size: .89rem;
  }
  .registry-table th,
  .registry-table td {
    padding: 10px 12px;
    border-bottom: 1px solid #e8edf5;
    text-align: left;
    vertical-align: top;
  }
  .registry-table th {
    position: sticky;
    top: 0;
    background: #fbfcff;
    color: var(--bib-blue);
    font-size: .82rem;
    text-transform: uppercase;
    letter-spacing: .04em;
  }
  .registry-table tr { cursor: pointer; }
  .registry-table tr:hover { background: #f7faff; }
  .registry-table tr.active { background: #eef5ff; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .pill {
    display: inline-block;
    background: #eef4fc;
    color: var(--bib-blue);
    border: 1px solid #d0ddf2;
    border-radius: 999px;
    padding: 4px 9px;
    font-size: .76rem;
    white-space: nowrap;
  }
  .study-chip {
    background: #f4f7fb;
    color: #375a7f;
    border-color: #d6e0ec;
  }
  .detail-body { padding: 16px; }
  .detail-card dl {
    display: grid;
    grid-template-columns: 118px 1fr;
    gap: 10px 12px;
    margin: 0;
    font-size: .9rem;
  }
  .detail-card dt {
    color: var(--muted);
    font-weight: 600;
  }
  .detail-card dd { margin: 0; line-height: 1.6; }
  .detail-empty { color: var(--muted); line-height: 1.7; }
  .source-link {
    color: var(--bib-blue);
    text-decoration: none;
  }
  .source-link:hover { text-decoration: underline; }
  .empty-state {
    padding: 26px;
    color: var(--muted);
    text-align: center;
  }
  @media (max-width: 1200px) {
    .registry-layout { grid-template-columns: 250px minmax(0, 1fr); }
    .detail-card { grid-column: 1 / -1; }
  }
  @media (max-width: 900px) {
    .hero, .stats, .registry-layout { grid-template-columns: 1fr; }
    .timeline-section { grid-template-columns: 1fr; }
    .timeline-item { grid-template-columns: 1fr; gap: 6px; }
    .timeline-date { text-align: left; }
    .controls { flex-direction: column; align-items: stretch; }
    .controls .meta { white-space: normal; }
    .results-table-wrap, .theme-list { max-height: none; }
  }
</style>
</head>
<body>
<header>
  <span class="logo">🗂</span>
  <div>
    <h1>BiB Canonical Variable Registry</h1>
    <div class="sub">Grounded only in real HTML + CSV metadata</div>
  </div>
  <div class="nav-links">
    <a href="/assistant">🔬 Research Assistant</a>
    <a href="/">← Data Dictionary</a>
  </div>
</header>

<div class="page">
  <section class="hero">
    <div class="hero-card">
      <h2>Real variables first</h2>
      <p>
        This page implements the new canonical metadata layer for the BiB assistant. It uses only
        real data dictionary sources — the HTML files in <span class="mono">docs/</span> plus the CSV metadata tables — to build a finite, auditable registry of valid variables. Themes are assigned from metadata signals such as topic, section headings, labels, and table context.
      </p>
    </div>
  </section>

  <section class="stats">
    <div class="stat-card"><div class="k">Variables</div><div class="v" id="stat-vars">—</div></div>
    <div class="stat-card"><div class="k">Themes</div><div class="v" id="stat-themes">—</div></div>
    <div class="stat-card"><div class="k">Tables</div><div class="v" id="stat-tables">—</div></div>
    <div class="stat-card"><div class="k">HTML Sources</div><div class="v" id="stat-html">—</div></div>
  </section>

  <section class="panel timeline-panel">
    <div class="timeline-head">
      <div>
        <h3>Longitudinal study design</h3>
        <p>A simplified view of the main BiB cohort phases, from recruitment and baseline measures through early childhood follow-up, school-age linkage, and later sub-studies such as Growing Up and Age of Wonder.</p>
      </div>
      <div class="timeline-range">2007 → 2025</div>
    </div>
    <div class="timeline-grid">
      <div class="timeline-section">
        <div class="timeline-section-name">Pregnancy</div>
        <div class="timeline-items">
          <div class="timeline-item pregnancy"><div class="timeline-item-label">Recruitment</div><div class="timeline-track"><div class="timeline-bar" style="left:0%;width:22%;"></div></div><div class="timeline-date">2007–2011</div></div>
          <div class="timeline-item pregnancy"><div class="timeline-item-label">Baseline Survey</div><div class="timeline-track"><div class="timeline-bar" style="left:0%;width:22%;"></div></div><div class="timeline-date">2007–2011</div></div>
          <div class="timeline-item pregnancy"><div class="timeline-item-label">Maternity Records</div><div class="timeline-track"><div class="timeline-bar" style="left:0%;width:27%;"></div></div><div class="timeline-date">2007–2011</div></div>
        </div>
      </div>

      <div class="timeline-section">
        <div class="timeline-section-name">Early childhood</div>
        <div class="timeline-items">
          <div class="timeline-item early"><div class="timeline-item-label">Birth</div><div class="timeline-track"><div class="timeline-bar" style="left:4%;width:23%;"></div></div><div class="timeline-date">2007–2011</div></div>
          <div class="timeline-item early"><div class="timeline-item-label">BiB 1000 – 6m</div><div class="timeline-track"><div class="timeline-bar" style="left:7%;width:18%;"></div></div><div class="timeline-date">2008–2011</div></div>
          <div class="timeline-item early"><div class="timeline-item-label">BiB 1000 – 12m</div><div class="timeline-track"><div class="timeline-bar" style="left:10%;width:19%;"></div></div><div class="timeline-date">2008–2012</div></div>
          <div class="timeline-item early"><div class="timeline-item-label">BiB 1000 – 18m</div><div class="timeline-track"><div class="timeline-bar" style="left:13%;width:20%;"></div></div><div class="timeline-date">2009–2012</div></div>
          <div class="timeline-item early"><div class="timeline-item-label">BiB 1000 – 24m</div><div class="timeline-track"><div class="timeline-bar" style="left:16%;width:21%;"></div></div><div class="timeline-date">2009–2013</div></div>
          <div class="timeline-item early"><div class="timeline-item-label">BiB 1000 – 36m</div><div class="timeline-track"><div class="timeline-bar" style="left:22%;width:22%;"></div></div><div class="timeline-date">2010–2014</div></div>
        </div>
      </div>

      <div class="timeline-section">
        <div class="timeline-section-name">School age</div>
        <div class="timeline-items">
          <div class="timeline-item school"><div class="timeline-item-label">Starting School</div><div class="timeline-track"><div class="timeline-bar" style="left:27%;width:24%;"></div></div><div class="timeline-date">2011–2015</div></div>
          <div class="timeline-item school"><div class="timeline-item-label">Primary School Years</div><div class="timeline-track"><div class="timeline-bar" style="left:38%;width:32%;"></div></div><div class="timeline-date">2013–2019</div></div>
        </div>
      </div>

      <div class="timeline-section">
        <div class="timeline-section-name">Later studies</div>
        <div class="timeline-items">
          <div class="timeline-item later"><div class="timeline-item-label">Growing Up</div><div class="timeline-track"><div class="timeline-bar" style="left:54%;width:27%;"></div></div><div class="timeline-date">2016–2020</div></div>
          <div class="timeline-item later"><div class="timeline-item-label">Age of Wonder</div><div class="timeline-track"><div class="timeline-bar" style="left:92%;width:8%;"></div></div><div class="timeline-date">2023–2025</div></div>
        </div>
      </div>
    </div>
  </section>

  <section class="controls">
    <input id="search" type="search" placeholder="Search variable name, label, description, study, table, or theme…">
    <div class="meta" id="result-meta">Loading registry…</div>
  </section>

  <section class="registry-layout">
    <aside class="panel">
      <h3>Themes</h3>
      <div class="theme-list" id="theme-list"></div>
    </aside>

    <div class="panel results-wrap">
      <h3>Registry results</h3>
      <div class="results-table-wrap" id="results-table-wrap">
        <table class="registry-table">
          <thead>
            <tr>
              <th>Variable</th>
              <th>Label</th>
              <th>Study</th>
              <th>Theme</th>
              <th>Table</th>
              <th>Type</th>
              <th>N</th>
            </tr>
          </thead>
          <tbody id="results-body"></tbody>
        </table>
        <div class="empty-state" id="empty-state" style="display:none;">No variables matched the current filter.</div>
      </div>
    </div>

    <aside class="detail-card">
      <h3>Selected variable</h3>
      <div class="detail-body" id="detail-body">
        <div class="detail-empty">Select a variable to inspect its grounded registry record.</div>
      </div>
    </aside>
  </section>
</div>

<script>
const state = {
  q: '',
  theme: 'All',
  rows: [],
  themes: [],
  summary: {},
  total: 0,
  selectedKey: '',
};

function escHtml(t) {
  return String(t || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function fmt(n) {
  const num = Number(n || 0);
  return Number.isFinite(num) ? num.toLocaleString() : '—';
}
function detailValue(v) {
  return v ? escHtml(v) : '<span style="color:#8a94a6">—</span>';
}

async function loadRegistry() {
  const params = new URLSearchParams();
  if (state.q) params.set('q', state.q);
  if (state.theme && state.theme !== 'All') params.set('theme', state.theme);
  params.set('limit', '200');

  const res = await fetch('/api/registry?' + params.toString());
  const data = await res.json();
  state.rows = data.rows || [];
  state.themes = data.themes || [];
  state.summary = data.summary || {};
  state.total = data.total || 0;

  renderSummary();
  renderThemes();
  renderRows();

  if (state.rows.length) {
    const current = state.rows.find(r => (r.table + '::' + r.variable) === state.selectedKey);
    selectRow(current || state.rows[0]);
  } else {
    state.selectedKey = '';
    document.getElementById('detail-body').innerHTML = '<div class="detail-empty">No registry record is selected.</div>';
  }
}

function renderSummary() {
  document.getElementById('stat-vars').textContent = fmt(state.summary.variables);
  document.getElementById('stat-themes').textContent = fmt(state.summary.themes);
  document.getElementById('stat-tables').textContent = fmt(state.summary.tables);
  document.getElementById('stat-html').textContent = fmt(state.summary.html_files);

  const parts = [];
  parts.push(`${fmt(state.total)} result${state.total === 1 ? '' : 's'}`);
  if (state.theme && state.theme !== 'All') parts.push(`theme: ${state.theme}`);
  if (state.q) parts.push(`search: “${state.q}”`);
  document.getElementById('result-meta').textContent = parts.join(' · ');
}

function renderThemes() {
  const el = document.getElementById('theme-list');
  const allCount = state.summary.variables || 0;
  const items = [{ name: 'All', count: allCount }, ...state.themes];
  el.innerHTML = items.map(item => {
    const active = item.name === state.theme ? 'active' : '';
    return `<button class="theme-btn ${active}" data-theme="${escHtml(item.name)}">
      <span>${escHtml(item.name)}</span>
      <span class="theme-count">${fmt(item.count)}</span>
    </button>`;
  }).join('');

  el.querySelectorAll('[data-theme]').forEach(btn => {
    btn.addEventListener('click', () => {
      state.theme = btn.getAttribute('data-theme');
      loadRegistry();
    });
  });
}

function renderRows() {
  const body = document.getElementById('results-body');
  const empty = document.getElementById('empty-state');
  if (!state.rows.length) {
    body.innerHTML = '';
    empty.style.display = 'block';
    return;
  }
  empty.style.display = 'none';
  body.innerHTML = state.rows.map(row => {
    const key = `${row.table}::${row.variable}`;
    const active = key === state.selectedKey ? 'active' : '';
    return `<tr class="${active}" data-key="${escHtml(key)}">
      <td class="mono">${escHtml(row.variable)}</td>
      <td>${escHtml(row.label || row.description || '')}</td>
      <td><span class="pill study-chip">${escHtml(row.study_context || '')}</span></td>
      <td><span class="pill">${escHtml(row.theme)}</span></td>
      <td class="mono">${escHtml(row.table)}</td>
      <td>${escHtml(row.type || '')}</td>
      <td>${escHtml(row.non_missing || '')}</td>
    </tr>`;
  }).join('');

  body.querySelectorAll('tr[data-key]').forEach(tr => {
    tr.addEventListener('click', () => {
      const row = state.rows.find(r => `${r.table}::${r.variable}` === tr.getAttribute('data-key'));
      if (row) selectRow(row);
    });
  });
}

function selectRow(row) {
  state.selectedKey = row.table + '::' + row.variable;
  renderRows();
  const source = row.source_html
    ? `<a class="source-link" href="/${encodeURI(row.source_html)}" target="_blank" rel="noopener">${escHtml(row.source_html)}</a>`
    : '<span style="color:#8a94a6">—</span>';
  document.getElementById('detail-body').innerHTML = `
    <dl>
      <dt>Variable</dt><dd><span class="mono">${escHtml(row.variable)}</span></dd>
      <dt>Label</dt><dd>${detailValue(row.label)}</dd>
      <dt>Description</dt><dd>${detailValue(row.description)}</dd>
      <dt>Study</dt><dd><span class="pill study-chip">${escHtml(row.study_context || 'Study not inferred')}</span></dd>
      <dt>Theme</dt><dd><span class="pill">${escHtml(row.theme)}</span></dd>
      <dt>Topic</dt><dd>${detailValue(row.topic)}</dd>
      <dt>Section</dt><dd>${detailValue(row.section)}</dd>
      <dt>Table</dt><dd><span class="mono">${escHtml(row.table)}</span></dd>
      <dt>Table label</dt><dd>${detailValue(row.table_display)}</dd>
      <dt>Project</dt><dd>${detailValue(row.project)}</dd>
      <dt>Type</dt><dd>${detailValue(row.type)}</dd>
      <dt>Non-missing</dt><dd>${detailValue(row.non_missing)}</dd>
      <dt>Entity-complete</dt><dd>${detailValue(row.entities_complete)}</dd>
      <dt>Source HTML</dt><dd>${source}</dd>
    </dl>
  `;
}

let searchTimer = null;
document.getElementById('search').addEventListener('input', (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    state.q = e.target.value.trim();
    loadRegistry();
  }, 250);
});

loadRegistry();
</script>
</body>
</html>
"""


@app.route("/registry")
def registry_page():
    return Response(REGISTRY_HTML, mimetype="text/html")


@app.route("/api/registry")
def registry_api():
    data = _get_variable_registry()
    rows = data["rows"]
    q = (request.args.get("q") or "").strip()
    theme = (request.args.get("theme") or "").strip()
    try:
        limit = max(1, min(int(request.args.get("limit", 200)), 500))
    except ValueError:
        limit = 200

    if theme and theme != "All":
        rows = [row for row in rows if row.get("theme") == theme]

    if q:
        terms = [t for t in re.split(r"\s+", q.lower()) if t]
        filtered = []
        for row in rows:
            haystack = " ".join([
                row.get("variable", ""), row.get("label", ""), row.get("description", ""),
            row.get("table", ""), row.get("section", ""), row.get("topic", ""), row.get("theme", ""),
            row.get("study_context", ""), row.get("project", ""), row.get("table_display", ""),
            ]).lower()
            if all(term in haystack for term in terms):
                filtered.append(row)
        rows = sorted(filtered, key=lambda row: (-_registry_score(row, q), row.get("variable", "")))

    total = len(rows)
    return jsonify({
        "summary": data["summary"],
        "themes": data["themes"],
        "total": total,
        "rows": rows[:limit],
    })


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
    print(f"   Variable registry: http://{args.host}:{args.port}/registry")
    print(f"   Full assistant   : http://{args.host}:{args.port}/assistant")
    print(f"   Chat API         : POST http://{args.host}:{args.port}/api/chat")
    print("\n   Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
