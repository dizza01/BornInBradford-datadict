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

  # Use a local quantized GGUF model:
  python server.py --llm-backend llama_cpp --gguf-model-path models/bib-llama-3.1-8b.Q4_K_M.gguf

  # Experimental: use a non-quantized local Hugging Face model:
  python server.py --llm-backend transformers_local --model meta-llama/Llama-3.2-3B-Instruct

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
RCADS25_PDF  = DATADICT_DIR / "papers" / "RCADS25-Youth-English-2018.pdf"
PRODUCTION_DD_REFERENCE = SCRIPT_DIR / "production_data_dictionary_reference.md"

# ── Import RAG engine from bib_research_assistant.py ──────────────────────────
sys.path.insert(0, str(SCRIPT_DIR))
from bib_research_assistant import (
    SYSTEM_PROMPT,
    retrieve_context,
    query as rag_query,
    query_stream as rag_query_stream,
    get_chroma_client,
  parse_html_sections,
    _get_hf_client,
    DEFAULT_MODEL,
    GGUF_MODEL_DEFAULT,
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
current_llm_backend: str = os.getenv("LLM_BACKEND", "hf_api")
current_gguf_model_path: str = os.getenv("GGUF_MODEL_PATH", str(GGUF_MODEL_DEFAULT))
current_llama_n_ctx: int = int(os.getenv("LLAMA_CPP_N_CTX", "4096"))
current_llama_n_gpu_layers: int = int(os.getenv("LLAMA_CPP_N_GPU_LAYERS", "-1"))
current_llama_chat_format: str = os.getenv("LLAMA_CPP_CHAT_FORMAT", "llama-3")
current_llama_n_threads: int = int(os.getenv("LLAMA_CPP_N_THREADS", "0"))
current_llama_verbose: bool = os.getenv("LLAMA_CPP_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}
current_transformers_device: str = os.getenv("TRANSFORMERS_LOCAL_DEVICE", "auto")
current_transformers_dtype: str = os.getenv("TRANSFORMERS_LOCAL_DTYPE", "auto")
current_transformers_attn_implementation: str = os.getenv("TRANSFORMERS_LOCAL_ATTN_IMPLEMENTATION", "")
current_rag_n_results: int = int(os.getenv("RAG_N_RESULTS", "0") or "0")
current_rag_context_max_chars: int = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "0") or "0")
_init_lock = threading.Lock()
_registry_lock = threading.Lock()
_registry_cache: Optional[dict[str, Any]] = None
_rcads25_items_cache: Optional[list[tuple[int, str]]] = None

def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_production_dd_reference() -> str:
    """Load the compact data-dictionary reference for production LLM calls."""
    if not _env_flag("BIB_USE_DD_REFERENCE", True):
        return ""
    try:
        text = PRODUCTION_DD_REFERENCE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    max_chars = int(os.getenv("BIB_DD_REFERENCE_MAX_CHARS", "6500"))
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


_PRODUCTION_DD_REFERENCE_TEXT = _load_production_dd_reference()

PRODUCTION_SYSTEM_PROMPT = SYSTEM_PROMPT + """

Production answer style:
- Be useful, not terse. For non-variable questions, answer in short prose or simple bullet points. Avoid markdown tables unless the user explicitly asks for a table.
- Stay grounded. Only mention papers, findings, measures, tables, and variables that appear in the retrieved context or deterministic registry results.
- Keep variable discovery/listing answers unchanged: when the user asks for variables, continue to show the complete queried variable list in the existing variable table/export UI.
- Distinguish source types carefully. Published research papers/articles are different from data summaries, protocols, questionnaires, survey instruments, data dictionary pages, registry tables, and internal documentation PDFs.
- Do not count a source as a research paper just because it is a PDF or appears under retrieved paper/PDF context. If a source title includes terms like "Data Summary", "questionnaire", "survey", "protocol", "data dictionary", or "registry", label it as documentation/protocol/questionnaire unless the retrieved context clearly shows it is a published research article.
- For paper-listing questions, include up to 5 relevant published papers with title, year, why each is relevant, and what evidence/measure/topic links it to the question. If the retrieved context only contains documentation or data summaries, say that no clearly relevant published research paper was found in the retrieved context and then separately summarize the documentation evidence.
- For one-paper requests, give the paper title/year and a concise explanation of what it studied, why it is relevant to the user question, and any limits of the retrieved context.
- For paper summaries, summarize the specific paper named or clearly identified by the latest user turn. If the requested paper is not in the retrieved context, say you do not have enough context for that paper.
- For "papers related to X" questions, require substantive relevance to X. A passing mention that X data exist is not enough to list the source as a paper about X; describe it as a passing mention or supporting documentation instead.
- Do not include table rows, bullets, or list items with blank or vague relevance. If the relevant-information field would be empty, omit that source or state that the retrieved context is insufficient.
- Avoid one-sentence answers when the user asks to explain, summarize, compare, or provide research papers.
- Avoid repeating the same point in multiple paragraphs. If evidence is limited, say that briefly rather than restating general background.
- For definition questions like "what does X mean?" or "what does X stand for?", answer the definition and brief context only. Do not include variable tables unless the user asks where it occurs in the data.
- When the user asks about time periods, dates, waves, ages, or when something was carried out, answer the timing question first in prose or bullets. Separate exact calendar dates from age ranges, school stages, study waves, or cohort labels. If the retrieved context gives a study stage but not exact calendar dates, say that explicitly. Do not include administration procedures, task descriptions, or methodological detail unless the user asks how the measure was administered.
""" + (
    "\nProduction data dictionary quick reference:\n"
    + _PRODUCTION_DD_REFERENCE_TEXT
    + "\n"
    if _PRODUCTION_DD_REFERENCE_TEXT
    else ""
)


def _ensure_clients():
    global chroma_client, llm_client
    with _init_lock:
        if chroma_client is None:
            chroma_client = get_chroma_client()
        if llm_client is None:
            llm_client = _get_hf_client(
                current_model,
                backend=current_llm_backend,
                gguf_model_path=current_gguf_model_path,
                llama_n_ctx=current_llama_n_ctx,
                llama_n_gpu_layers=current_llama_n_gpu_layers,
                llama_chat_format=current_llama_chat_format,
                llama_n_threads=current_llama_n_threads,
                llama_verbose=current_llama_verbose,
                transformers_device=current_transformers_device,
                transformers_dtype=current_transformers_dtype,
                transformers_attn_implementation=current_transformers_attn_implementation,
            )


def _llm_unavailable_message() -> str:
    if current_llm_backend == "llama_cpp":
        return "LLM client not available — check llama-cpp-python install and GGUF model path"
    if current_llm_backend == "transformers_local":
        return "LLM client not available — check torch/transformers install, model access, and available memory"
    return "LLM client not available — check HF_TOKEN in .env"


def _effective_rag_n_results() -> int:
    if current_rag_n_results > 0:
        return current_rag_n_results
    return 5


def _effective_rag_context_max_chars() -> int:
    if current_rag_context_max_chars > 0:
        return current_rag_context_max_chars
    return 0


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

_VARIABLE_DISCOVERY_RE = re.compile(
  r"\b("
  r"what|which|find|show|list|search|give|return|identify|available|exist|exists|include|includes|contain|contains"
  r")\b.*\b("
  r"variable|variables|measure|measures|field|fields|data|dataset|datasets"
  r")\b"
  r"|\b(variable|variables|measure|measures|field|fields|data|dataset|datasets)\b.*\b("
  r"available|exist|exists|related|about|around|for|on"
  r")\b",
  re.I,
)

_VARIABLE_DETAIL_RE = re.compile(
  r"\b("
  r"what does|what is|explain|define|meaning of|mean|means|"
  r"items? of|questions? of|response options?|categories|values|scoring|score|"
  r"calculated|derived|coded|coding|interpret|interpretation"
  r")\b",
  re.I,
)

_DEFINITION_ONLY_RE = re.compile(
  r"\b(what does|what is|define|meaning of|mean|means|stand(?:s)? for)\b",
  re.I,
)

_VARIABLE_SEARCH_STOPWORDS = {
  "a", "about", "across", "after", "age", "all", "also", "an", "and", "any",
  "are", "around", "as", "at", "available", "be", "been", "before",
  "between", "bib", "born", "bradford", "by", "can", "cohort", "cohorts",
  "contain", "contains", "data", "dataset", "datasets", "different", "do",
  "does", "during", "exist", "exists", "field", "fields", "find", "for",
  "from", "give", "has", "have", "how", "in", "include", "includes",
  "including", "into", "is", "item", "items", "list", "measure", "measures",
  "near", "of", "on", "or", "occur", "occurs", "over", "please", "quality",
  "related", "return", "search", "show", "study", "studies", "table",
  "tables", "that", "the", "their", "theme", "themes", "there", "these",
  "this", "through", "to", "used", "using", "variable", "variables", "was",
  "were", "what", "when", "where", "which", "who", "with", "within",
  "wonder",
}

_VARIABLE_SYNONYMS: dict[str, list[str]] = {
  "anxiety": ["anxiety", "anxious", "rcads", "gad", "worry", "worries", "panic", "fear", "phobia", "nervous"],
  "depression": ["depression", "depressive", "depressed", "epds", "rcads", "mood", "sad", "low mood"],
  "mental health": ["mental health", "wellbeing", "well-being", "anxiety", "depression", "stress", "mood", "sdq", "rcads", "epds"],
  "wellbeing": ["wellbeing", "well-being", "quality of life", "life satisfaction", "happiness", "mood"],
  "stress": ["stress", "stressed", "distress", "strain"],
  "ethnicity": ["ethnicity", "ethnic", "race", "heritage", "country of birth", "language"],
  "birthweight": ["birthweight", "birth weight", "birth_weight", "weight at birth"],
  "gestation": ["gestation", "gestational", "gestational age", "pregnancy weeks", "term birth"],
  "pregnancy": ["pregnancy", "pregnant", "antenatal", "maternity", "maternal", "gestation", "parity"],
  "sleep": ["sleep", "bedtime", "insomnia", "night waking", "tired", "fatigue"],
  "diet": ["diet", "dietary", "nutrition", "food", "fruit", "vegetable", "ffq", "food frequency"],
  "smoking": ["smoking", "smoke", "smoker", "cigarette", "tobacco", "vaping", "vape"],
  "alcohol": ["alcohol", "drinking", "drink", "units alcohol"],
  "breastfeeding": ["breastfeeding", "breast feeding", "breastfeed", "breastfed", "infant feeding"],
  "education": ["education", "educational", "school", "gcse", "ks1", "ks2", "ks4", "eyfs", "attainment"],
  "school": ["school", "education", "attendance", "attainment", "eyfs", "ks1", "ks2", "ks4", "gcse"],
  "development": ["development", "developmental", "cognitive", "language", "speech", "motor", "milestone"],
  "physical activity": ["physical activity", "activity", "exercise", "sedentary", "sport"],
  "bmi": ["bmi", "body mass index", "height", "weight", "obesity", "overweight", "adiposity"],
  "obesity": ["obesity", "obese", "overweight", "bmi", "adiposity", "body mass"],
  "pollution": ["pollution", "air pollution", "air quality", "no2", "pm10", "pm2.5", "pm25"],
  "environment": ["environment", "environmental", "pollution", "green space", "greenspace", "neighbourhood", "neighborhood"],
  "deprivation": ["deprivation", "deprived", "imd", "socioeconomic", "ses", "poverty"],
  "income": ["income", "earnings", "salary", "benefits", "financial", "affluence", "fas"],
  "postcode": ["postcode", "address", "lsoa", "ward", "geography", "geographic"],
  "genetics": ["genetic", "genetics", "genotype", "genotyping", "dna", "snp", "polygenic", "omics"],
  "omics": ["omics", "metabolomics", "proteomics", "glycomics", "methylation", "genomics"],
  "biosample": ["biosample", "biobank", "sample", "blood", "serum", "plasma", "urine", "saliva"],
  "hospital": ["hospital", "hes", "admission", "inpatient", "outpatient", "a&e", "emergency"],
  "gp": ["gp", "general practice", "primary care", "prescription", "medication"],
}

_STUDY_FILTER_ALIASES: dict[str, list[str]] = {
  "Age of Wonder": ["age of wonder", "ageofwonder", "aow"],
  "Baseline": ["baseline", "bib_baseline"],
  "BiB 1000": ["bib1000", "bib 1000", "bib_1000"],
  "Growing Up": ["growing up", "growingup"],
  "BiBBS": ["bibbs", "better start"],
  "Core Cohort": ["cohortinfo", "cohort info", "core cohort"],
  "Starting School": ["starting school", "startingschool"],
  "Primary School Years": ["primary school", "primaryschoolyears"],
  "Geographic Linkage": ["geographic", "geography", "postcode", "lsoa"],
  "Biosamples & Biobank": ["biosamples", "biobank"],
  "Metabolomics": ["metabolomics"],
  "Proteomics": ["proteomics"],
  "COVID-19 Surveys": ["covid", "covid-19"],
}

_BROAD_SYNONYM_KEYS = {
  "bmi",
  "development",
  "education",
  "environment",
  "mental health",
  "obesity",
  "omics",
  "pregnancy",
  "school",
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
    csv_variable_id = _clean_value(row.get("variable_id", ""))
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
      "variable_id": csv_variable_id or f"{table_id}.{variable}".strip("."),
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
  by_variable_id = {
    row["variable_id"].lower(): row
    for row in rows
    if row.get("variable_id")
  }
  by_variable: dict[str, list[dict[str, Any]]] = {}
  for row in rows:
    variable = _clean_value(row.get("variable", ""))
    if variable:
      by_variable.setdefault(variable.lower(), []).append(row)
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
    "by_variable_id": by_variable_id,
    "by_variable": by_variable,
  }


def _identifier_in_text(identifier: str, text: str) -> bool:
  if not identifier:
    return False
  pattern = rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])"
  return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _table_parts(table_id: str) -> tuple[str, str, str]:
  """Normalise current and legacy table IDs for answer-to-registry matching."""
  table_id = _clean_value(table_id).lower()
  if "." in table_id:
    project, table = table_id.split(".", 1)
  else:
    project, table = "", table_id

  project_base = re.sub(r"_20\d{2}$", "", project)
  table_base = re.sub(r"_dr\d+$", "", table)
  return project_base, table_base, table


def _table_match_score(answer_table: str, registry_table: str) -> int:
  """Score exact or legacy table aliases, e.g. survey_mod02_dr23 -> survey_mod02."""
  answer_table = _clean_value(answer_table).lower()
  registry_table = _clean_value(registry_table).lower()
  if not answer_table or not registry_table:
    return 0
  if answer_table == registry_table:
    return 100

  answer_project, answer_base, answer_raw_table = _table_parts(answer_table)
  registry_project, registry_base, _ = _table_parts(registry_table)
  if answer_project and registry_project and answer_project != registry_project:
    return 0
  if answer_base != registry_base:
    return 0

  score = 60
  release_match = re.search(r"_dr(\d{2})$", answer_raw_table)
  if release_match and f"_20{release_match.group(1)}" in registry_table:
    score += 30
  return score


def _extract_table_candidates(text: str) -> list[str]:
  candidates = re.findall(r"\bBiB_[A-Za-z0-9_]+\.[A-Za-z0-9_]+\b", text)
  seen: set[str] = set()
  ordered: list[str] = []
  for candidate in candidates:
    key = candidate.lower()
    if key not in seen:
      seen.add(key)
      ordered.append(candidate)
  return ordered


def _variable_export_record(row: dict[str, Any]) -> dict[str, str]:
  return {
    "variable_id": _clean_value(row.get("variable_id", "")),
    "variable": _clean_value(row.get("variable", "")),
    "table": _clean_value(row.get("table", "")),
    "label": _clean_value(row.get("label", "")),
    "description": _clean_value(row.get("description", "")),
    "type": _clean_value(row.get("type", "")),
    "non_missing": _clean_value(row.get("non_missing", "")),
    "topic": _clean_value(row.get("topic", "")),
    "theme": _clean_value(row.get("theme", "")),
    "study_context": _clean_value(row.get("study_context", "")),
  }


def _normalise_search_text(text: str) -> str:
  text = _clean_value(text).lower()
  text = text.replace("_", " ")
  text = re.sub(r"[^a-z0-9.+&/ -]+", " ", text)
  return re.sub(r"\s+", " ", text).strip()


def _term_in_text(term: str, text: str) -> bool:
  term = _normalise_search_text(term)
  text = _normalise_search_text(text)
  if not term or not text:
    return False
  if re.search(r"[^a-z0-9]", term):
    return term in text
  return re.search(rf"\b{re.escape(term)}s?\b", text) is not None


def _row_search_text(row: dict[str, Any]) -> str:
  return " ".join(
    _clean_value(row.get(field, ""))
    for field in [
      "variable_id", "variable", "label", "description", "topic", "theme",
      "section", "table", "table_name", "table_display", "project",
      "study_context", "entity_type",
    ]
  )


def _looks_like_variable_discovery(question: str) -> bool:
  if _looks_like_variable_detail_question(question):
    return False
  return bool(_VARIABLE_DISCOVERY_RE.search(question or ""))


def _looks_like_variable_detail_question(question: str) -> bool:
  q = _normalise_search_text(question)
  if not q:
    return False

  if _VARIABLE_DETAIL_RE.search(q):
    return True
  if re.search(r"\b\d+\s+items?\b", q):
    return True
  if re.search(r"\bitems?\s+(?:in|for|from|on)\b", q):
    return True
  return False


def _looks_like_available_item_registry_question(question: str) -> bool:
  """Detect item/field availability questions that should use the registry.

  "What are the 25 items of RCAD25?" is a scale-content question, but
  "What items are available for RCAD?" usually means "which data fields exist?".
  Keep this narrow so explanatory questionnaire questions still go through RAG.
  """
  q = _normalise_search_text(question)
  if not q or not re.search(r"\bitems?\b", q):
    return False
  if re.search(r"\b\d+\s+items?\b", q):
    return False
  return bool(
    re.search(
      r"\b(available|exist|exists|included|recorded|collected|stored|present)\b",
      q,
    )
  )


def _history_mentions_rcads25(history: list[dict[str, str]]) -> bool:
  history_text = " ".join(
    _clean_value(turn.get("content", ""))
    for turn in (history or [])[-4:]
    if isinstance(turn, dict)
  )
  q = _normalise_search_text(history_text)
  return bool(re.search(r"\brcads?\s*[- ]?\s*25\b|\brcads?25\b|\brevised child(?:ren)?s? anxiety and depression scale\b", q))


def _looks_like_rcads25_item_wording_question(
  question: str,
  history: Optional[list[dict[str, str]]] = None,
) -> bool:
  q = _normalise_search_text(question)
  if not q or not re.search(r"\bitems?\b|\bquestions?\b|\bwording\b", q):
    return False
  if re.search(r"\brcads?\s*[- ]?\s*25\b|\brcads?25\b|\brcads?\b", q):
    return True
  if re.search(r"\b(these|those|the)\s+(items?|questions?)\b", q) and _history_mentions_rcads25(history or []):
    return True
  return False


def _load_rcads25_items() -> list[tuple[int, str]]:
  """Extract the 25 RCADS-25 item wordings from the source PDF."""
  global _rcads25_items_cache
  if _rcads25_items_cache is not None:
    return _rcads25_items_cache
  if not RCADS25_PDF.exists():
    _rcads25_items_cache = []
    return _rcads25_items_cache

  try:
    import fitz
    doc = fitz.open(str(RCADS25_PDF))
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()
  except Exception:
    _rcads25_items_cache = []
    return _rcads25_items_cache

  matches = re.findall(
    r"(?ms)^\s*(\d{1,2})\.\s+(.*?)(?:\n\s*Never\s*\n\s*Sometimes\s*\n\s*Often\s*\n\s*Always)",
    text,
  )
  items: list[tuple[int, str]] = []
  for number, wording in matches:
    item_number = int(number)
    if 1 <= item_number <= 25:
      clean_wording = re.sub(r"\s+", " ", wording).strip()
      items.append((item_number, clean_wording))

  seen: set[int] = set()
  deduped: list[tuple[int, str]] = []
  for number, wording in sorted(items, key=lambda item: item[0]):
    if number not in seen:
      deduped.append((number, wording))
      seen.add(number)
  _rcads25_items_cache = deduped
  return _rcads25_items_cache


def _direct_rcads25_item_answer(
  question: str,
  history: Optional[list[dict[str, str]]] = None,
) -> Optional[dict[str, Any]]:
  if not _looks_like_rcads25_item_wording_question(question, history):
    return None

  items = _load_rcads25_items()
  if len(items) != 25:
    return None

  q = _normalise_search_text(question)
  wants_asked_format = bool(re.search(r"\b(how|asked|ask|module|survey|response|options?)\b", q))
  lines = [
    (
      "In the RCADS-25 Youth-English source questionnaire, each item is asked as a statement "
      "with the response options `Never`, `Sometimes`, `Often`, and `Always`."
      if wants_asked_format
      else "The 25 RCADS-25 Youth-English items in the source questionnaire are:"
    ),
    "",
  ]
  for number, wording in items:
    lines.append(f"{number}. {wording}")
  lines.extend([
    "",
    f"Source: `{RCADS25_PDF.name}`.",
    "Note: these are questionnaire item wordings, not separate `rcad_ga` variable IDs. Derived registry variables such as `rcad_ga`, `rcad_md`, and `rcad_total` are scale scores/categories.",
  ])
  return {"answer": "\n".join(lines), "variables": []}


def _looks_like_definition_only_question(question: str) -> bool:
  """Return True for glossary/explanation questions, not data-location asks."""
  q = _normalise_search_text(question)
  if not q or not _DEFINITION_ONLY_RE.search(q):
    return False
  if re.search(
    r"\b(where|occur|occurs|available|variables?|fields?|tables?|dataset|data dictionary|export|csv)\b",
    q,
  ):
    return False
  return True


def _infer_study_filters(question: str) -> list[str]:
  q = _normalise_search_text(question)
  filters: list[str] = []
  for study, aliases in _STUDY_FILTER_ALIASES.items():
    if any(_term_in_text(alias, q) for alias in aliases):
      filters.append(study)
  return filters


def _expand_variable_search_terms(question: str) -> list[str]:
  q = _normalise_search_text(question)
  terms: list[str] = []

  def add(term: str) -> None:
    term = _normalise_search_text(term)
    if term and term not in terms:
      terms.append(term)

  for key, synonyms in _VARIABLE_SYNONYMS.items():
    key_in_query = _term_in_text(key, q)
    alias_in_query = key not in _BROAD_SYNONYM_KEYS and any(
      _term_in_text(alias, q) for alias in synonyms
    )
    if key_in_query or alias_in_query:
      for synonym in synonyms:
        add(synonym)

  for token in re.findall(r"\b[a-z][a-z0-9_]{2,}\b", question.lower()):
    token_norm = _normalise_search_text(token)
    if token_norm in _VARIABLE_SEARCH_STOPWORDS:
      continue
    if token_norm.startswith("bib_"):
      continue
    add(token_norm)

  for phrase in re.findall(r"['\"]([^'\"]{3,80})['\"]", question):
    add(phrase)

  # Drop study aliases from content matching; they are handled as filters.
  study_aliases = {
    _normalise_search_text(alias)
    for aliases in _STUDY_FILTER_ALIASES.values()
    for alias in aliases
  }
  study_alias_tokens = {
    token
    for alias in study_aliases
    for token in alias.split()
    if len(token) > 3
  }
  filtered_terms = [
    term for term in terms
    if term not in study_aliases and term not in study_alias_tokens
  ]

  # For physical activity, standalone "activity" and "physical" are too broad:
  # they match unrelated registry areas such as dental activity or physical IDs.
  if "physical activity" in filtered_terms:
    filtered_terms = [
      term for term in filtered_terms
      if term not in {"activity", "physical"}
    ]
  return filtered_terms


def _variable_search_score(row: dict[str, Any], terms: list[str]) -> tuple[int, list[str]]:
  if not terms:
    return 1, []

  weighted_fields = {
    "variable_id": 5,
    "variable": 5,
    "label": 4,
    "description": 3,
    "topic": 3,
    "theme": 3,
    "section": 2,
    "table": 2,
    "table_display": 2,
    "project": 1,
    "study_context": 1,
  }
  score = 0
  matched: list[str] = []
  for term in terms:
    term_score = 0
    for field, weight in weighted_fields.items():
      if _term_in_text(term, _clean_value(row.get(field, ""))):
        term_score = max(term_score, weight)
    if term_score:
      score += term_score
      matched.append(term)

  return score, matched


def _row_matches_study_filters(row: dict[str, Any], study_filters: list[str]) -> bool:
  if not study_filters:
    return True
  study_context = _normalise_search_text(_clean_value(row.get("study_context", "")))
  for study in study_filters:
    study_norm = _normalise_search_text(study)
    if not study_norm:
      continue
    if study_context == study_norm:
      return True
    if study_context.startswith(f"{study_norm} "):
      return True
    if study_context.startswith(f"{study_norm} ("):
      return True
  return False


def _wants_variable_study_summary(question: str) -> bool:
  """Detect variable questions that ask where matches occur, not just what matches."""
  q = _normalise_search_text(question)
  if not q:
    return False
  return bool(re.search(r"\b(cohort|cohorts|study|studies|wave|waves|where|occur|occurs)\b", q))


def _summarise_variable_matches_by_study(matches: list[dict[str, str]]) -> list[dict[str, Any]]:
  grouped: dict[str, dict[str, Any]] = {}
  for row in matches:
    study = _clean_value(row.get("study_context", "")) or "Study not inferred"
    table = _clean_value(row.get("table", ""))
    variable_id = _clean_value(row.get("variable_id", ""))
    label = _clean_value(row.get("label", ""))

    item = grouped.setdefault(
      study,
      {
        "study_context": study,
        "n_variables": 0,
        "_tables": set(),
        "examples": [],
      },
    )
    item["n_variables"] += 1
    if table:
      item["_tables"].add(table)
    if variable_id and len(item["examples"]) < 4:
      example = {"variable_id": variable_id}
      if label:
        example["label"] = label
      item["examples"].append(example)

  summary: list[dict[str, Any]] = []
  for item in grouped.values():
    tables = sorted(item.pop("_tables"))
    item["n_tables"] = len(tables)
    item["tables"] = tables[:5]
    summary.append(item)

  summary.sort(key=lambda item: (-int(item.get("n_variables", 0) or 0), item.get("study_context", "")))
  return summary


def _search_variable_registry(
  question: str,
  limit: int = 5000,
  require_discovery_intent: bool = True,
) -> Optional[dict[str, Any]]:
  """Complete CSV-backed variable discovery with synonym expansion."""
  if require_discovery_intent and not _looks_like_variable_discovery(question):
    return None

  terms = _expand_variable_search_terms(question)
  study_filters = _infer_study_filters(question)
  if not terms and not study_filters:
    return None

  matches: list[dict[str, Any]] = []
  for row in _get_variable_registry()["rows"]:
    if not _row_matches_study_filters(row, study_filters):
      continue

    score, matched_terms = _variable_search_score(row, terms)
    if terms and score <= 0:
      continue

    record = _variable_export_record(row)
    record["matched_terms"] = ", ".join(matched_terms)
    record["_score"] = str(score)
    matches.append(record)

  matches.sort(
    key=lambda r: (
      -float(r.get("_score", "0") or 0),
      r.get("study_context", ""),
      r.get("table", ""),
      r.get("variable", ""),
    )
  )

  total = len(matches)
  limited = matches[: max(1, int(limit))]
  for row in limited:
    row.pop("_score", None)

  study_summary = _summarise_variable_matches_by_study(matches) if _wants_variable_study_summary(question) else []

  result = {
    "query": question,
    "terms": terms,
    "study_filters": study_filters,
    "total": total,
    "returned": len(limited),
    "truncated": total > len(limited),
    "rows": limited,
  }
  if study_summary:
    result["summary_mode"] = "study_context"
    result["study_summary"] = study_summary
  return result


def _search_available_item_registry(question: str) -> Optional[dict[str, Any]]:
  if not _looks_like_available_item_registry_question(question):
    return None
  result = _search_variable_registry(
    question,
    limit=5000,
    require_discovery_intent=False,
  )
  if result and int(result.get("total", 0) or 0) > 0:
    result["summary_mode"] = "available_items"
  return result


def _format_variable_discovery_answer(variable_results: dict[str, Any]) -> str:
  total = int(variable_results.get("total", 0) or 0)
  returned = int(variable_results.get("returned", 0) or 0)
  study_filters = variable_results.get("study_filters") or []
  terms = variable_results.get("terms") or []

  if total == 0:
    return (
      "I could not find matching variables in the registry for that request. "
      "Try a broader term or remove study/wave filters."
    )

  if variable_results.get("summary_mode") == "study_context":
    summary = variable_results.get("study_summary") or []
    term_text = f" Matched terms: {', '.join(terms[:8])}." if terms else ""
    shown_summary = summary[:10]
    lines = [
      (
        f"These variables occur across {len(summary)} study context/cohort "
        f"label{'s' if len(summary) != 1 else ''} in the data dictionary registry "
        f"({total} matching variable{'s' if total != 1 else ''})."
      ),
      "Study contexts with matching variables:",
    ]
    for item in shown_summary:
      study = item.get("study_context") or "Study not inferred"
      n_variables = int(item.get("n_variables", 0) or 0)
      n_tables = int(item.get("n_tables", 0) or 0)
      examples = [
        example.get("variable_id", "")
        for example in item.get("examples", [])
        if example.get("variable_id")
      ][:3]
      table_text = f" across {n_tables} table{'s' if n_tables != 1 else ''}" if n_tables else ""
      example_text = f" Examples: {', '.join(examples)}." if examples else ""
      lines.append(
        f"- {study}: {n_variables} variable{'s' if n_variables != 1 else ''}{table_text}.{example_text}"
      )
    if len(summary) > len(shown_summary):
      lines.append(
        f"- {len(summary) - len(shown_summary)} more study context/cohort "
        "labels are included in the exportable results below."
      )
    if term_text:
      lines.append(term_text.strip())
    if variable_results.get("truncated"):
      lines.append(
        f"The first {returned} variable rows are shown below; use Add all or Export CSV for the available set."
      )
    else:
      lines.append("The full matching variable set is shown below and can be added to the export basket.")
    return "\n".join(lines)

  if variable_results.get("summary_mode") == "available_items":
    scope = f" in {', '.join(study_filters)}" if study_filters else ""
    term_text = f" Matched terms: {', '.join(terms[:8])}." if terms else ""
    truncated = (
      f" The first {returned} are available below; use Add all or Export CSV for the full set."
      if variable_results.get("truncated")
      else " The complete set is shown below and can be added to the export basket."
    )
    return (
      f"I found {total} data dictionary item/field match{'es' if total != 1 else ''}{scope}. "
      "These are registry variables/fields rather than questionnaire item wording."
      f"{term_text}{truncated}"
    )

  scope = f" in {', '.join(study_filters)}" if study_filters else ""
  term_text = f" Matching terms included: {', '.join(terms[:8])}." if terms else ""
  truncated = (
    f" The first {returned} are available below; use Add all or Export CSV for the full set."
    if variable_results.get("truncated")
    else " The complete set is shown below and can be added to the export basket."
  )
  return (
    f"I found {total} matching variable{'s' if total != 1 else ''}{scope} "
    f"using the data dictionary registry.{term_text}{truncated}"
  )


def _extract_variables_for_export(text: str, limit: int = 80) -> list[dict[str, str]]:
  """Find valid registry variables mentioned in a chat answer for UX export."""
  if not text:
    return []

  registry = _get_variable_registry()
  by_variable_id = registry.get("by_variable_id", {})
  by_variable = registry.get("by_variable", {})
  selected: dict[str, dict[str, str]] = {}
  table_candidates = _extract_table_candidates(text)

  full_id_candidates = re.findall(
    r"\bBiB_[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+\b",
    text,
  )
  for variable_id in full_id_candidates:
    row = by_variable_id.get(variable_id.lower())
    if row:
      selected[variable_id] = _variable_export_record(row)

  if len(selected) >= limit:
    return list(selected.values())[:limit]

  variable_candidates = {
    token.lower()
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_]{1,}\b", text)
  }
  for variable_l in variable_candidates:
    matches = by_variable.get(variable_l, [])
    if not matches:
      continue
    variable = matches[0].get("variable", "")

    if len(matches) == 1:
      # Avoid exporting generic prose matches such as "relationship" from paper
      # summaries. Full variable IDs are handled above; this fallback is only
      # for code-like variable names such as rcad_total_cat or awb7_3_foo.
      if "_" not in variable and not re.search(r"\d", variable):
        continue
      row = matches[0]
      selected.setdefault(row.get("variable_id", ""), _variable_export_record(row))
      continue

    # For duplicated names like "id", require the table to be present too.
    for row in matches:
      table_id = _clean_value(row.get("table", ""))
      if table_id and any(
        _table_match_score(candidate, table_id) > 0 for candidate in table_candidates
      ):
        selected.setdefault(row.get("variable_id", ""), _variable_export_record(row))
      if len(selected) >= limit:
        return list(selected.values())[:limit]

  return list(selected.values())[:limit]


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
<a id="bib-registry-nav" href="/registry" title="Open variable registry">
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
        return jsonify({"error": _llm_unavailable_message()}), 503

    if not chroma_client:
        return jsonify({"error": "Vector database not initialised — run --build first"}), 503

    try:
        direct_answer = _direct_rcads25_item_answer(question, history)
        if direct_answer:
            result: dict = direct_answer
            if show_ctx:
                result["context"] = f"Source: {RCADS25_PDF.name}"
            return jsonify(result)

        variable_results = (
            _search_variable_registry(question)
            or _search_available_item_registry(question)
        )
        if variable_results:
            answer = _format_variable_discovery_answer(variable_results)
            result: dict = {
                "answer": answer,
                "variables": variable_results.get("rows", []),
                "variable_results": variable_results,
            }
            if show_ctx:
                result["context"] = ""
            return jsonify(result)

        context = retrieve_context(question, chroma_client)
        answer  = rag_query(
            question, chroma_client, llm_client,
            model=current_model, show_context=False,
            history=history,
            system_prompt=PRODUCTION_SYSTEM_PROMPT,
            retrieval_n_results=_effective_rag_n_results(),
            max_context_chars=_effective_rag_context_max_chars(),
        )
        result: dict = {
            "answer": answer,
            "variables": [] if _looks_like_definition_only_question(question) else _extract_variables_for_export(answer),
        }
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
      data: {"variables": [...]} — validated variables mentioned in the answer
      data: {"variable_results": {...}} — complete registry search for variable-discovery questions
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
        return jsonify({"error": _llm_unavailable_message()}), 503
    if not chroma_client:
        return jsonify({"error": "Vector database not initialised — run --build first"}), 503

    def generate():
        full_text = ""
        try:
            direct_answer = _direct_rcads25_item_answer(question, history)
            if direct_answer:
                full_text = direct_answer["answer"]
                yield f"data: {json.dumps({'token': full_text})}\n\n"
                yield f"data: {json.dumps({'variables': []})}\n\n"
                yield 'data: {"done": true}\n\n'
                return

            variable_results = (
                _search_variable_registry(question)
                or _search_available_item_registry(question)
            )
            if variable_results:
                full_text = _format_variable_discovery_answer(variable_results)
                yield f"data: {json.dumps({'token': full_text})}\n\n"
                yield f"data: {json.dumps({'variables': variable_results.get('rows', [])})}\n\n"
                yield f"data: {json.dumps({'variable_results': variable_results})}\n\n"
                yield 'data: {"done": true}\n\n'
                return

            for token in rag_query_stream(
                question, chroma_client, llm_client,
                model=current_model, history=history,
                system_prompt=PRODUCTION_SYSTEM_PROMPT,
                retrieval_n_results=_effective_rag_n_results(),
                max_context_chars=_effective_rag_context_max_chars(),
            ):
                full_text += token
                yield f"data: {json.dumps({'token': token})}\n\n"

            # Strip any footer boilerplate accumulated in the full response
            cleaned = _strip_filler(full_text)
            if cleaned != full_text:
                yield f"data: {json.dumps({'replace': cleaned})}\n\n"
            full_text = cleaned

            variables = [] if _looks_like_definition_only_question(question) else _extract_variables_for_export(full_text)
            if variables:
                yield f"data: {json.dumps({'variables': variables})}\n\n"

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

  #variable-basket {
    background: #fff;
    border-top: 1px solid #e0e4ef;
    padding: 10px 20px;
    box-shadow: 0 -1px 6px rgba(0,0,0,.04);
  }
  .basket-row {
    max-width: 920px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 12px;
    align-items: center;
  }
  .basket-title {
    color: var(--bib-blue);
    font-size: .84rem;
    font-weight: 700;
  }
  #basket-count {
    display: inline-block;
    margin-left: 6px;
    background: #e8f0fb;
    border-radius: 999px;
    padding: 1px 8px;
    color: #14397a;
  }
  #basket-list {
    display: flex;
    gap: 6px;
    overflow-x: auto;
  }
  .basket-empty {
    color: #7a8499;
    font-size: .78rem;
  }
  .var-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #eef4ff;
    border: 1px solid #c3d4f0;
    border-radius: 999px;
    padding: 4px 6px 4px 10px;
    white-space: nowrap;
  }
  .var-chip code { color: #14397a; font-size: .78rem; }
  .var-chip button,
  .basket-actions button,
  .detected-vars button {
    border: 0;
    cursor: pointer;
    font-family: inherit;
  }
  .var-chip button {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #d5e2fb;
    color: #42506b;
  }
  .var-chip button:hover { background: #c24747; color: #fff; }
  .basket-actions {
    display: flex;
    gap: 8px;
  }
  .basket-actions button {
    border-radius: 8px;
    padding: 7px 11px;
    font-size: .78rem;
  }
  #export-vars { background: var(--bib-blue); color: #fff; }
  #clear-vars { background: #e9eef8; color: #42506b; }
  .basket-actions button:disabled {
    opacity: .45;
    cursor: not-allowed;
  }
  .detected-vars {
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid #d0d8e8;
    white-space: normal;
  }
  .detected-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    color: #42506b;
    font-size: .8rem;
    font-weight: 700;
    margin-bottom: 8px;
  }
  .add-all-vars {
    background: var(--bib-blue);
    color: #fff;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: .76rem;
  }
  .detected-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 7px;
  }
  .detected-var {
    background: #f7f9fe;
    border: 1px solid #d0d8e8 !important;
    border-radius: 9px;
    padding: 8px 10px;
    text-align: left;
    color: #1a1a2e;
  }
  .detected-var:hover { border-color: var(--bib-blue) !important; background: #fff; }
  .detected-var code { display: block; margin-bottom: 3px; }
  .detected-var span {
    display: block;
    color: #5f6f91;
    font-size: .76rem;
    line-height: 1.3;
  }
  .study-summary {
    background: linear-gradient(145deg, #ffffff 0%, #eef5ff 100%);
    border: 1px solid #d0d8e8;
    border-radius: 14px;
    padding: 14px;
    white-space: normal;
  }
  .study-summary-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 12px;
  }
  .study-summary-head strong {
    display: block;
    color: #14397a;
    font-size: .98rem;
  }
  .study-summary-head span {
    display: block;
    color: #52647f;
    font-size: .8rem;
    line-height: 1.35;
    margin-top: 2px;
  }
  .study-term-pills {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 6px;
  }
  .study-term-pills span {
    background: #dfeaff;
    border: 1px solid #bfd0ef;
    border-radius: 999px;
    color: #14397a;
    font-size: .74rem;
    padding: 3px 8px;
  }
  .study-summary-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 9px;
    max-height: 420px;
    overflow-y: auto;
    padding-right: 4px;
    scrollbar-gutter: stable;
  }
  .study-card {
    background: rgba(255,255,255,.92);
    border: 1px solid #d3ddf1;
    border-radius: 11px;
    padding: 10px;
  }
  .study-card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 10px;
  }
  .study-card-top strong {
    color: #122b57;
    display: block;
    font-size: .86rem;
  }
  .study-card-top span {
    color: #667691;
    display: block;
    font-size: .76rem;
    line-height: 1.3;
    margin-top: 2px;
  }
  .study-card-top button {
    background: var(--bib-blue);
    border: 0;
    border-radius: 999px;
    color: #fff;
    cursor: pointer;
    flex-shrink: 0;
    font-family: inherit;
    font-size: .72rem;
    padding: 4px 9px;
  }
  .study-vars {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 190px;
    margin-top: 8px;
    overflow-y: auto;
    padding-right: 4px;
    scrollbar-gutter: stable;
  }
  .study-vars button {
    background: #fbfdff;
    border: 1px solid #d3ddf1;
    border-radius: 9px;
    color: #1a1a2e;
    cursor: pointer;
    font-family: inherit;
    padding: 6px 8px;
    text-align: left;
  }
  .study-vars button:hover {
    background: #f1f6ff;
    border-color: var(--bib-blue);
  }
  .study-vars code {
    background: #edf3ff !important;
    color: #14397a;
    display: block;
    font-size: .72rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .study-vars span {
    color: #667691;
    display: block;
    font-size: .72rem;
    line-height: 1.3;
    margin-top: 3px;
  }
  .study-card-note {
    color: #687895;
    font-size: .72rem;
    line-height: 1.3;
    margin-top: 7px;
  }
  .study-summary-note {
    color: #687895;
    font-size: .76rem;
    line-height: 1.3;
    margin-top: 10px;
  }
  .variable-results {
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid #d0d8e8;
    white-space: normal;
  }
  .variable-results-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 10px;
    color: #42506b;
    font-size: .8rem;
    margin-bottom: 8px;
  }
  .variable-results-head strong { color: #14397a; }
  .variable-results-meta,
  .variable-results-more {
    color: #687895;
    font-size: .76rem;
    line-height: 1.3;
    margin-top: 2px;
  }
  .add-all-results {
    background: var(--bib-blue);
    color: #fff;
    border: 0;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: .76rem;
    cursor: pointer;
    font-family: inherit;
    flex-shrink: 0;
  }
  .variable-results-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 7px;
    max-height: 460px;
    overflow-y: auto;
    padding-right: 4px;
    scrollbar-gutter: stable;
  }
  .variable-result {
    background: #f7f9fe;
    border: 1px solid #d0d8e8 !important;
    border-radius: 9px;
    padding: 8px 10px;
    text-align: left;
    color: #1a1a2e;
    cursor: pointer;
    font-family: inherit;
  }
  .variable-result:hover { border-color: var(--bib-blue) !important; background: #fff; }
  .variable-result code { display: block; margin-bottom: 3px; }
  .variable-result span {
    display: block;
    color: #5f6f91;
    font-size: .76rem;
    line-height: 1.3;
  }
  .variable-results-more {
    margin-top: 8px;
  }

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

<div id="variable-basket">
  <div class="basket-row">
    <div class="basket-title">Selected variables <span id="basket-count">0</span></div>
    <div id="basket-list"></div>
    <div class="basket-actions">
      <button id="export-vars" type="button" disabled>Export CSV</button>
      <button id="clear-vars" type="button" disabled>Clear</button>
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
const selectedVariables = new Map();
const STORAGE_KEY = 'bibSelectedVariables';

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
function csvEscape(value) {
  const text = String(value || '');
  if (/[",\n\r]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
  return text;
}
function variableKey(v) {
  return v.variable_id || [v.table, v.variable].filter(Boolean).join('.');
}
function loadSelectedVariables() {
  try {
    const rows = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || '[]');
    rows.forEach(v => {
      const key = variableKey(v);
      if (key) selectedVariables.set(key, v);
    });
  } catch {
    selectedVariables.clear();
  }
}
function saveSelectedVariables() {
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(Array.from(selectedVariables.values())));
}
function renderBasket() {
  const rows = Array.from(selectedVariables.values());
  const countEl = document.getElementById('basket-count');
  const listEl = document.getElementById('basket-list');
  const exportBtn = document.getElementById('export-vars');
  const clearBtn = document.getElementById('clear-vars');

  countEl.textContent = String(rows.length);
  exportBtn.disabled = rows.length === 0;
  clearBtn.disabled = rows.length === 0;

  if (!rows.length) {
    listEl.innerHTML = '<div class="basket-empty">Add variables from chat results to export them.</div>';
    return;
  }

  listEl.innerHTML = rows.map(v => `
    <span class="var-chip" title="${escHtml(v.variable_id || '')}">
      <code>${escHtml(v.variable || v.variable_id)}</code>
      <button type="button" data-remove-var="${escHtml(variableKey(v))}" title="Remove">x</button>
    </span>
  `).join('');

  listEl.querySelectorAll('[data-remove-var]').forEach(btn => {
    btn.addEventListener('click', () => {
      selectedVariables.delete(btn.getAttribute('data-remove-var'));
      saveSelectedVariables();
      renderBasket();
    });
  });
}
function addVariable(v) {
  const key = variableKey(v);
  if (!key) return;
  selectedVariables.set(key, v);
  saveSelectedVariables();
  renderBasket();
}
function exportSelectedVariables() {
  const rows = Array.from(selectedVariables.values());
  if (!rows.length) return;
  const headers = [
    'variable_id', 'variable', 'table', 'label', 'description',
    'type', 'non_missing', 'topic', 'theme', 'study_context'
  ];
  const csv = [
    headers.join(','),
    ...rows.map(row => headers.map(h => csvEscape(row[h])).join(',')),
  ].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'bib-selected-variables.csv';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
function renderVariablePicker(container, variables) {
  if (!container || !variables || !variables.length) return;
  const unique = [];
  const seen = new Set();
  variables.forEach(v => {
    const key = variableKey(v);
    if (key && !seen.has(key)) {
      seen.add(key);
      unique.push(v);
    }
  });
  if (!unique.length) return;

  const picker = document.createElement('div');
  picker.className = 'detected-vars';
  picker.innerHTML = `
    <div class="detected-head">
      <span>${unique.length} variable${unique.length === 1 ? '' : 's'} found</span>
      <button type="button" class="add-all-vars">Add all</button>
    </div>
    <div class="detected-list">
      ${unique.map(v => `
        <button type="button" class="detected-var" data-var-key="${escHtml(variableKey(v))}">
          <code>${escHtml(v.variable || v.variable_id)}</code>
          <span>${escHtml(v.label || v.table || '')}</span>
        </button>
      `).join('')}
    </div>
  `;
  container.appendChild(picker);

  picker.querySelector('.add-all-vars').addEventListener('click', () => {
    unique.forEach(addVariable);
  });
  picker.querySelectorAll('[data-var-key]').forEach(btn => {
    btn.addEventListener('click', () => {
      const row = unique.find(v => variableKey(v) === btn.getAttribute('data-var-key'));
      if (row) addVariable(row);
    });
  });
}
function renderVariableStudySummary(container, result) {
  const summary = result.study_summary || [];
  if (!container || !summary.length) return;

  const terms = (result.terms || []).slice(0, 8);
  const rows = result.rows || [];
  const rowByKey = new Map(rows.map(row => [variableKey(row), row]));
  const panel = document.createElement('div');
  panel.className = 'study-summary';
  panel.innerHTML = `
    <div class="study-summary-head">
      <div>
        <strong>Variables by study/cohort</strong>
        <span>${result.total} matching variable${result.total === 1 ? '' : 's'} across ${summary.length} study context/cohort label${summary.length === 1 ? '' : 's'}</span>
      </div>
      ${terms.length ? `
        <div class="study-term-pills">
          ${terms.map(term => `<span>${escHtml(term)}</span>`).join('')}
        </div>
      ` : ''}
    </div>
    <div class="study-summary-list">
      ${summary.map((item, idx) => {
        const studyRows = rows.filter(row => row.study_context === item.study_context);
        const examples = studyRows.length
          ? studyRows
          : (item.examples || []).map(example => ({
              variable_id: example.variable_id,
              variable: example.variable_id,
              label: example.label,
            }));
        const nVars = Number(item.n_variables || 0);
        const nTables = Number(item.n_tables || 0);
        return `
          <div class="study-card">
            <div class="study-card-top">
              <div>
                <strong>${escHtml(item.study_context || 'Study not inferred')}</strong>
                <span>${nVars} variable${nVars === 1 ? '' : 's'} · ${nTables} table${nTables === 1 ? '' : 's'}</span>
              </div>
              <button type="button" data-study-index="${idx}">Add cohort</button>
            </div>
            ${examples.length ? `
              <div class="study-vars" aria-label="Variables in ${escHtml(item.study_context || 'study')}">
                ${examples.map(row => `
                  <button type="button" data-study-var-key="${escHtml(variableKey(row))}">
                    <code>${escHtml(row.variable_id || row.variable || '')}</code>
                    <span>${escHtml(row.label || row.table || '')}</span>
                  </button>
                `).join('')}
              </div>
              ${studyRows.length && studyRows.length < nVars ? `<div class="study-card-note">Showing ${studyRows.length} of ${nVars} variables for this cohort.</div>` : ''}
            ` : ''}
          </div>
        `;
      }).join('')}
    </div>
    <div class="study-summary-note">
      The full matching variable set is shown below for review and CSV export.
    </div>
  `;
  container.appendChild(panel);

  panel.querySelectorAll('[data-study-index]').forEach(btn => {
    btn.addEventListener('click', () => {
      const item = summary[Number(btn.getAttribute('data-study-index'))];
      const study = item && item.study_context;
      rows
        .filter(row => row.study_context === study)
        .forEach(addVariable);
    });
  });
  panel.querySelectorAll('[data-study-var-key]').forEach(btn => {
    btn.addEventListener('click', () => {
      const row = rowByKey.get(btn.getAttribute('data-study-var-key'));
      if (row) addVariable(row);
    });
  });
}
function renderVariableResults(container, result) {
  if (!container || !result || !result.rows || !result.rows.length) return;
  const rows = result.rows;
  const terms = (result.terms || []).slice(0, 10).join(', ');
  const filters = (result.study_filters || []).join(', ');

  const panel = document.createElement('div');
  panel.className = 'variable-results';
  panel.innerHTML = `
    <div class="variable-results-head">
      <div>
        <strong>${result.total} variable${result.total === 1 ? '' : 's'} found</strong>
        <div class="variable-results-meta">
          ${terms ? `Matched terms: ${escHtml(terms)}` : 'Matched by registry filters'}
          ${filters ? ` · Study: ${escHtml(filters)}` : ''}
          ${result.truncated ? ` · Showing ${result.returned}` : ''}
        </div>
      </div>
      <button type="button" class="add-all-results">Add all</button>
    </div>
    <div class="variable-results-list">
      ${rows.map(v => `
        <button type="button" class="variable-result" data-var-key="${escHtml(variableKey(v))}">
          <code>${escHtml(v.variable_id || v.variable)}</code>
          <span>${escHtml(v.label || v.table || '')}</span>
        </button>
      `).join('')}
    </div>
  `;
  container.appendChild(panel);

  panel.querySelector('.add-all-results').addEventListener('click', () => {
    rows.forEach(addVariable);
  });
  panel.querySelectorAll('[data-var-key]').forEach(btn => {
    btn.addEventListener('click', () => {
      const row = rows.find(v => variableKey(v) === btn.getAttribute('data-var-key'));
      if (row) addVariable(row);
    });
  });
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
  let detectedVariables = [];
  let variableResults = null;

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
        if (evt.variables) {
          detectedVariables = evt.variables;
        }
        if (evt.variable_results) {
          variableResults = evt.variable_results;
        }
        if (evt.done) {
          if (!msgEl) { removeThinking(); msgEl = appendMsg('assistant', ''); }
          if (variableResults && variableResults.rows && variableResults.rows.length) {
            if (variableResults.summary_mode === 'study_context' && variableResults.study_summary) {
              msgEl.innerHTML = '';
              renderVariableStudySummary(msgEl, variableResults);
            } else {
              msgEl.innerHTML = formatAnswer(fullText);
            }
            renderVariableResults(msgEl, variableResults);
          } else {
            msgEl.innerHTML = formatAnswer(fullText);
            renderVariablePicker(msgEl, detectedVariables);
          }
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

document.getElementById('export-vars').addEventListener('click', exportSelectedVariables);
document.getElementById('clear-vars').addEventListener('click', () => {
  selectedVariables.clear();
  saveSelectedVariables();
  renderBasket();
});
loadSelectedVariables();
renderBasket();
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
<title>BiB Variable Registry</title>
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
    <h1>BiB Variable Registry</h1>
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
      <p>
        This page implements the new metadata layer for the BiB assistant. It uses only
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
      <dt>Variable ID</dt><dd><span class="mono">${escHtml(row.variable_id || '')}</span></dd>
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


@app.route("/api/variables/search")
def variables_search_api():
    """Complete variable discovery endpoint backed by the CSV registry."""
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"error": "q is required"}), 400
    try:
        limit = max(1, min(int(request.args.get("limit", 5000)), 10000))
    except ValueError:
        limit = 5000

    result = _search_variable_registry(
        q,
        limit=limit,
        require_discovery_intent=False,
    )
    if result is None:
        result = {
            "query": q,
            "terms": [],
            "study_filters": [],
            "total": 0,
            "returned": 0,
            "truncated": False,
            "rows": [],
        }
    return jsonify(result)


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
    global current_model, current_llm_backend, current_gguf_model_path
    global current_llama_n_ctx, current_llama_n_gpu_layers
    global current_llama_chat_format, current_llama_n_threads, current_llama_verbose
    global current_transformers_device, current_transformers_dtype, current_transformers_attn_implementation
    global current_rag_n_results, current_rag_context_max_chars

    parser = argparse.ArgumentParser(description="BiB Research Assistant Web Server")
    parser.add_argument("--port",  type=int, default=5050, help="Port to listen on (default: 5050)")
    parser.add_argument("--host",  type=str, default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model for hf_api or transformers_local backend (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        choices=["hf_api", "llama_cpp", "transformers_local"],
        default=current_llm_backend,
        help=(
            "LLM backend: hf_api for Hugging Face API/endpoints, "
            "llama_cpp for local GGUF files, "
            "transformers_local for experimental non-quantized local HF models"
        ),
    )
    parser.add_argument(
        "--gguf-model-path",
        type=str,
        default=current_gguf_model_path,
        help=f"Path to a quantized GGUF model for --llm-backend llama_cpp (default: {GGUF_MODEL_DEFAULT})",
    )
    parser.add_argument("--llama-n-ctx", type=int, default=current_llama_n_ctx, help="llama.cpp context window size (default: 4096)")
    parser.add_argument("--llama-n-gpu-layers", type=int, default=current_llama_n_gpu_layers, help="llama.cpp GPU layers to offload. -1 means all supported layers")
    parser.add_argument("--llama-chat-format", type=str, default=current_llama_chat_format, help="llama.cpp chat_format (default: llama-3)")
    parser.add_argument("--llama-n-threads", type=int, default=current_llama_n_threads, help="llama.cpp CPU thread count. 0 lets llama.cpp choose")
    parser.add_argument("--llama-verbose", action="store_true", default=current_llama_verbose, help="Enable verbose llama.cpp logging")
    parser.add_argument(
        "--transformers-device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default=current_transformers_device,
        help="Device for --llm-backend transformers_local (default: auto)",
    )
    parser.add_argument(
        "--transformers-dtype",
        type=str,
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        default=current_transformers_dtype,
        help="Torch dtype for --llm-backend transformers_local (default: auto)",
    )
    parser.add_argument(
        "--transformers-attn-implementation",
        type=str,
        default=current_transformers_attn_implementation,
        help="Optional transformers_local attn_implementation, e.g. sdpa",
    )
    parser.add_argument(
        "--rag-n-results",
        type=int,
        default=current_rag_n_results,
        help="Retrieved results per collection for answer generation. 0 uses the standard default of 5",
    )
    parser.add_argument(
        "--rag-context-max-chars",
        type=int,
        default=current_rag_context_max_chars,
        help="Maximum formatted context characters sent to the LLM. 0 means unlimited",
    )
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    current_model = args.model
    current_llm_backend = (args.llm_backend or "hf_api").strip().lower()
    current_gguf_model_path = args.gguf_model_path
    current_llama_n_ctx = int(args.llama_n_ctx)
    current_llama_n_gpu_layers = int(args.llama_n_gpu_layers)
    current_llama_chat_format = args.llama_chat_format
    current_llama_n_threads = int(args.llama_n_threads)
    current_llama_verbose = bool(args.llama_verbose)
    current_transformers_device = args.transformers_device
    current_transformers_dtype = args.transformers_dtype
    current_transformers_attn_implementation = args.transformers_attn_implementation
    current_rag_n_results = int(args.rag_n_results)
    current_rag_context_max_chars = int(args.rag_context_max_chars)

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Born in Bradford — Research Assistant Server        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Docs dir  : {DOCS_DIR}")
    print(f"  ChromaDB  : {SCRIPT_DIR / '.chroma_db'}")
    print(f"  LLM backend: {current_llm_backend}")
    if current_llm_backend == "llama_cpp":
        print(f"  GGUF model : {current_gguf_model_path}")
        print(f"  RAG results: {_effective_rag_n_results()} per collection")
        print(f"  RAG context: {_effective_rag_context_max_chars()} chars max")
    elif current_llm_backend == "transformers_local":
        print(f"  HF weights : {current_model}")
        print(f"  Device/dtype: {current_transformers_device}/{current_transformers_dtype}")
        print("  Mode       : experimental non-quantized local Transformers")
        print(f"  RAG results: {_effective_rag_n_results()} per collection")
        print(f"  RAG context: {_effective_rag_context_max_chars()} chars max")
    else:
        print(f"  LLM model  : {current_model}")
    print()

    # Pre-initialise clients
    _ensure_clients()

    # Index health check
    if chroma_client:
        _check_index(chroma_client)

    if not llm_client:
        print("⚠️  Starting without LLM — /api/chat will return 503")
        if current_llm_backend == "llama_cpp":
            print("   Install llama-cpp-python and confirm --gguf-model-path exists")
        elif current_llm_backend == "transformers_local":
            print("   Install torch/transformers/accelerate/sentencepiece and confirm the model fits in memory")
        else:
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
