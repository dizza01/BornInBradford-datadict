"""
Born in Bradford - RAG-Powered Research Assistant
==================================================
Indexes three knowledge sources into ChromaDB:
  1. bib_papers_metadata.json  - 500 paper abstracts
  2. docs/csv/all_variables_meta.csv - 26k variables
  3. docs/csv/all_tables.csv   - 291 tables
  4. docs/*.html               - section groupings (closer_title)

Usage:
  # Build index (one-time, ~2-5 mins):
  python bib_research_assistant.py --build

  # Interactive chat:
  python bib_research_assistant.py --chat

  # Single query:
  python bib_research_assistant.py --query "What variables measure anxiety in Age of Wonder?"

  # Set your HuggingFace token (free at huggingface.co/settings/tokens):
  export HF_TOKEN="hf_..."

  # Chat with default model (Mistral-7B):
  python bib_research_assistant.py --chat

  # Use a different model:
  python bib_research_assistant.py --model "meta-llama/Llama-3.1-8B-Instruct" --chat
  python bib_research_assistant.py --model "microsoft/Phi-3-mini-4k-instruct" --query "What is BiB1000?"

  # Recommended models (all free via HF Inference API):
  #   Qwen/Qwen2.5-72B-Instruct            (default — best free quality)
  #   meta-llama/Llama-3.1-8B-Instruct     (good — accept licence on HF first)
  #   HuggingFaceH4/zephyr-7b-beta         (reliable, no sign-up needed)
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Any, Optional

# ── Load .env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# ── Dependency check ───────────────────────────────────────────────────────────
def check_deps():
    missing = []
    for pkg in ["chromadb", "pandas"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import chromadb
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
DATADICT_DIR = SCRIPT_DIR.parent                                 # BornInBradford-datadict/
PAPERS_JSON  = DATADICT_DIR / "papers" / "bib_papers_metadata.json"
CSV_DIR      = DATADICT_DIR / "docs" / "csv"
HTML_DIR     = DATADICT_DIR / "docs"
CHROMA_DIR   = SCRIPT_DIR / ".chroma_db"

TABLES_CSV    = CSV_DIR / "all_tables.csv"
VARIABLES_CSV = CSV_DIR / "all_variables_meta.csv"
PDFS_DIR      = DATADICT_DIR / "papers"

# ── LLM model default ─────────────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# ── ChromaDB setup ─────────────────────────────────────────────────────────────
def get_chroma_client():
    CHROMA_DIR.mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: Parse HTML files for section context (closer_title)
# ══════════════════════════════════════════════════════════════════════════════

def parse_html_sections() -> dict:
    """
    Extract per-variable metadata from the embedded Reactable JSON blobs in
    every data dictionary HTML file.  The main table blob contains parallel
    arrays for each variable:

      variable[]       - variable name  (e.g. 'rcad_ga')
      label[]          - full human-readable description
                         (e.g. 'RCADS-25 General anxiety. Raw score')
      closer_title[]   - topic/section heading  (e.g. 'Mental health')

    Capturing all three means every variable gets a rich description in its
    embedding text, eliminating ambiguities like 'rcad_ga' vs 'dental_ga'.

    Returns:
        { stem: { variable_name: {"section": closer_title,
                                   "description": label} } }
    """
    print("📄 Parsing HTML files for variable descriptions and section context...")
    sections: dict = {}
    html_files = list(HTML_DIR.glob("*.html"))

    for html_path in html_files:
        stem = html_path.stem

        try:
            raw = html_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        script_blocks = re.findall(r"<script[^>]*>(.*?)</script>", raw, re.DOTALL)
        for block in script_blocks:
            if '"variable":' not in block or '"closer_title":' not in block:
                continue
            try:
                data  = json.loads(block)
                inner = data.get("x", {}).get("tag", {}).get("attribs", {}).get("data", {})
                variables = inner.get("variable", [])       # variable name
                titles    = inner.get("closer_title", [])   # topic / section
                labels    = inner.get("label", [])           # full description
                if not variables or not titles or len(variables) != len(titles):
                    continue
                var_map: dict = {}
                for j, var in enumerate(variables):
                    var_map[var] = {
                        "section":     (titles[j] or "").strip(),
                        "description": (labels[j] if j < len(labels) else "") or "",
                    }
                sections[stem] = var_map
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue

    print(f"   ✅ Parsed variable metadata from {len(sections)} HTML files")
    return sections


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: Build ChromaDB index
# ══════════════════════════════════════════════════════════════════════════════

def _safe_str(val) -> str:
    """Convert any value to a clean string, handling NaN/None."""
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return ""
    return str(val).strip()


def _batch(lst: list, size: int):
    """Yield successive batches from lst."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def build_papers_collection(client: chromadb.ClientAPI, papers_path: Path):
    """Index paper abstracts into the 'bib_papers' collection."""
    print("\n📚 Indexing paper abstracts...")

    try:
        client.delete_collection("bib_papers")
    except Exception:
        pass
    collection = client.create_collection("bib_papers")

    with open(papers_path, encoding="utf-8") as f:
        papers = json.load(f)

    docs, ids, metas = [], [], []
    for i, p in enumerate(papers):
        title    = _safe_str(p.get("title", ""))
        abstract = _safe_str(p.get("abstract", ""))
        if not title and not abstract:
            continue

        text = f"Title: {title}\n\nAbstract: {abstract}"
        docs.append(text)
        ids.append(f"paper_{i}")
        metas.append({
            "title":   title[:500],
            "year":    _safe_str(p.get("year", "")),
            "authors": _safe_str(p.get("authors", ""))[:300],
            "doi":     _safe_str(p.get("doi", "")),
            "journal": _safe_str(p.get("journal", "")),
        })

    # Add in batches of 500
    total = 0
    for doc_batch, id_batch, meta_batch in zip(
        _batch(docs, 500), _batch(ids, 500), _batch(metas, 500)
    ):
        collection.add(documents=doc_batch, ids=id_batch, metadatas=meta_batch)
        total += len(doc_batch)

    print(f"   ✅ Indexed {total} papers")
    return collection


# ══════════════════════════════════════════════════════════════════════════════
#  PDF Full-Text Extraction Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract all text from a PDF using PyMuPDF.
    Falls back to an empty string on any error.
    """
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        print(f"   ⚠️  Could not read {pdf_path.name}: {e}")
        return ""


def _chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks of ~chunk_size characters.
    Returns a list of non-empty chunk strings.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]  # drop tiny tail fragments


def _title_from_filename(stem: str) -> str:
    """
    Convert a PDF filename stem to a human-readable title.
    e.g. 'Born_in_Bradford_s_Age_of_Wonder_cohort__2024' → 'Born in Bradford s Age of Wonder cohort'
    """
    # Strip trailing year like _2024 or __2024
    cleaned = re.sub(r'[_\s]*\d{4}$', '', stem)
    # Replace underscores with spaces, collapse multiples
    cleaned = re.sub(r'_+', ' ', cleaned).strip()
    return cleaned


def _year_from_filename(stem: str) -> str:
    """Extract a 4-digit year from the end of a filename stem."""
    m = re.search(r'(\d{4})$', stem)
    return m.group(1) if m else ""


def index_pdf_fulltext_into_papers(
    client: chromadb.ClientAPI,
    pdfs_dir: Path,
    papers_metadata: list[dict],
) -> int:
    """
    Extract full text from all PDFs in pdfs_dir, chunk it, and upsert the
    chunks into the existing 'bib_papers' collection.

    Tries to cross-reference each PDF with the metadata JSON by title
    similarity so chunks inherit year / authors / doi where possible.

    Returns the total number of chunks added.
    """
    pdf_files = sorted(pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        print("   ℹ️  No PDF files found — skipping full-text indexing")
        return 0

    print(f"\n📑 Indexing full text from {len(pdf_files)} PDFs...")

    # Build a quick lowercase title → metadata lookup for cross-referencing
    meta_lookup: dict[str, dict] = {
        p.get("title", "").lower()[:80]: p
        for p in papers_metadata
        if p.get("title")
    }

    try:
        collection = client.get_collection("bib_papers")
    except Exception:
        collection = client.create_collection("bib_papers")

    total_chunks = 0
    for pdf_path in pdf_files:
        stem  = pdf_path.stem
        title = _title_from_filename(stem)
        year  = _year_from_filename(stem)

        # Cross-reference with metadata JSON (first 80 chars of title, case-insensitive)
        meta = meta_lookup.get(title.lower()[:80], {})
        authors = _safe_str(meta.get("authors", ""))
        doi     = _safe_str(meta.get("doi", ""))
        journal = _safe_str(meta.get("journal", ""))
        if not year:
            year = _safe_str(meta.get("year", ""))

        full_text = _extract_pdf_text(pdf_path)
        if not full_text.strip():
            continue

        chunks = _chunk_text(full_text)
        docs, ids, metas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = f"pdf_{re.sub(r'[^a-z0-9]', '_', stem.lower()[:60])}_chunk_{i}"
            header   = f"Title: {title}\nYear: {year}\nSource: full-text PDF\n\n"
            docs.append(header + chunk)
            ids.append(chunk_id)
            metas.append({
                "title":   title[:500],
                "year":    year,
                "authors": authors[:300],
                "doi":     doi,
                "journal": journal[:200],
                "source":  "pdf_fulltext",
                "pdf_file": pdf_path.name[:200],
                "chunk":   str(i),
            })

        # Upsert in batches (handles re-runs without duplicate IDs)
        for doc_batch, id_batch, meta_batch in zip(
            _batch(docs, 500), _batch(ids, 500), _batch(metas, 500)
        ):
            collection.upsert(documents=doc_batch, ids=id_batch, metadatas=meta_batch)
        total_chunks += len(docs)
        print(f"   ✅ {pdf_path.name[:60]}  → {len(docs)} chunks")

    print(f"\n   📑 Total PDF chunks added: {total_chunks}")
    return total_chunks


def build_tables_collection(client: chromadb.ClientAPI, tables_path: Path):
    """Index table descriptions into the 'bib_tables' collection."""
    print("\n🗂  Indexing table descriptions...")

    try:
        client.delete_collection("bib_tables")
    except Exception:
        pass
    collection = client.create_collection("bib_tables")

    df = pd.read_csv(tables_path)

    docs, ids, metas = [], [], []
    for _, row in df.iterrows():
        table_id    = _safe_str(row.get("table_id", ""))
        display     = _safe_str(row.get("display_name", ""))
        project     = _safe_str(row.get("project_name", ""))
        table_name  = _safe_str(row.get("table_name", ""))
        entity_type = _safe_str(row.get("entity_type", ""))
        data_subs   = _safe_str(row.get("data_subjects", ""))
        cohort      = _safe_str(row.get("cohort_membership", ""))
        n_vars      = _safe_str(row.get("n_variables", ""))
        n_rows      = _safe_str(row.get("n_rows", ""))
        n_entities  = _safe_str(row.get("n_entities", ""))
        updated     = _safe_str(row.get("last_updated", ""))

        text = (
            f"Table: {table_id}\n"
            f"Display name: {display}\n"
            f"Project: {project}\n"
            f"Entity type: {entity_type}\n"
            f"Data subjects: {data_subs}\n"
            f"Cohort: {cohort}\n"
            f"Variables: {n_vars} | Rows: {n_rows} | Entities: {n_entities}\n"
            f"Last updated: {updated}"
        )
        docs.append(text)
        ids.append(f"table_{table_id.replace('.', '_')}")
        metas.append({
            "table_id":   table_id[:200],
            "project":    project[:100],
            "table_name": table_name[:100],
            "n_variables": n_vars,
            "n_rows":     n_rows,
        })

    collection.add(documents=docs, ids=ids, metadatas=metas)
    print(f"   ✅ Indexed {len(docs)} tables")
    return collection


def build_variables_collection(
    client: chromadb.ClientAPI,
    variables_path: Path,
    html_sections: dict,
):
    """Index all variables into the 'bib_variables' collection."""
    print("\n🔬 Indexing variables (this may take a minute)...")

    try:
        client.delete_collection("bib_variables")
    except Exception:
        pass
    collection = client.create_collection("bib_variables")

    df = pd.read_csv(variables_path)

    # Build HTML stem → table_id lookup
    # HTML stem format: bib_ageofwonder_survey_mod02_dr23
    # table_id format:  BiB_AgeOfWonder.survey_mod02_dr23
    # We'll use the table_name part (after last '.') to match
    table_name_to_sections: dict = {}
    for stem, var_map in html_sections.items():
        # stem usually has table_name as suffix after last '_'
        # e.g. bib_ageofwonder_survey_mod02_dr23 → survey_mod02_dr23
        # We store the full stem → var_map, joined later by table column
        table_name_to_sections[stem] = var_map

    docs, ids, metas = [], [], []

    for i, row in df.iterrows():
        var_id    = _safe_str(row.get("variable_id", ""))
        table_id  = _safe_str(row.get("table_id", ""))
        project   = _safe_str(row.get("project", ""))
        table_nm  = _safe_str(row.get("table", ""))
        variable  = _safe_str(row.get("variable", ""))
        label     = _safe_str(row.get("label", ""))
        val_type  = _safe_str(row.get("value_type", ""))
        categories = _safe_str(row.get("categories", ""))
        topic     = _safe_str(row.get("topic", ""))
        n_complete = _safe_str(row.get("n_complete", ""))

        # Look up section and full description from HTML (best-effort)
        # html stem pattern: bib_{project_lower}_{table_name}
        project_lower   = project.lower().replace("_", "")
        html_stem_guess = f"bib_{project_lower}_{table_nm}"
        html_info       = table_name_to_sections.get(html_stem_guess, {}).get(variable, {})
        section         = html_info.get("section", "")     if isinstance(html_info, dict) else ""
        html_desc       = html_info.get("description", "") if isinstance(html_info, dict) else ""

        # Build rich text for embedding
        parts = [
            f"Table: {table_id}",
            f"Variable: {variable}",
            f"Label: {label}",
        ]
        # Add the HTML description when it carries extra detail not in the CSV label
        # e.g. 'RCADS-25 General anxiety. Raw score' vs 'RCADS-25 GA Raw score'
        if html_desc and html_desc.lower() != label.lower():
            parts.append(f"Description: {html_desc}")
        if topic:
            parts.append(f"Topic: {topic}")
        if section:
            parts.append(f"Section: {section}")
        if val_type:
            parts.append(f"Type: {val_type}")
        if categories:
            # Truncate long category lists
            cats = categories[:400]
            parts.append(f"Categories/Values: {cats}")
        if n_complete:
            parts.append(f"Non-missing records: {n_complete}")

        text = "\n".join(parts)
        docs.append(text)
        ids.append(f"var_{i}")
        metas.append({
            "variable_id": var_id[:200],
            "table_id":    table_id[:200],
            "project":     project[:100],
            "table_name":  table_nm[:100],
            "variable":    variable[:100],
            "topic":       topic[:100],
            "value_type":  val_type[:50],
        })

    # Add in batches of 2000 (ChromaDB can handle large batches)
    total = 0
    batch_size = 2000
    for b_docs, b_ids, b_metas in zip(
        _batch(docs, batch_size),
        _batch(ids, batch_size),
        _batch(metas, batch_size),
    ):
        collection.add(documents=b_docs, ids=b_ids, metadatas=b_metas)
        total += len(b_docs)
        sys.stdout.write(f"\r   Indexed {total}/{len(docs)} variables...")
        sys.stdout.flush()

    print(f"\n   ✅ Indexed {total} variables")
    return collection


def build_index():
    """Run the full indexing pipeline."""
    print("╔══════════════════════════════════════════════════════╗")
    print("║  BiB Research Assistant — Building Knowledge Index   ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # Validate source files
    for path, name in [
        (PAPERS_JSON, "Papers JSON"),
        (TABLES_CSV, "Tables CSV"),
        (VARIABLES_CSV, "Variables CSV"),
    ]:
        if not path.exists():
            print(f"❌ {name} not found at: {path}")
            sys.exit(1)

    client = get_chroma_client()

    # Parse HTML section context
    html_sections = parse_html_sections()

    # Build all collections
    build_papers_collection(client, PAPERS_JSON)
    build_tables_collection(client, TABLES_CSV)
    build_variables_collection(client, VARIABLES_CSV, html_sections)

    # Load paper metadata for cross-referencing
    with open(PAPERS_JSON, encoding="utf-8") as f:
        papers_meta = json.load(f)

    # Index full text from local PDFs (adds chunks into bib_papers collection)
    index_pdf_fulltext_into_papers(client, PDFS_DIR, papers_meta)

    print(f"\n✅ Index built and saved to: {CHROMA_DIR}")
    print("   Run --chat or --query to start querying.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: RAG Query Engine
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert research assistant for the Born in Bradford (BiB) longitudinal cohort study.
You help researchers understand the dataset, find relevant variables, plan analyses, and understand what has already been published.

The BiB study involves:
- ~13,000 pregnancies recruited 2007-2010 in Bradford, UK
- Multi-ethnic cohort (White British, Pakistani, other)
- Longitudinal follow-up: pregnancy → birth → BiB1000 (6-36m) → school age → Age of Wonder (adolescence) → Growing Up
- Key linkages: maternity records, education, NHS health records, environmental data, omics (genetics, methylation, metabolomics)
- Key identifier: BiBPersonID links individuals across all tables — NEVER include this in results (privacy)

Data structure:
- Tables named as Project.table_name (e.g. BiB_AgeOfWonder.survey_mod02_dr23)
- Joined via BiBPersonID (privacy-sensitive — use COUNT/aggregate, never SELECT)
- Projects include: BiB_CohortInfo, BiB_Baseline, BiB_1000, BiB_AgeOfWonder, BiB_GrowingUp, BiB_Geographic, BiB_Biosamples, BiB_Metabolomics

When answering:
1. Cite specific variable names and table IDs when recommending variables
2. Reference published papers when relevant (include title and year)
3. Note data quality issues (n_complete, cohort waves) when relevant
4. Suggest appropriate covariates based on published BiB methodology
5. Flag privacy rules: never SELECT individual identifiers
6. Be honest about limitations — if data may not exist, say so

Context retrieved from the BiB knowledge base is provided below. Use it to ground your answer.

Important style rules:
- Never open with filler phrases such as "Certainly!", "Of course!", "Sure!", "Absolutely!", "Great question!", "Happy to help!", or similar. Begin your response directly with the substantive answer.
- Do NOT append generic boilerplate sections at the end of your response, such as "### Privacy Rules", "### Limitations", "### Note", "### Important", "### Disclaimer", or closing lines like "If you need further assistance…", "Feel free to ask!", "Let me know if…", or similar. End your answer when the content is complete.
- When listing multiple variables, use a compact markdown table instead of nested bullet points. Preferred format:

  | Variable | Table | Label | Type | N (non-missing) |
  |---|---|---|---|---|
  | rcad_ga | BiB_AgeOfWonder.survey_mod232_derived_dr24 | RCADS-25 General anxiety. Raw score | integer | 8421 |

  Omit columns that are not available. For a single variable, inline prose is fine."""


def retrieve_context(query: str, client: chromadb.ClientAPI, n_results: int = 5) -> str:
    """Retrieve relevant docs from all three collections and format as context."""
    context_parts = []

    # ── Papers ───────────────────────────────────────────────────────────────
    try:
        papers_col = client.get_collection("bib_papers")
        results = papers_col.query(query_texts=[query], n_results=n_results)
        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        if docs:
            context_parts.append("## Relevant Published Papers\n")
            for doc, meta in zip(docs, metas):
                context_parts.append(
                    f"**{meta.get('title','')[:120]}** "
                    f"({meta.get('year','')}) — {meta.get('authors','')[:80]}\n"
                    f"{doc[doc.find('Abstract:'):doc.find('Abstract:')+600] if 'Abstract:' in doc else doc[:400]}\n"
                )
    except Exception as e:
        context_parts.append(f"[Papers collection unavailable: {e}]\n")

    # ── Variables ────────────────────────────────────────────────────────────
    try:
        vars_col = client.get_collection("bib_variables")
        results = vars_col.query(query_texts=[query], n_results=n_results * 2)
        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        if docs:
            context_parts.append("\n## Relevant Variables\n")
            for doc, meta in zip(docs, metas):
                context_parts.append(f"```\n{doc}\n```\n")
    except Exception as e:
        context_parts.append(f"[Variables collection unavailable: {e}]\n")

    # ── Tables ───────────────────────────────────────────────────────────────
    try:
        tables_col = client.get_collection("bib_tables")
        results = tables_col.query(query_texts=[query], n_results=n_results)
        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        if docs:
            context_parts.append("\n## Relevant Tables\n")
            for doc, meta in zip(docs, metas):
                context_parts.append(f"```\n{doc}\n```\n")
    except Exception as e:
        context_parts.append(f"[Tables collection unavailable: {e}]\n")

    return "\n".join(context_parts)


_FILLER_RE = re.compile(
    r"^(?:"
    r"Certainly[!,.]?\s*|"
    r"Of\s+course[!,.]?\s*|"
    r"Sure[!,.]?\s*|"
    r"Absolutely[!,.]?\s*|"
    r"Great\s+question[!,.]?\s*|"
    r"Happy\s+to\s+help[!,.]?\s*|"
    r"I['\u2019]d\s+be\s+happy\s+to[^.!]*[.!]?\s*|"
    r"I['\u2019]m\s+happy\s+to\s+help[^.!]*[.!]?\s*|"
    r"I['\u2019]ll\s+help\s+you\s+with\s+that[^.!]*[.!]?\s*|"
    r"Glad\s+(?:you\s+asked|to\s+help)[!,.]?\s*|"
    r"Thank\s+you\s+for\s+(?:your\s+)?question[^.!]*[.!]?\s*"
    r")+",
    re.IGNORECASE,
)

# Boilerplate footer sections the model sometimes appends unprompted.
# Matched from the section heading to end-of-string.
_FOOTER_RE = re.compile(
    r"\n+"
    r"(?:"
    # markdown headings for common boilerplate sections
    r"#{1,4}\s*(?:Privacy\s+Rules?|Limitations?|Important\s+(?:Notes?|Considerations?)|Notes?|Disclaimer|Caveats?)[^\n]*\n"
    r"|"
    # closing filler sentences
    r"(?:If\s+you\s+(?:need|have|want)|Feel\s+free\s+to|Let\s+me\s+know\s+if|Don['\u2019]t\s+hesitate|Please\s+(?:let\s+me\s+know|feel\s+free)|Hope\s+this\s+helps)[^\n]*"
    r").*$",
    re.IGNORECASE | re.DOTALL,
)

def _strip_filler(text: str) -> str:
    """Remove hollow opener phrases and boilerplate footer sections."""
    text = _FILLER_RE.sub("", text).lstrip()
    text = _FOOTER_RE.sub("", text).rstrip()
    return text


def _strip_opener(text: str) -> str:
    """Strip only opener filler — used on the streaming prefix buffer."""
    return _FILLER_RE.sub("", text).lstrip()


def query_stream(
    question: str,
    client: chromadb.ClientAPI,
    llm_client: Any,
    model: str = DEFAULT_MODEL,
    history: list | None = None,
):
    """
    Streaming version of query(). Yields raw token strings as they are generated
    by the HuggingFace model, enabling the server to forward them as SSE events
    so the UI can display the response token-by-token instead of waiting for the
    full response.

    The first batch of output is buffered (~80 chars) so opener filler phrases
    can be stripped before anything reaches the client.

    Falls back to a single yield of the full non-streamed answer when the client
    object doesn't expose the raw HF interface.
    """
    context = retrieve_context(question, client)
    prior = history or []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *prior,
        {
            "role": "user",
            "content": (
                f"Retrieved knowledge base context:\n\n{context}\n\n"
                f"---\n\nResearcher question: {question}"
            ),
        },
    ]

    hf = getattr(llm_client, "_hf_raw", None)
    if hf is None:
        # Fallback: yield the complete answer in one chunk
        yield query(question, client, llm_client, model=model, history=history)
        return

    try:
        stream = hf.chat_completion(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=900,
            stream=True,
        )
        # Buffer opening tokens to strip any filler opener before first display
        prefix_buf = ""
        prefix_sent = False
        OPENER_THRESHOLD = 80

        for chunk in stream:
            if not chunk.choices:
                continue
            token = (chunk.choices[0].delta.content or "")
            if not token:
                continue
            if not prefix_sent:
                prefix_buf += token
                if len(prefix_buf) >= OPENER_THRESHOLD:
                    cleaned = _strip_opener(prefix_buf)
                    prefix_sent = True
                    if cleaned:
                        yield cleaned
            else:
                yield token

        # Flush buffer if stream ended before the threshold was reached
        if not prefix_sent and prefix_buf:
            cleaned = _strip_opener(prefix_buf)
            if cleaned:
                yield cleaned

    except Exception as e:
        yield f"\n[Stream error: {e}]"


def query(question: str, client: chromadb.ClientAPI, llm_client: Any,
          model: str = DEFAULT_MODEL, show_context: bool = False,
          history: list | None = None) -> str:
    """Run a RAG query: retrieve context → call HuggingFace LLM → return answer."""

    context = retrieve_context(question, client)

    if show_context:
        print("\n── Retrieved Context ──────────────────────────────────────────")
        print(context[:3000])
        print("──────────────────────────────────────────────────────────────\n")

    prior = history or []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *prior,
        {
            "role": "user",
            "content": (
                f"Retrieved knowledge base context:\n\n{context}\n\n"
                f"---\n\nResearcher question: {question}"
            ),
        },
    ]

    response = llm_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=900,
    )
    answer = response.choices[0].message.content
    # Strip common filler openers that some models insist on producing
    answer = _strip_filler(answer)
    return answer


def _check_index(client: chromadb.ClientAPI) -> bool:
    """Return True if all required collections exist and are populated."""
    try:
        cols = {c.name: c.count() for c in client.list_collections()}
        required = ["bib_papers", "bib_variables", "bib_tables"]
        missing = [c for c in required if c not in cols or cols[c] == 0]
        if missing:
            print(f"⚠️  Missing/empty collections: {missing}")
            print("    Run: python bib_research_assistant.py --build")
            return False
        # Count how many entries are PDF full-text chunks vs abstract entries
        try:
            paper_col = client.get_collection("bib_papers")
            pdf_chunks = paper_col.get(where={"source": "pdf_fulltext"}, include=[])
            n_pdf = len(pdf_chunks.get("ids", []))
        except Exception:
            n_pdf = 0
        n_abstracts = cols['bib_papers'] - n_pdf
        print(f"✅ Index ready — {n_abstracts} abstracts + {n_pdf} PDF chunks | "
              f"{cols['bib_variables']} variables | {cols['bib_tables']} tables")
        return True
    except Exception as e:
        print(f"❌ Could not read index: {e}")
        return False


def _get_hf_client(model: str) -> Optional[Any]:
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        return None
    token = os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_TOKEN", "")
    if not token:
        print("⚠️  HF_TOKEN not set.")
        print("   export HF_TOKEN='hf_...'  or add it to .env")
        print("   Get a free token at: https://huggingface.co/settings/tokens")
        return None
    client = InferenceClient(token=token)
    # Wrap in a namespace so the call site (client.chat.completions.create)
    # stays identical to the OpenAI SDK
    class _ChatCompletions:
        def __init__(self, hf): self._hf = hf
        def create(self, model, messages, temperature=0.2, max_tokens=1500, **kw):
            return self._hf.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
    class _Chat:
        def __init__(self, hf): self.completions = _ChatCompletions(hf)
    class _Wrapper:
        def __init__(self, hf):
            self.chat = _Chat(hf)
            self._hf_raw = hf  # exposed for streaming via query_stream()
    return _Wrapper(client)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def chat_loop(client: chromadb.ClientAPI, llm_client: Any,
              model: str = DEFAULT_MODEL):
    show_ctx = os.getenv("SHOW_CONTEXT", "").lower() in ("1", "true", "yes")

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Born in Bradford — AI Research Assistant            ║")
    print("║  Type 'exit' to quit | 'context on/off' to toggle   ║")
    print(f"║  Model: {model:<44}║")
    print("╚══════════════════════════════════════════════════════╝\n")

    while True:
        try:
            user_input = input("🔬 Ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "context on":
            show_ctx = True
            print("Context display: ON")
            continue
        if user_input.lower() == "context off":
            show_ctx = False
            print("Context display: OFF")
            continue

        print("\n⏳ Thinking...\n")
        try:
            answer = query(user_input, client, llm_client, model=model, show_context=show_ctx)
            print("─" * 70)
            print(answer)
            print("─" * 70 + "\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="BiB RAG Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--build",   action="store_true", help="Build the ChromaDB index from source files")
    parser.add_argument("--chat",    action="store_true", help="Start interactive chat session")
    parser.add_argument("--query",   type=str,            help="Run a single query and exit")
    parser.add_argument("--context", action="store_true", help="Show retrieved context alongside answer")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    if args.build:
        build_index()
        return

    if args.query or args.chat:
        client = get_chroma_client()
        if not _check_index(client):
            return

        llm_client = _get_hf_client(args.model)

        if not llm_client:
            return

        if args.context:
            os.environ["SHOW_CONTEXT"] = "1"

        if args.query:
            print(f"\n🔬 Query: {args.query}")
            print(f"   Model: {args.model}\n")
            print("⏳ Thinking...\n")
            answer = query(
                args.query, client, llm_client,
                model=args.model, show_context=args.context
            )
            print("─" * 70)
            print(answer)
            print("─" * 70)
        elif args.chat:
            chat_loop(client, llm_client, model=args.model)
        return

    # No args — print help
    parser.print_help()
    print("\n💡 Quick start:")
    print("  export HF_TOKEN='hf_...'")
    print("  python bib_research_assistant.py --chat")
    print(f"\n  Default model: {DEFAULT_MODEL}")
    print("  Other models:  meta-llama/Llama-3.1-8B-Instruct")
    print("                 HuggingFaceH4/zephyr-7b-beta\n")


if __name__ == "__main__":
    main()
