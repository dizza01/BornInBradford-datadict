"""
Born in Bradford - RAG-Powered Research Assistant
==================================================
Indexes five knowledge sources into ChromaDB:
  1. bib_papers_metadata.json  - 500 paper abstracts
  2. docs/csv/all_variables_meta.csv - 26k variables
  3. docs/csv/all_tables.csv   - 291 tables
  4. docs/*.html               - section groupings (closer_title)
  5. questionnaires/*.pdf      - study questionnaires and survey instruments

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

  # Run a local quantized GGUF model with llama.cpp:
  python bib_research_assistant.py --llm-backend llama_cpp --gguf-model-path models/bib-llama-3.1-8b.Q4_K_M.gguf --chat

  # Experimental: run a non-quantized local Hugging Face model with Transformers:
  python bib_research_assistant.py --llm-backend transformers_local --model meta-llama/Llama-3.2-3B-Instruct --chat

  # Recommended models (all free via HF Inference API):
  #   meta-llama/Llama-3.1-70B-Instruct    best in metrics so far
  #   Qwen/Qwen2.5-72B-Instruct            (default — best free quality)
  #   meta-llama/Llama-3.1-8B-Instruct     (good — accept licence on HF first)
  #   HuggingFaceH4/zephyr-7b-beta         (reliable, no sign-up needed)
"""

import os
import sys
import json
import re
import argparse
import math
from collections import Counter, defaultdict
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
QUESTIONNAIRES_DIR = DATADICT_DIR / "questionnaires"
MODELS_DIR   = DATADICT_DIR / "models"

# ── LLM model default ─────────────────────────────────────────────────────────
DEFAULT_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
GGUF_MODEL_DEFAULT = MODELS_DIR / "bib-llama-3.1-8b.Q4_K_M.gguf"

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


def _extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """
    Extract text page-by-page from a PDF.

    Returning page numbers lets the index preserve evidence location instead
    of flattening the whole paper into anonymous character windows.
    """
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(pdf_path))
        pages = [(idx + 1, page.get_text("text")) for idx, page in enumerate(doc)]
        doc.close()
        return [(page_no, text) for page_no, text in pages if (text or "").strip()]
    except Exception as e:
        print(f"   ⚠️  Could not read {pdf_path.name}: {e}")
        return []


def _is_pdf_path(path: Path) -> bool:
    """Return True when a file is a PDF, even if the extension is missing."""
    if path.suffix.lower() == ".pdf":
        return True
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


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


def _chunk_pdf_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = 1800,
    overlap_chars: int = 250,
) -> list[tuple[str, int, int]]:
    """Chunk PDF text while preserving page spans.

    Chunks are assembled from whole pages where possible. Large single pages
    are split locally, but metadata still records their page number.
    """
    chunks: list[tuple[str, int, int]] = []
    current_parts: list[str] = []
    current_start_page = 0
    current_end_page = 0

    def flush() -> str:
        nonlocal current_parts, current_start_page, current_end_page
        text = "\n\n".join(current_parts).strip()
        if len(text) > 50:
            chunks.append((text, current_start_page, current_end_page))
        current_parts = []
        current_start_page = 0
        current_end_page = 0
        return text[-overlap_chars:] if overlap_chars > 0 else ""

    for page_no, raw_text in pages:
        page_text = re.sub(r"\n{3,}", "\n\n", (raw_text or "").strip())
        if not page_text:
            continue

        if len(page_text) > chunk_size:
            carry = flush() if current_parts else ""
            page_chunks = _chunk_text(page_text, chunk_size=chunk_size, overlap=overlap_chars)
            for page_chunk in page_chunks:
                text = f"{carry}\n\n{page_chunk}".strip() if carry else page_chunk
                chunks.append((text, page_no, page_no))
                carry = page_chunk[-overlap_chars:] if overlap_chars > 0 else ""
            continue

        projected_len = sum(len(part) for part in current_parts) + len(page_text)
        if current_parts and projected_len > chunk_size:
            carry = flush()
            if carry:
                current_parts = [carry]
                current_start_page = page_no
                current_end_page = page_no

        if not current_parts:
            current_start_page = page_no
        current_parts.append(page_text)
        current_end_page = page_no

    if current_parts:
        flush()
    return chunks


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


def _title_from_path(path: Path) -> str:
    """Convert a local filename to a readable title without a file extension."""
    stem = path.stem if path.suffix else path.name
    cleaned = re.sub(r"[_]+", " ", stem)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or path.name


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

        pages = _extract_pdf_pages(pdf_path)
        if not pages:
            continue

        chunks = _chunk_pdf_pages(pages)
        docs, ids, metas = [], [], []
        for i, (chunk, page_start, page_end) in enumerate(chunks):
            chunk_id = f"pdf_{re.sub(r'[^a-z0-9]', '_', stem.lower()[:60])}_chunk_{i}"
            page_label = f"page {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}"
            header = (
                f"Title: {title}\n"
                f"Year: {year}\n"
                f"Source: full-text PDF\n"
                f"PDF file: {pdf_path.name}\n"
                f"Location: {page_label}\n\n"
            )
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
                "page_start": str(page_start),
                "page_end": str(page_end),
                "chunk_type": "page_window",
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


def build_questionnaires_collection(
    client: chromadb.ClientAPI,
    questionnaires_dir: Path,
):
    """Index questionnaire PDFs into the 'bib_questionnaires' collection."""
    print("\n📝 Indexing questionnaires...")

    try:
        client.delete_collection("bib_questionnaires")
    except Exception:
        pass
    collection = client.create_collection("bib_questionnaires")

    if not questionnaires_dir.exists():
        print("   ℹ️  No questionnaires directory found — skipping")
        return collection

    files = sorted(p for p in questionnaires_dir.iterdir() if p.is_file())
    pdf_files = [p for p in files if _is_pdf_path(p)]
    skipped = [p.name for p in files if not _is_pdf_path(p)]

    if skipped:
        preview = ", ".join(skipped[:5])
        suffix = " ..." if len(skipped) > 5 else ""
        print(
            "   ⚠️  Skipping non-PDF questionnaires "
            f"(convert to PDF to index): {preview}{suffix}"
        )

    if not pdf_files:
        print("   ℹ️  No questionnaire PDFs found — skipping")
        return collection

    total_chunks = 0
    for pdf_path in pdf_files:
        text = _extract_pdf_text(pdf_path)
        if not text.strip():
            continue

        title = _title_from_path(pdf_path)
        chunks = _chunk_text(text, chunk_size=1200, overlap=150)
        docs, ids, metas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = f"questionnaire_{re.sub(r'[^a-z0-9]', '_', title.lower())[:60]}_chunk_{i}"
            header = (
                f"Title: {title}\n"
                f"Source: questionnaire PDF\n"
                f"File: {pdf_path.name}\n\n"
            )
            docs.append(header + chunk)
            ids.append(chunk_id)
            metas.append({
                "title": title[:500],
                "source": "questionnaire_pdf",
                "file_name": pdf_path.name[:200],
                "chunk": str(i),
            })

        for doc_batch, id_batch, meta_batch in zip(
            _batch(docs, 500), _batch(ids, 500), _batch(metas, 500)
        ):
            collection.upsert(documents=doc_batch, ids=id_batch, metadatas=meta_batch)

        total_chunks += len(docs)
        print(f"   ✅ {pdf_path.name[:60]}  → {len(docs)} chunks")

    print(f"\n   📝 Total questionnaire chunks added: {total_chunks}")
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
            f"Variable ID: {var_id}",
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
    build_questionnaires_collection(client, QUESTIONNAIRES_DIR)

    # Load paper metadata for cross-referencing
    with open(PAPERS_JSON, encoding="utf-8") as f:
        papers_meta = json.load(f)

    # Index full text from local PDFs (adds chunks into bib_papers collection)
    index_pdf_fulltext_into_papers(client, PDFS_DIR, papers_meta)
    build_tool_references_collection(client)

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
1. If the user asks about BiB variables or tables, cite specific variable names, full variable IDs, and table IDs.
2. If the user asks about published studies, answer directly from the study context (title, year, key findings).
3. If the user asks about questionnaire wording, modules, or survey instruments, answer from questionnaire context and cite the questionnaire title where possible.
4. Note data quality issues (n_complete, cohort waves) when directly relevant to the question.
5. Be honest about limitations — if the answer is not in the context, say so clearly.


Answer only what the user asked:

- Do NOT suggest additional analyses, covariates, or “for a comprehensive analysis…” style advice unless the user explicitly asks for it.
- Do NOT introduce BiB variables, tables, or modelling steps unless the question is explicitly about variables/tables or analysis design.
- When the user asks about a specific study, focus on reporting that study’s findings in concise prose, grounded only in the retrieved context.


Context retrieved from the BiB knowledge base is provided below. Use it to ground your answer.

Important style rules:
- Never open with filler phrases such as "Certainly!", "Of course!", "Sure!", "Absolutely!", "Great question!", "Happy to help!", or similar. Begin your response directly with the substantive answer.
- Do NOT append generic boilerplate sections at the end of your response, such as "### Privacy Rules", "### Limitations", "### Note", "### Important", "### Disclaimer", or closing lines like "If you need further assistance…", "Feel free to ask!", "Let me know if…", or similar. End your answer when the content is complete.
- For non-variable questions, prefer short prose or simple bullet points. Do not use markdown tables unless the user explicitly asks for a table.
- When listing multiple variables, always include the full `variable_id` as a column. Use a compact markdown table instead of nested bullet points. Preferred format:

  | Variable ID | Variable | Table | Label | Type | N (non-missing) |
  |---|---|---|---|---|---|
  | BiB_AgeOfWonder.survey_mod232_derived_dr24.rcad_ga | rcad_ga | BiB_AgeOfWonder.survey_mod232_derived_dr24 | RCADS-25 General anxiety. Raw score | integer | 8421 |

  Omit columns that are not available. For a single variable, inline prose is fine."""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _retrieval_query_variants(query: str) -> list[str]:
    """Return query variants for common instrument spelling differences."""
    variants = [query]
    q = query or ""
    q_l = q.lower()

    # Users often write "rcad 25" while sources use RCADS-25/RCADS25.
    if re.search(r"\brcad\s*[- ]?\s*25\b", q_l):
        variants.extend([
            re.sub(r"\brcad\s*[- ]?\s*25\b", "rcads 25", q_l),
            re.sub(r"\brcad\s*[- ]?\s*25\b", "rcads-25", q_l),
            re.sub(r"\brcad\s*[- ]?\s*25\b", "rcads25", q_l),
        ])
    if re.search(r"\brcad\b", q_l):
        variants.append(re.sub(r"\brcad\b", "rcads", q_l))

    return list(dict.fromkeys(v for v in variants if v))


def _compact_alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _tokenize_with_variants(text: str) -> list[str]:
    tokens = _tokenize(text)
    expanded = list(tokens)
    token_set = set(tokens)
    if "rcad" in token_set:
        expanded.append("rcads")
    if "rcads" in token_set:
        expanded.append("rcad")
    if ("rcad" in token_set or "rcads" in token_set) and "25" in token_set:
        expanded.extend(["rcad25", "rcads25"])
    return expanded


def _rrf_fuse(rank_lists: list[list[str]], rrf_k: int) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked_ids in rank_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] += 1.0 / (rrf_k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def _exact_compact_source_boost_ids(
    query: str,
    cache: dict[str, Any],
    limit: int = 10,
) -> list[str]:
    """Boost exact compact source matches such as "rcad 25" -> RCADS25 PDFs."""
    tokens = set(_tokenize_with_variants(query))
    compact_keys = {
        token
        for token in tokens
        if len(token) >= 5 and re.search(r"[a-z]", token) and re.search(r"\d", token)
    }
    if ("rcad" in tokens or "rcads" in tokens) and "25" in tokens:
        compact_keys.update({"rcad25", "rcads25"})
    if not compact_keys:
        return []

    scored: list[tuple[str, float]] = []
    for doc_id, meta in (cache.get("meta_by_id") or {}).items():
        doc = (cache.get("doc_by_id") or {}).get(doc_id, "")
        title = str((meta or {}).get("title", ""))
        pdf_file = str((meta or {}).get("pdf_file", ""))
        source_name = f"{title} {pdf_file}"
        compact_source = _compact_alnum(source_name)
        compact_doc = _compact_alnum(doc[:2000])

        score = 0.0
        for key in compact_keys:
            if key in compact_source:
                score += 8.0
            if key in compact_doc:
                score += 2.0
        if score:
            scored.append((doc_id, score + _simple_rerank_score(query, doc)))

    return [
        doc_id
        for doc_id, _ in sorted(scored, key=lambda pair: pair[1], reverse=True)[:limit]
    ]


def _paper_context_excerpt(query: str, doc: str, meta: dict[str, Any]) -> str:
    """Return a prompt-sized paper excerpt, expanding exact item/question PDFs."""
    if "Abstract:" in doc:
        start = doc.find("Abstract:")
        return doc[start:start + int(os.getenv("PAPER_ABSTRACT_EXCERPT_CHARS", "600"))]

    source = str((meta or {}).get("source", ""))
    source_name = f"{(meta or {}).get('title', '')} {(meta or {}).get('pdf_file', '')}"
    compact_source = _compact_alnum(source_name)
    asks_for_items = bool(re.search(r"\b(item|items|question|questions|wording)\b", query or "", re.I))
    asks_for_rcads25 = "rcad25" in set(_tokenize_with_variants(query)) or "rcads25" in set(_tokenize_with_variants(query))

    if source == "pdf_fulltext" and asks_for_items and asks_for_rcads25 and "rcads25" in compact_source:
        return doc[:int(os.getenv("PAPER_EXACT_ITEM_EXCERPT_CHARS", "4200"))]

    return doc[:int(os.getenv("PAPER_FULLTEXT_EXCERPT_CHARS", "1800"))]


def _paper_or_document_source_type(meta: dict[str, Any]) -> str:
    """Return a source label that avoids treating every PDF as a paper."""
    meta = meta or {}
    title = _safe_str(meta.get("title", ""))
    pdf_file = _safe_str(meta.get("pdf_file", ""))
    source = _safe_str(meta.get("source", ""))
    source_text = f"{title} {pdf_file}".lower()

    if source == "pdf_fulltext" and re.search(
        r"\b(data summary|data dictionary|registry|questionnaire|survey instrument|readme|manual)\b",
        source_text,
    ):
        return "Documentation PDF"
    if source == "pdf_fulltext":
        return "Full-text PDF"
    return "Published paper metadata"


def _expand_pdf_sibling_chunks(
    query: str,
    doc_ids: list[str],
    cache: dict[str, Any],
    limit: int,
) -> list[str]:
    """Include neighbouring chunks from an exact PDF when the user asks for items."""
    asks_for_items = bool(re.search(r"\b(item|items|question|questions|wording)\b", query or "", re.I))
    if not asks_for_items:
        return doc_ids[:limit]

    tokens = set(_tokenize_with_variants(query))
    compact_keys = {
        token
        for token in tokens
        if len(token) >= 5 and re.search(r"[a-z]", token) and re.search(r"\d", token)
    }
    if ("rcad" in tokens or "rcads" in tokens) and "25" in tokens:
        compact_keys.update({"rcad25", "rcads25"})

    out: list[str] = []
    seen: set[str] = set()
    max_siblings = int(os.getenv("PDF_ITEM_SIBLING_CHUNKS", "4"))
    meta_by_id = cache.get("meta_by_id") or {}

    def append(doc_id: str) -> None:
        if doc_id not in seen and len(out) < limit:
            out.append(doc_id)
            seen.add(doc_id)

    for doc_id in doc_ids:
        append(doc_id)
        meta = meta_by_id.get(doc_id, {}) or {}
        pdf_file = str(meta.get("pdf_file", ""))
        source_name = f"{meta.get('title', '')} {pdf_file}"
        if (
            meta.get("source") != "pdf_fulltext"
            or not pdf_file
            or not any(key in _compact_alnum(source_name) for key in compact_keys)
        ):
            continue

        siblings = [
            sibling_id
            for sibling_id, sibling_meta in meta_by_id.items()
            if (sibling_meta or {}).get("source") == "pdf_fulltext"
            and (sibling_meta or {}).get("pdf_file") == pdf_file
        ]
        siblings = sorted(
            siblings,
            key=lambda sibling_id: int(str((meta_by_id.get(sibling_id) or {}).get("chunk", "0")) or 0),
        )
        for sibling_id in siblings[:max_siblings]:
            append(sibling_id)

    for doc_id in doc_ids:
        append(doc_id)
    return out


class _SparseBM25:
    def __init__(self, ids: list[str], docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.ids = ids
        self.k1 = k1
        self.b = b
        self.n_docs = len(ids)
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}
        self._build(docs)

    def _build(self, docs: list[str]) -> None:
        total_len = 0
        for idx, text in enumerate(docs):
            tokens = _tokenize(text)
            total_len += len(tokens)
            self.doc_len.append(len(tokens))
            counts = Counter(tokens)
            for term, tf in counts.items():
                self.postings[term].append((idx, tf))

        self.avgdl = (total_len / self.n_docs) if self.n_docs else 0.0
        for term, plist in self.postings.items():
            df = len(plist)
            self.idf[term] = math.log(1.0 + ((self.n_docs - df + 0.5) / (df + 0.5)))

    def search(self, query: str, top_n: int) -> list[str]:
        if top_n <= 0 or not query or self.n_docs == 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in _tokenize_with_variants(query):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_idx, tf in plist:
                dl = self.doc_len[doc_idx] if doc_idx < len(self.doc_len) else 0
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                scores[doc_idx] += idf * ((tf * (self.k1 + 1.0)) / (denom or 1.0))

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [self.ids[idx] for idx, _ in ranked]


def _simple_rerank_score(query: str, doc: str) -> float:
    q_tokens = _tokenize_with_variants(query)
    d_tokens = _tokenize(doc)
    if not q_tokens or not d_tokens:
        return 0.0

    q_set = set(q_tokens)
    d_set = set(d_tokens)
    overlap = len(q_set & d_set) / max(1, len(q_set))

    q_bigrams = set(zip(q_tokens, q_tokens[1:]))
    d_bigrams = set(zip(d_tokens, d_tokens[1:]))
    bigram_overlap = len(q_bigrams & d_bigrams) / max(1, len(q_bigrams)) if q_bigrams else 0.0

    contains_query = 1.0 if query.lower().strip() in doc.lower() else 0.0
    return (0.65 * overlap) + (0.25 * bigram_overlap) + (0.10 * contains_query)


_QUESTIONNAIRE_QUERY_RE = re.compile(
    r"\b(questionnaire|questionnaires|survey|surveys|module|modules|asked|ask|wording|"
    r"items?|questions?|months?|age of wonder|ague of wonder)\b",
    re.I,
)

_PAPER_QUERY_RE = re.compile(
    r"\b(paper|papers|publication|publications|published|article|articles|"
    r"study|studies|research|summari[sz]e|findings?)\b",
    re.I,
)

_METHOD_OR_SCALE_QUERY_RE = re.compile(
    r"\b(psychometric|scale|scales|screening|tool|tools|assessment|assessments|"
    r"measure|measures|method|methods|test|tests|questionnaire|instrument|instruments)\b",
    re.I,
)

_VARIABLE_OR_TABLE_QUERY_RE = re.compile(
    r"\b(variable|variables|field|fields|table|tables|dataset|registry|"
    r"available|exist|exists|occur|occurs)\b",
    re.I,
)

_ACRONYM_DEFINITION_QUERY_RE = re.compile(
    r"\b(stand(?:s)? for|meaning of|mean(?:s)?)\b",
    re.I,
)


def _context_source_plan(query: str) -> dict[str, bool]:
    """Choose evidence sources for the prompt without changing retrieval ranking.

    Paper-heavy local prompts are slow because every query was pulling every
    collection. Routing obvious paper/method turns away from unrelated
    collections reduces prompt-eval time and usually improves focus.
    """
    q = query or ""
    wants_questionnaires = bool(_QUESTIONNAIRE_QUERY_RE.search(q))
    wants_papers = bool(_PAPER_QUERY_RE.search(q))
    wants_methods = bool(_METHOD_OR_SCALE_QUERY_RE.search(q))
    wants_variables = bool(_VARIABLE_OR_TABLE_QUERY_RE.search(q))
    wants_acronym_definition = bool(_ACRONYM_DEFINITION_QUERY_RE.search(q))
    wants_tools = wants_methods

    if wants_acronym_definition and not wants_variables:
        return {"tools": True, "papers": True, "questionnaires": False, "variables": False, "tables": False}

    if wants_questionnaires and not wants_papers:
        return {"tools": wants_tools, "papers": True, "questionnaires": True, "variables": wants_variables, "tables": wants_variables}

    if wants_papers:
        return {"tools": wants_tools, "papers": True, "questionnaires": False, "variables": wants_variables, "tables": wants_variables}

    if wants_methods and not wants_variables:
        return {"tools": True, "papers": True, "questionnaires": wants_questionnaires, "variables": True, "tables": False}

    return {"tools": wants_tools, "papers": True, "questionnaires": True, "variables": True, "tables": True}

_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def _normalise_questionnaire_text(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("’", "'").replace("‘", "'")
    # Common typo seen in chat: "ague of wonder" should match Age of Wonder.
    text = re.sub(r"\bague of wonder\b", "age of wonder", text)
    for word, digit in _NUMBER_WORDS.items():
        text = re.sub(rf"\bmodule {word}\b", f"module {digit}", text)
    text = re.sub(r"\b(\d{4})[-/](\d{2})\b", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _questionnaire_token_set(text: str) -> set[str]:
    stop = {
        "the", "was", "were", "what", "asked", "ask", "in", "of", "a", "an",
        "and", "to", "for", "survey", "questionnaire", "questionnaires",
        "young", "people", "s",
    }
    return {
        token for token in _normalise_questionnaire_text(text).split()
        if token not in stop and len(token) > 1
    }


def _questionnaire_focus_score(query: str, meta: dict[str, Any]) -> float:
    q_norm = _normalise_questionnaire_text(query)
    title = str((meta or {}).get("title", ""))
    file_name = str((meta or {}).get("file_name", ""))
    title_norm = _normalise_questionnaire_text(title)
    file_norm = _normalise_questionnaire_text(file_name)
    candidate_norm = f"{title_norm} {file_norm}".strip()
    if not q_norm or not candidate_norm:
        return 0.0

    q_tokens = _questionnaire_token_set(query)
    c_tokens = _questionnaire_token_set(candidate_norm)
    overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
    score = overlap * 10.0

    if title_norm and title_norm in q_norm:
        score += 20.0
    if file_norm and file_norm.replace(" pdf", "") in q_norm:
        score += 20.0

    q_module = re.search(r"\bmodule (\d+)\b", q_norm)
    c_module = re.search(r"\bmodule (\d+)\b", candidate_norm)
    if q_module:
        score += 8.0 if c_module and q_module.group(1) == c_module.group(1) else -8.0

    q_month = re.search(r"\b(6|12|18|24|36) months?\b", q_norm)
    c_month = re.search(r"\b(6|12|18|24|36) months?\b", candidate_norm)
    if q_month:
        score += 8.0 if c_month and q_month.group(1) == c_month.group(1) else -6.0

    q_year = re.search(r"\b(20\d{2})\s+(\d{2})\b", q_norm)
    if q_year:
        year_a, year_b = q_year.groups()
        score += 6.0 if year_a in candidate_norm and year_b in candidate_norm else -4.0

    if "age of wonder" in q_norm:
        score += 5.0 if "age of wonder" in candidate_norm else -5.0

    return score


def _best_questionnaire_file(query: str, collection) -> dict[str, Any] | None:
    if not _QUESTIONNAIRE_QUERY_RE.search(query or ""):
        return None

    try:
        rows = collection.get(include=["metadatas"])
    except Exception:
        return None

    best_by_file: dict[str, dict[str, Any]] = {}
    for meta in rows.get("metadatas", []) or []:
        if not meta:
            continue
        file_name = str(meta.get("file_name", ""))
        if not file_name or file_name in best_by_file:
            continue
        score = _questionnaire_focus_score(query, meta)
        best_by_file[file_name] = {
            "file_name": file_name,
            "title": meta.get("title", ""),
            "score": score,
        }

    if not best_by_file:
        return None
    best = max(best_by_file.values(), key=lambda item: item["score"])
    return best if best["score"] >= 6.0 else None


def _chunk_number(meta: dict[str, Any]) -> int:
    try:
        return int((meta or {}).get("chunk", 0))
    except (TypeError, ValueError):
        return 0


def _get_questionnaire_file_chunks(collection, file_name: str) -> list[tuple[str, dict[str, Any]]]:
    try:
        rows = collection.get(
            where={"file_name": {"$eq": file_name}},
            include=["documents", "metadatas"],
        )
    except Exception:
        return []
    docs = rows.get("documents", []) or []
    metas = rows.get("metadatas", []) or []
    pairs = [(doc, meta or {}) for doc, meta in zip(docs, metas) if doc]
    return sorted(pairs, key=lambda pair: _chunk_number(pair[1]))


def _select_questionnaire_chunks(
    query: str,
    chunks: list[tuple[str, dict[str, Any]]],
    max_chunks: int,
) -> list[tuple[str, dict[str, Any]]]:
    if len(chunks) <= max_chunks:
        return chunks

    q_norm = _normalise_questionnaire_text(query)
    if re.search(r"\b(what was asked|what questions|what items|module|questionnaire|survey)\b", q_norm):
        # For broad wording questions, early chunks usually contain the module
        # overview and first question rows. Add a few query-relevant later chunks.
        selected = chunks[: min(6, max_chunks)]
        remaining = [pair for pair in chunks if pair not in selected]
        ranked = sorted(
            remaining,
            key=lambda pair: _simple_rerank_score(query, pair[0]),
            reverse=True,
        )
        selected.extend(ranked[: max(0, max_chunks - len(selected))])
        return sorted(selected, key=lambda pair: _chunk_number(pair[1]))

    return sorted(
        chunks,
        key=lambda pair: _simple_rerank_score(query, pair[0]),
        reverse=True,
    )[:max_chunks]


_PAPERS_CACHE: dict[str, Any] = {
    "count": -1,
    "doc_by_id": {},
    "meta_by_id": {},
    "sparse": None,
    "uppercase_terms": set(),
    "defined_acronyms": set(),
}

_QUESTIONNAIRES_CACHE: dict[str, Any] = {
    "count": -1,
    "rows": [],
    "uppercase_terms": set(),
    "defined_acronyms": set(),
}

_ACRONYM_QUERY_STOPWORDS = {
    "about", "after", "again", "age", "also", "and", "anxiety", "any", "are", "born",
    "bradford", "carried", "could", "data", "does", "for", "from",
    "have", "how", "in", "into", "is", "it", "its", "me", "mentioned",
    "of", "on", "out", "paper", "papers", "please", "publication",
    "publications", "published", "research", "show", "stand", "stands", "study",
    "studies", "survey", "surveys", "table", "tell", "the", "there", "these", "this",
    "time", "to", "use", "used", "variable", "variables", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "wonder",
    "years", "bib", "uk",
}

_UPPERCASE_TERM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,11}\b")
_DEFINITION_FIRST_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9,&/\- ]{4,120}?)\s*\(\s*([A-Z][A-Z0-9]{1,11})\s*\)"
)
_ACRONYM_FIRST_RE = re.compile(
    r"\b([A-Z][A-Z0-9]{1,11})\s*\(\s*([A-Z][A-Za-z0-9,&/\- ]{4,120}?)\s*\)"
)


def _get_papers_cache(client: chromadb.ClientAPI) -> dict[str, Any]:
    papers_col = client.get_collection("bib_papers")
    count = papers_col.count()

    if _PAPERS_CACHE["count"] == count and _PAPERS_CACHE["sparse"] is not None:
        return _PAPERS_CACHE

    all_rows = papers_col.get(include=["documents", "metadatas"])
    ids = all_rows.get("ids", []) or []
    docs = all_rows.get("documents", []) or []
    metas = all_rows.get("metadatas", []) or []

    _PAPERS_CACHE["count"] = count
    _PAPERS_CACHE["doc_by_id"] = {doc_id: doc for doc_id, doc in zip(ids, docs)}
    _PAPERS_CACHE["meta_by_id"] = {doc_id: (meta or {}) for doc_id, meta in zip(ids, metas)}
    _PAPERS_CACHE["sparse"] = _SparseBM25(ids, docs)
    _PAPERS_CACHE["uppercase_terms"] = _uppercase_terms_from_docs(docs)
    _PAPERS_CACHE["defined_acronyms"] = _defined_acronyms_from_docs(docs)
    return _PAPERS_CACHE


def _uppercase_terms_from_docs(docs: list[str]) -> set[str]:
    """Return lowercase forms of acronym-like uppercase tokens seen in docs."""
    terms: set[str] = set()
    for doc in docs:
        for match in _UPPERCASE_TERM_RE.findall(doc or ""):
            if match.lower() not in _ACRONYM_QUERY_STOPWORDS:
                terms.add(match.lower())
    return terms


def _defined_acronyms_from_docs(docs: list[str]) -> set[str]:
    """Return lowercase acronyms that appear in definition patterns."""
    terms: set[str] = set()
    for doc in docs:
        text = doc or ""
        for match in _DEFINITION_FIRST_RE.finditer(text):
            terms.add(match.group(2).lower())
        for match in _ACRONYM_FIRST_RE.finditer(text):
            terms.add(match.group(1).lower())
    return terms


def _get_questionnaires_cache(collection) -> dict[str, Any]:
    count = collection.count()
    if _QUESTIONNAIRES_CACHE["count"] == count and _QUESTIONNAIRES_CACHE["rows"]:
        return _QUESTIONNAIRES_CACHE

    rows = collection.get(include=["documents", "metadatas"])
    ids = rows.get("ids", []) or []
    docs = rows.get("documents", []) or []
    metas = rows.get("metadatas", []) or []
    cached_rows = [
        {"id": doc_id, "document": doc, "metadata": meta or {}}
        for doc_id, doc, meta in zip(ids, docs, metas)
        if doc
    ]

    _QUESTIONNAIRES_CACHE["count"] = count
    _QUESTIONNAIRES_CACHE["rows"] = cached_rows
    _QUESTIONNAIRES_CACHE["uppercase_terms"] = _uppercase_terms_from_docs(docs)
    _QUESTIONNAIRES_CACHE["defined_acronyms"] = _defined_acronyms_from_docs(docs)
    return _QUESTIONNAIRES_CACHE


def _query_acronym_candidates(query: str, known_terms: set[str]) -> list[str]:
    candidates = []
    for token in _tokenize(query):
        if not (2 <= len(token) <= 12):
            continue
        if token in _ACRONYM_QUERY_STOPWORDS:
            continue
        if token in known_terms:
            candidates.append(token)
    return list(dict.fromkeys(candidates))


def _acronym_snippet(doc: str, term: str, max_chars: int = 900) -> str:
    match = re.search(rf"\b{re.escape(term)}\b", doc or "", flags=re.IGNORECASE)
    if not match:
        return (doc or "")[:max_chars]
    start = max(0, match.start() - max_chars // 3)
    end = min(len(doc), start + max_chars)
    snippet = (doc or "")[start:end].strip()
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(doc or "") else ""
    return f"{prefix}{snippet}{suffix}"


def _acronym_doc_score(query: str, doc: str, term: str) -> float:
    text = doc or ""
    term_re = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    occurrences = len(term_re.findall(text))
    definition_bonus = 0.0
    if re.search(rf"\(\s*{re.escape(term)}\s*\)", text, flags=re.IGNORECASE):
        definition_bonus += 4.0
    if re.search(rf"\b{re.escape(term)}\s*\(", text, flags=re.IGNORECASE):
        definition_bonus += 2.0
    return (occurrences * 1.5) + definition_bonus + _simple_rerank_score(query, text)


def _exact_acronym_context(query: str, client: chromadb.ClientAPI) -> str:
    """Prepend exact paper/questionnaire snippets for known acronym-like terms.

    This handles queries where users write acronyms in lowercase ("ckat") and
    semantic retrieval would otherwise miss the explanatory definition chunk.
    """
    try:
        papers_cache = _get_papers_cache(client)
    except Exception:
        papers_cache = {"doc_by_id": {}, "meta_by_id": {}, "uppercase_terms": set(), "defined_acronyms": set()}

    try:
        questionnaires_col = client.get_collection("bib_questionnaires")
        questionnaires_cache = _get_questionnaires_cache(questionnaires_col)
    except Exception:
        questionnaires_cache = {"rows": [], "uppercase_terms": set(), "defined_acronyms": set()}

    known_terms = set(papers_cache.get("uppercase_terms", set()))
    known_terms |= set(papers_cache.get("defined_acronyms", set()))
    known_terms |= set(questionnaires_cache.get("uppercase_terms", set()))
    known_terms |= set(questionnaires_cache.get("defined_acronyms", set()))

    terms = _query_acronym_candidates(query, known_terms)
    if not terms:
        return ""

    parts = ["## Exact Acronym/Tool Matches\n"]
    max_terms = int(os.getenv("ACRONYM_BOOST_MAX_TERMS", "3"))
    max_docs_per_term = int(os.getenv("ACRONYM_BOOST_DOCS_PER_TERM", "3"))

    for term in terms[:max_terms]:
        candidates: list[tuple[float, str, str, dict[str, Any]]] = []

        for doc_id, doc in (papers_cache.get("doc_by_id", {}) or {}).items():
            if not re.search(rf"\b{re.escape(term)}\b", doc or "", flags=re.IGNORECASE):
                continue
            meta = (papers_cache.get("meta_by_id", {}) or {}).get(doc_id, {})
            candidates.append((_acronym_doc_score(query, doc, term), "paper", doc, meta))

        for row in questionnaires_cache.get("rows", []) or []:
            doc = row.get("document", "")
            if not re.search(rf"\b{re.escape(term)}\b", doc or "", flags=re.IGNORECASE):
                continue
            meta = row.get("metadata", {}) or {}
            candidates.append((_acronym_doc_score(query, doc, term), "questionnaire", doc, meta))

        ranked = []
        seen_sources: set[tuple[str, str]] = set()
        for item in sorted(candidates, key=lambda item: item[0], reverse=True):
            _, source_kind, _, meta = item
            if source_kind == "paper":
                source_key = (source_kind, str(meta.get("title", "")))
            else:
                source_key = (source_kind, str(meta.get("file_name", "") or meta.get("title", "")))
            if source_key in seen_sources:
                continue
            ranked.append(item)
            seen_sources.add(source_key)
            if len(ranked) >= max_docs_per_term:
                break
        if not ranked:
            continue

        parts.append(f"[ACRONYM MATCH: {term.upper()}]\n")
        for _, source_kind, doc, meta in ranked:
            if source_kind == "paper":
                year_suffix = f" ({meta.get('year', '')})" if meta.get("year") else ""
                source = f"Paper: {meta.get('title', '')}{year_suffix}"
            else:
                file_suffix = (
                    f" ({meta.get('file_name', '')})" if meta.get("file_name") else ""
                )
                source = f"Questionnaire: {meta.get('title', '')}{file_suffix}"
            parts.append(f"{source}\n{_acronym_snippet(doc, term)}\n")

    return "\n".join(parts).strip()


def _iter_tool_definition_matches(doc: str) -> list[tuple[str, str]]:
    """Extract (acronym, expansion) pairs from definition patterns."""
    matches: list[tuple[str, str]] = []
    text = doc or ""
    for match in _DEFINITION_FIRST_RE.finditer(text):
        expansion = re.sub(r"\s+", " ", match.group(1)).strip(" -,:;")
        acronym = match.group(2).strip()
        if acronym and expansion:
            matches.append((acronym, expansion))
    for match in _ACRONYM_FIRST_RE.finditer(text):
        acronym = match.group(1).strip()
        expansion = re.sub(r"\s+", " ", match.group(2)).strip(" -,:;")
        if acronym and expansion:
            matches.append((acronym, expansion))
    return matches


def _looks_like_tool_definition(acronym: str, expansion: str) -> bool:
    acronym_l = (acronym or "").lower()
    expansion_l = (expansion or "").lower()
    if acronym_l in _ACRONYM_QUERY_STOPWORDS:
        return False
    if not (2 <= len(acronym_l) <= 12):
        return False
    if len(expansion_l.split()) < 2:
        return False
    useful_words = {
        "assessment", "battery", "index", "inventory", "measure", "questionnaire",
        "scale", "score", "screening", "survey", "test", "tool",
    }
    return bool(useful_words & set(_tokenize(expansion_l)))


def build_tool_references_collection(client: chromadb.ClientAPI) -> int:
    """Build a compact collection of scale/tool/acronym definitions.

    The source documents remain papers/questionnaires, but this derived index
    gives method questions a short, high-signal retrieval path.
    """
    print("\n🧰 Indexing tool, scale, and acronym references...")
    try:
        client.delete_collection("bib_tool_references")
    except Exception:
        pass
    collection = client.create_collection("bib_tool_references")

    source_rows: list[tuple[str, str, dict[str, Any]]] = []
    for collection_name, source_kind in [
        ("bib_papers", "paper"),
        ("bib_questionnaires", "questionnaire"),
    ]:
        try:
            rows = client.get_collection(collection_name).get(include=["documents", "metadatas"])
        except Exception:
            continue
        docs = rows.get("documents", []) or []
        metas = rows.get("metadatas", []) or []
        for doc, meta in zip(docs, metas):
            if doc:
                source_rows.append((source_kind, doc, meta or {}))

    docs, ids, metas = [], [], []
    seen: set[tuple[str, str, str, str]] = set()
    for source_kind, source_doc, source_meta in source_rows:
        for acronym, expansion in _iter_tool_definition_matches(source_doc):
            if not _looks_like_tool_definition(acronym, expansion):
                continue
            title = _safe_str(source_meta.get("title", ""))
            file_name = _safe_str(source_meta.get("file_name", ""))
            source_key = title or file_name
            key = (acronym.lower(), expansion.lower(), source_kind, source_key.lower())
            if key in seen:
                continue
            seen.add(key)

            snippet = _acronym_snippet(source_doc, acronym, max_chars=900)
            doc = (
                f"Tool/acronym: {acronym}\n"
                f"Expansion: {expansion}\n"
                f"Source type: {source_kind}\n"
                f"Source title: {title}\n"
                f"Year: {_safe_str(source_meta.get('year', ''))}\n"
                f"Source file: {file_name}\n\n"
                f"Evidence snippet:\n{snippet}"
            )
            idx = len(ids)
            ids.append(f"tool_ref_{idx}")
            docs.append(doc)
            metas.append({
                "acronym": acronym[:50],
                "expansion": expansion[:300],
                "source": source_kind,
                "title": title[:500],
                "year": _safe_str(source_meta.get("year", "")),
                "file_name": file_name[:200],
            })

    total = 0
    for doc_batch, id_batch, meta_batch in zip(
        _batch(docs, 500), _batch(ids, 500), _batch(metas, 500)
    ):
        collection.upsert(documents=doc_batch, ids=id_batch, metadatas=meta_batch)
        total += len(doc_batch)

    print(f"   ✅ Indexed {total} tool/reference snippets")
    return total


def _retrieve_tool_reference_docs(collection, query: str, n_results: int) -> list[str]:
    """Retrieve tool refs, preferring exact acronym hits over semantic matches."""
    try:
        rows = collection.get(include=["documents", "metadatas"])
    except Exception:
        rows = {"documents": [], "metadatas": []}

    q_tokens = set(_tokenize(query))
    scored: list[tuple[float, str]] = []
    exact_scored: list[tuple[float, str]] = []
    for doc, meta in zip(rows.get("documents", []) or [], rows.get("metadatas", []) or []):
        meta = meta or {}
        acronym = _safe_str(meta.get("acronym", "")).lower()
        expansion = _safe_str(meta.get("expansion", "")).lower()
        score = _simple_rerank_score(query, doc or "")
        exact_score = 0.0
        for token in q_tokens:
            if len(token) < 2 or token in _ACRONYM_QUERY_STOPWORDS:
                continue
            if acronym and (acronym == token or acronym.startswith(token) or token.startswith(acronym)):
                score += 20.0
                exact_score += 20.0
            if expansion and _term_in_text_for_tools(token, expansion):
                score += 4.0
        if score > 0:
            scored.append((score, doc))
        if exact_score > 0:
            exact_scored.append((score, doc))

    if exact_scored:
        return [
            doc
            for _, doc in sorted(exact_scored, key=lambda item: item[0], reverse=True)
        ][:n_results]

    ranked = [doc for _, doc in sorted(scored, key=lambda item: item[0], reverse=True)]
    if ranked:
        return ranked[:n_results]

    try:
        semantic = collection.query(query_texts=[query], n_results=n_results)
        return semantic.get("documents", [[]])[0] or []
    except Exception:
        return []


def _term_in_text_for_tools(term: str, text: str) -> bool:
    if not term or not text:
        return False
    if re.search(r"[^a-z0-9]", term):
        return term in text
    return re.search(rf"\b{re.escape(term)}s?\b", text) is not None


def _exact_match_registry_lookup(query: str, client: chromadb.ClientAPI) -> str:
    """Return a '## Exact Registry Matches' context block for identifiers named in the query.

    Performs metadata-filtered exact Chroma lookups before semantic retrieval so the
    classifier sees authoritative registry evidence — both positive (entity found with
    its true properties) and negative (entity confirmed absent) — as the highest-priority
    context section.

    Covers: variable names, table IDs, paper DOIs, and quoted paper titles.
    """
    parts: list[str] = []

    # ── Entity extraction ─────────────────────────────────────────────────────
    # Table IDs:  e.g. "BiB_Metabolomics.metms_2k_r"
    table_id_re = re.compile(r'\b([A-Za-z][A-Za-z0-9_]*\.[a-z][a-z0-9_]+)\b')
    table_ids = list(dict.fromkeys(table_id_re.findall(query)))

    # Variable names: extracted after the keyword "variable" or from
    # "Is <identifier> a BiB variable" phrasing.
    var_after_kw_re = re.compile(r'\bvariable\s+([A-Za-z][A-Za-z0-9_]+)\b')
    var_is_re = re.compile(r'\bIs\s+([A-Za-z][A-Za-z0-9_]+)\s+a\s+BiB\s+variable\b', re.IGNORECASE)
    var_names = list(dict.fromkeys(var_after_kw_re.findall(query) + var_is_re.findall(query)))

    # DOIs: e.g. "10.1234/some.doi"
    doi_re = re.compile(r"10\.\d{4,}/[^\s'\".,)]+")
    dois = list(dict.fromkeys(doi_re.findall(query)))

    # Quoted paper titles: single or double quotes, 10–200 chars
    title_re = re.compile(r"['\"](.{10,200})['\"]")
    quoted_titles = list(dict.fromkeys(title_re.findall(query)))

    def _ensure_header() -> None:
        if not parts:
            parts.append("## Exact Registry Matches\n")

    # ── Variable exact lookup ─────────────────────────────────────────────────
    try:
        vars_col = client.get_collection("bib_variables")
        for var_name in var_names[:3]:
            res = vars_col.get(where={"variable": {"$eq": var_name}}, include=["documents"])
            _ensure_header()
            if res.get("documents"):
                # A variable can appear in multiple tables; show all matches so
                # the classifier can verify the exact variable–table pairing.
                for doc in res["documents"][:3]:
                    parts.append(f"[VARIABLE FOUND: {var_name}]\n```\n{doc}\n```\n")
            else:
                parts.append(
                    f"[VARIABLE NOT IN REGISTRY: '{var_name}' does not exist"
                    f" in the BiB datasphere]\n"
                )
    except Exception:
        pass

    # ── Table exact lookup ────────────────────────────────────────────────────
    try:
        tables_col = client.get_collection("bib_tables")
        for tid in table_ids[:4]:
            res = tables_col.get(where={"table_id": {"$eq": tid}}, include=["documents"])
            _ensure_header()
            if res.get("documents"):
                for doc in res["documents"][:1]:
                    parts.append(f"[TABLE FOUND: {tid}]\n```\n{doc}\n```\n")
            else:
                parts.append(
                    f"[TABLE NOT IN REGISTRY: '{tid}' does not exist"
                    f" in the BiB datasphere]\n"
                )
    except Exception:
        pass

    # ── Paper DOI exact lookup ────────────────────────────────────────────────
    try:
        papers_col = client.get_collection("bib_papers")
        for doi in dois[:2]:
            res = papers_col.get(where={"doi": {"$eq": doi}}, include=["metadatas"])
            _ensure_header()
            if res.get("metadatas"):
                meta = next(
                    (m for m in res["metadatas"] if m and m.get("source") != "pdf_fulltext"),
                    (res["metadatas"] or [{}])[0] or {},
                )
                parts.append(
                    f"[PAPER DOI FOUND: {doi}]\n"
                    f"Title: {meta.get('title', '')}\n"
                    f"Year: {meta.get('year', '')}\n\n"
                )
            else:
                parts.append(
                    f"[PAPER DOI NOT IN REGISTRY: '{doi}' not found in BiB paper index]\n"
                )
    except Exception:
        pass

    # ── Paper title exact lookup (includes PDF-chunk presence check) ──────────
    try:
        papers_col = client.get_collection("bib_papers")
        for title in quoted_titles[:2]:
            # Try exact match first; also try truncated form (stored as title[:500])
            res = papers_col.get(where={"title": {"$eq": title}}, include=["metadatas"])
            if not res.get("metadatas"):
                res = papers_col.get(
                    where={"title": {"$eq": title[:500]}}, include=["metadatas"]
                )
            _ensure_header()
            if res.get("metadatas"):
                has_pdf = any(
                    (m or {}).get("source") == "pdf_fulltext"
                    for m in res["metadatas"]
                )
                # Prefer the abstract-level record for year/doi metadata
                meta = next(
                    (m for m in res["metadatas"] if m and m.get("source") != "pdf_fulltext"),
                    (res["metadatas"] or [{}])[0] or {},
                )
                parts.append(
                    f"[PAPER TITLE FOUND: '{title[:80]}']\n"
                    f"Year: {meta.get('year', '')}\n"
                    f"DOI: {meta.get('doi', '')}\n"
                    f"Has full-text PDF chunks in index: {'yes' if has_pdf else 'no'}\n\n"
                )
            else:
                parts.append(
                    f"[PAPER TITLE NOT IN REGISTRY: '{title[:80]}'"
                    f" not found in BiB paper index]\n"
                )
    except Exception:
        pass

    return "\n".join(parts)


def retrieve_context(query: str, client: chromadb.ClientAPI, n_results: int = 5) -> str:
    """Retrieve relevant docs from the indexed collections and format as context."""
    context_parts = []
    source_plan = _context_source_plan(query)

    # ── Exact registry lookups (highest authority, prepended before semantic results) ──
    exact_ctx = _exact_match_registry_lookup(query, client)
    if exact_ctx:
        context_parts.append(exact_ctx)

    # ── Exact acronym/tool mentions from papers/questionnaires ───────────────
    acronym_ctx = _exact_acronym_context(query, client)
    if acronym_ctx:
        context_parts.append(acronym_ctx)

    # ── Tool / scale / acronym definitions ──────────────────────────────────
    if source_plan["tools"]:
        try:
            tools_col = client.get_collection("bib_tool_references")
            docs = _retrieve_tool_reference_docs(tools_col, query, max(3, n_results))
            if docs:
                context_parts.append("\n## Relevant Tool and Scale References\n")
                for doc in docs:
                    context_parts.append(f"```\n{doc[:1200]}\n```\n")
        except Exception:
            pass

    # ── Papers ───────────────────────────────────────────────────────────────
    if source_plan["papers"]:
        try:
            papers_col = client.get_collection("bib_papers")
            cache = _get_papers_cache(client)

            dense_pool = int(os.getenv("RETRIEVAL_DENSE_POOL", "60"))
            sparse_pool = int(os.getenv("RETRIEVAL_SPARSE_POOL", "60"))
            rerank_pool = int(os.getenv("RETRIEVAL_RERANK_POOL", "50"))
            rrf_k = int(os.getenv("RETRIEVAL_RRF_K", "60"))

            dense_pool = max(dense_pool, n_results)
            sparse_pool = max(sparse_pool, n_results)
            rerank_pool = max(rerank_pool, n_results)

            query_variants = _retrieval_query_variants(query)[:4]
            rank_lists: list[list[str]] = []
            for query_variant in query_variants:
                dense_results = papers_col.query(
                    query_texts=[query_variant],
                    n_results=dense_pool,
                    include=[],
                )
                dense_ids = dense_results.get("ids", [[]])[0] or []
                if dense_ids:
                    rank_lists.append(dense_ids[:dense_pool])

                sparse_ids = cache["sparse"].search(query_variant, sparse_pool)
                if sparse_ids:
                    rank_lists.append(sparse_ids[:sparse_pool])

            boosted_ids = _exact_compact_source_boost_ids(query, cache, limit=min(10, rerank_pool))
            if boosted_ids:
                rank_lists.insert(0, boosted_ids)

            fused_ids = _rrf_fuse(rank_lists, rrf_k=rrf_k)
            rerank_candidates = fused_ids[:rerank_pool]
            reranked_ids = sorted(
                rerank_candidates,
                key=lambda doc_id: _simple_rerank_score(query, cache["doc_by_id"].get(doc_id, "")),
                reverse=True,
            )[:n_results]
            reranked_ids = _expand_pdf_sibling_chunks(query, reranked_ids, cache, n_results)

            docs = [cache["doc_by_id"].get(doc_id, "") for doc_id in reranked_ids]
            metas = [cache["meta_by_id"].get(doc_id, {}) for doc_id in reranked_ids]

            if any(docs):
                context_parts.append("## Relevant Papers and Documents\n")
                for doc, meta in zip(docs, metas):
                    if not doc:
                        continue
                    title = (meta or {}).get("title", "")
                    authors = (meta or {}).get("authors", "")
                    source_type = _paper_or_document_source_type(meta or {})
                    pdf_file = (meta or {}).get("pdf_file", "")
                    file_suffix = f"\nFile: {pdf_file}" if pdf_file else ""
                    context_parts.append(
                        f"Source type: {source_type}{file_suffix}\n"
                        f"**{title}** "
                        f"({meta.get('year','')}) — {authors[:160]}\n"
                        f"{_paper_context_excerpt(query, doc, meta)}\n"
                    )
        except Exception as e:
            context_parts.append(f"[Papers collection unavailable: {e}]\n")

    # ── Questionnaires ───────────────────────────────────────────────────────
    if source_plan["questionnaires"]:
      try:
        questionnaires_col = client.get_collection("bib_questionnaires")
        questionnaire_parts: list[str] = []
        targeted_file = _best_questionnaire_file(query, questionnaires_col)
        targeted_file_name = targeted_file.get("file_name", "") if targeted_file else ""
        seen_questionnaire_keys: set[tuple[str, str]] = set()

        if targeted_file_name:
            all_chunks = _get_questionnaire_file_chunks(questionnaires_col, targeted_file_name)
            max_file_chunks = int(os.getenv("QUESTIONNAIRE_FILE_CHUNKS", "10"))
            target_chars = int(os.getenv("QUESTIONNAIRE_TARGET_CHARS", "1800"))
            selected_chunks = _select_questionnaire_chunks(query, all_chunks, max_file_chunks)
            if selected_chunks:
                title = targeted_file.get("title") or targeted_file_name
                questionnaire_parts.append(
                    "## Targeted Questionnaire Match\n"
                    f"Matched questionnaire: **{title}** ({targeted_file_name})\n"
                    f"Match score: {targeted_file.get('score', 0):.1f}\n"
                )
                for doc, meta in selected_chunks:
                    chunk = str((meta or {}).get("chunk", ""))
                    seen_questionnaire_keys.add((targeted_file_name, chunk))
                    questionnaire_parts.append(
                        f"### {title} — chunk {chunk}\n"
                        f"{doc[:target_chars]}\n"
                    )

        semantic_n = max(n_results, int(os.getenv("QUESTIONNAIRE_SEMANTIC_RESULTS", "8")))
        results = questionnaires_col.query(query_texts=[query], n_results=semantic_n)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        for doc, meta in zip(docs, metas):
            meta = meta or {}
            file_name = meta.get("file_name", "")
            chunk = str(meta.get("chunk", ""))
            key = (file_name, chunk)
            if key in seen_questionnaire_keys:
                continue
            title = meta.get("title", "")
            questionnaire_parts.append(
                f"**{title}**"
                f"{f' ({file_name})' if file_name else ''}\n"
                f"{doc[:1200]}\n"
            )
            seen_questionnaire_keys.add(key)
            if len(seen_questionnaire_keys) >= semantic_n + (10 if targeted_file_name else 0):
                break

        if questionnaire_parts:
            context_parts.append("\n## Relevant Questionnaires\n")
            context_parts.extend(questionnaire_parts)
      except Exception as e:
        context_parts.append(f"[Questionnaires collection unavailable: {e}]\n")

    # ── Variables ────────────────────────────────────────────────────────────
    if source_plan["variables"]:
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
    if source_plan["tables"]:
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


_ANCHOR_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "been",
    "being",
    "between",
    "could",
    "does",
    "from",
    "have",
    "into",
    "more",
    "most",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "used",
    "uses",
    "using",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "were",
    "would",
    "study",
    "cohort",
    "questionnaire",
    "questionnaires",
}


def _extract_question_anchors(question: str, max_anchors: int = 14) -> list[str]:
    """Extract exact terms that should steer row/sentence selection."""
    anchors: list[str] = []

    def add(anchor: str) -> None:
        cleaned = anchor.strip(" \t\n\r\"'`.,;:()[]{}")
        if not cleaned:
            return
        key = cleaned.lower()
        if key not in {a.lower() for a in anchors}:
            anchors.append(cleaned)

    for phrase in re.findall(r"['\"]([^'\"]{2,80})['\"]", question):
        add(phrase)

    for token in re.findall(r"\b\d+(?:\.\d+)?%?\b|\b[A-Za-z][A-Za-z0-9_]{2,}\b", question):
        lower = token.lower()
        if lower in _ANCHOR_STOPWORDS:
            continue
        if len(token) < 4 and not re.search(r"\d", token):
            continue
        add(token)

    return anchors[:max_anchors]


def _split_context_units(context: str) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    section = "context"

    for raw_line in context.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            section = line.lstrip("#").strip()
            continue
        if line in {"```", "---"}:
            continue

        line = re.sub(r"^\s*[-*]\s+", "", line)
        if len(line) > 420:
            parts = re.split(r"(?<=[.!?])\s+", line)
        else:
            parts = [line]
        for part in parts:
            text = part.strip()
            if len(text) >= 20:
                units.append({"section": section, "text": text})

    return units


def _anchor_score(text: str, anchors: list[str]) -> tuple[float, int]:
    lower = text.lower()
    matched = 0
    score = 0.0

    for anchor in anchors:
        a = anchor.lower()
        if not a:
            continue
        if re.search(rf"(?<![a-z0-9_]){re.escape(a)}(?![a-z0-9_])", lower):
            matched += 1
            if re.search(r"\d", a):
                score += 2.0
            elif " " in a or "_" in a:
                score += 1.6
            else:
                score += 1.0

    length_penalty = min(len(text) / 1200.0, 0.5)
    return score - length_penalty, matched


def format_context_with_question_anchors(
    question: str,
    context: str,
    max_evidence_lines: int = 8,
    max_nearby_lines: int = 6,
) -> str:
    """Prepend a question-anchor guide while preserving the original context."""
    anchors = _extract_question_anchors(question)
    if not anchors:
        return context

    scored: list[dict[str, Any]] = []
    for unit in _split_context_units(context):
        score, matched = _anchor_score(unit["text"], anchors)
        if matched:
            scored.append({**unit, "score": score, "matched": matched})

    if not scored:
        return context

    scored.sort(key=lambda item: (item["score"], item["matched"]), reverse=True)
    evidence = scored[:max_evidence_lines]
    evidence_texts = {item["text"] for item in evidence}
    nearby = [
        item
        for item in scored[max_evidence_lines:]
        if item["text"] not in evidence_texts and item["matched"] < len(anchors)
    ][:max_nearby_lines]

    parts: list[str] = [
        "## Question-Anchored Evidence Guide",
        "Use these extracted anchors to choose the exact matching row or sentence. "
        "The guide is derived only from the retrieved context below.",
        "Do not cite match/near labels in the final answer; use the evidence content only.",
        "",
        "Question anchors to match:",
    ]
    parts.extend(f"- {anchor}" for anchor in anchors)

    parts.append("\nMatching evidence rows or sentences:")
    for i, item in enumerate(evidence, start=1):
        parts.append(f"- [match_{i} | {item['section']}] {item['text']}")

    if nearby:
        parts.append("\nDistractor rows or sentences to ignore because one or more anchors differ:")
        for i, item in enumerate(nearby, start=1):
            parts.append(f"- [near_{i} | {item['section']}] {item['text']}")

    parts.extend(["", "## Original Retrieved Context", context])
    return "\n".join(parts)


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

_FOLLOWUP_RE = re.compile(
    r"\b("
    r"this|that|the|it|above|previous|paper|study|article|summari[sz]e|explain|tell me more"
    r")\b",
    re.IGNORECASE,
)


def _retrieval_query_from_history(question: str, history: list | None) -> str:
    """Add recent turns to short follow-up queries so retrieval keeps the subject."""
    if not history or len(question.split()) > 8 or not _FOLLOWUP_RE.search(question):
        return question

    recent = []
    for turn in reversed(history[-4:]):
        content = str((turn or {}).get("content", "")).strip()
        if content:
            recent.append(content[:700])
        if len(recent) >= 3:
            break
    if not recent:
        return question
    return f"{question}\n\nRecent conversation context:\n" + "\n".join(reversed(recent))


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
    system_prompt: str = SYSTEM_PROMPT,
    retrieval_n_results: int = 5,
    max_context_chars: int = 0,
):
    """
    Streaming version of query(). Yields raw token strings as they are generated
    by the configured LLM backend, enabling the server to forward them as SSE
    events so the UI can display the response token-by-token instead of waiting
    for the full response.

    The first batch of output is buffered (~80 chars) so opener filler phrases
    can be stripped before anything reaches the client.

    Falls back to a single yield of the full non-streamed answer when the client
    object doesn't expose a streaming interface.
    """
    hf = getattr(llm_client, "_hf_raw", None)
    local_stream = getattr(llm_client, "stream_chat_completion", None)
    if hf is None and local_stream is None:
        # Fallback: yield the complete answer in one chunk.
        yield query(
            question,
            client,
            llm_client,
            model=model,
            history=history,
            system_prompt=system_prompt,
            retrieval_n_results=retrieval_n_results,
            max_context_chars=max_context_chars,
        )
        return

    prior = history or []
    retrieval_query = _retrieval_query_from_history(question, prior)
    raw_context = retrieve_context(retrieval_query, client, n_results=max(1, retrieval_n_results))
    context = format_context_with_question_anchors(question, raw_context)
    if max_context_chars and len(context) > max_context_chars:
        context = (
            context[:max_context_chars].rstrip()
            + "\n\n[Context truncated for local inference. Ask a narrower follow-up if more detail is needed.]"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        *prior,
        {
            "role": "user",
            "content": (
                f"Retrieved knowledge base context:\n\n{context}\n\n"
                f"---\n\nResearcher question: {question}"
            ),
        },
    ]

    try:
        if local_stream is not None:
            token_stream = local_stream(messages=messages, temperature=0.2, max_tokens=900)
        else:
            hf_stream = hf.chat_completion(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=900,
                stream=True,
            )

            def _hf_token_stream():
                for chunk in hf_stream:
                    if not chunk.choices:
                        continue
                    token = (chunk.choices[0].delta.content or "")
                    if token:
                        yield token
            token_stream = _hf_token_stream()

        # Buffer opening tokens to strip any filler opener before first display
        prefix_buf = ""
        prefix_sent = False
        OPENER_THRESHOLD = 80

        for token in token_stream:
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
          history: list | None = None,
          system_prompt: str = SYSTEM_PROMPT,
          retrieval_n_results: int = 5,
          max_context_chars: int = 0) -> str:
    """Run a RAG query: retrieve context → call HuggingFace LLM → return answer."""

    prior = history or []
    retrieval_query = _retrieval_query_from_history(question, prior)
    raw_context = retrieve_context(retrieval_query, client, n_results=max(1, retrieval_n_results))
    context = format_context_with_question_anchors(question, raw_context)
    if max_context_chars and len(context) > max_context_chars:
        context = (
            context[:max_context_chars].rstrip()
            + "\n\n[Context truncated for local inference. Ask a narrower follow-up if more detail is needed.]"
        )

    if show_context:
        print("\n── Retrieved Context ──────────────────────────────────────────")
        print(raw_context[:3000])
        print("──────────────────────────────────────────────────────────────\n")

    messages = [
        {"role": "system", "content": system_prompt},
        *prior,
        {
            "role": "user",
            "content": (
                f"Retrieved knowledge base context:\n\n{context}\n\n"
                f"---\n\nResearcher question: {question}"
            ),
        },
    ]

    # Custom handler mode: endpoint expects inputs={system, context, question} at the root.
    #
    # How to use:
    #   python bib_research_assistant.py \
    #     --endpoint-url https://...endpoints.huggingface.cloud \
    #     --endpoint-mode handler_structured \
    #     --endpoint-temperature 0.0 \
    #     --endpoint-max-new-tokens 900 \
    #     --query "..."
    #
    # Payload shape:
    #   POST <endpoint-url>
    #   {
    #     "inputs": {"system": "...", "context": "...", "question": "..."},
    #     "parameters": {...}
    #   }
    #   -> expects JSON with `generated_text`.
    #
    # Note: history is not included in this structured payload.
    if getattr(llm_client, "endpoint_mode", "") == "handler_structured" and hasattr(
        llm_client, "handler_structured_generate"
    ):
        answer = llm_client.handler_structured_generate(
            system=system_prompt,
            context=context,
            question=question,
        )
        answer = _strip_filler(str(answer or ""))
        return answer

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
        try:
            questionnaires_col = client.get_collection("bib_questionnaires")
            n_questionnaires = questionnaires_col.count()
        except Exception:
            n_questionnaires = 0
        try:
            tool_refs_col = client.get_collection("bib_tool_references")
            n_tool_refs = tool_refs_col.count()
        except Exception:
            n_tool_refs = 0
        n_abstracts = cols['bib_papers'] - n_pdf
        print(f"✅ Index ready — {n_abstracts} abstracts + {n_pdf} PDF chunks | "
              f"{cols['bib_variables']} variables | {cols['bib_tables']} tables | "
              f"{n_questionnaires} questionnaire chunks | {n_tool_refs} tool refs")
        return True
    except Exception as e:
        print(f"❌ Could not read index: {e}")
        return False


def _resolve_model_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    # Docker/server startup is often from the repo root, so use that as the
    # stable fallback for missing paths and error messages.
    return (DATADICT_DIR / path).resolve()


def _get_llama_cpp_client(
    gguf_model_path: str | Path = GGUF_MODEL_DEFAULT,
    llama_n_ctx: int = 4096,
    llama_n_gpu_layers: int = -1,
    llama_chat_format: str = "llama-3",
    llama_n_threads: int = 0,
    llama_verbose: bool = False,
) -> Optional[Any]:
    """Return a llama.cpp GGUF client with the same chat.completions API shape."""
    from types import SimpleNamespace

    model_path = _resolve_model_path(gguf_model_path or GGUF_MODEL_DEFAULT)
    if not model_path.exists():
        print(f"❌ GGUF model not found: {model_path}")
        print("   Create it with llm_poc/tools/quantize_to_gguf.sh or pass --gguf-model-path.")
        return None

    try:
        from llama_cpp import Llama
    except ImportError:
        print("❌ llama-cpp-python not installed.")
        print("   CPU install: pip install llama-cpp-python")
        print('   Apple Silicon Metal: CMAKE_ARGS="-DGGML_METAL=on" pip install -U llama-cpp-python --no-cache-dir')
        return None

    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": int(llama_n_ctx),
        "n_gpu_layers": int(llama_n_gpu_layers),
        "chat_format": str(llama_chat_format or "llama-3"),
        "verbose": bool(llama_verbose),
    }
    if int(llama_n_threads or 0) > 0:
        kwargs["n_threads"] = int(llama_n_threads)

    print(f"🧠 Loading local GGUF model: {model_path}")
    llm = Llama(**kwargs)
    default_top_p = float(os.getenv("LLAMA_CPP_TOP_P", "0.9"))
    default_repeat_penalty = float(os.getenv("LLAMA_CPP_REPEAT_PENALTY", "1.12"))

    class _LlamaCppChatCompletions:
        def create(self, model, messages, temperature=0.2, max_tokens=1500, **kw):
            result = llm.create_chat_completion(
                messages=messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                top_p=float(kw.get("top_p", default_top_p)),
                repeat_penalty=float(kw.get("repeat_penalty", default_repeat_penalty)),
            )
            choices = result.get("choices") or []
            if not choices:
                raise RuntimeError(f"No choices in llama.cpp response: {result}")
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            )

    class _LlamaCppChat:
        def __init__(self):
            self.completions = _LlamaCppChatCompletions()

    class _LlamaCppWrapper:
        def __init__(self):
            self.chat = _LlamaCppChat()
            self._hf_raw = None
            self.endpoint_mode = "llama_cpp"
            self.gguf_model_path = str(model_path)

        def stream_chat_completion(self, messages, temperature=0.2, max_tokens=900, **kw):
            stream = llm.create_chat_completion(
                messages=messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                top_p=float(kw.get("top_p", default_top_p)),
                repeat_penalty=float(kw.get("repeat_penalty", default_repeat_penalty)),
                stream=True,
            )
            for chunk in stream:
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] or {}
                delta = choice.get("delta") or {}
                token = delta.get("content")
                if token is None:
                    token = choice.get("text", "")
                if token:
                    yield str(token)

    return _LlamaCppWrapper()


def _get_transformers_local_client(
    model: str,
    transformers_device: str = "auto",
    transformers_dtype: str = "auto",
    transformers_attn_implementation: str = "",
) -> Optional[Any]:
    """Return an experimental local Transformers client for small comparison runs."""
    from types import SimpleNamespace

    token = os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_TOKEN", "") or None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    except ImportError:
        print("❌ torch/transformers not installed.")
        print("   Install: pip install torch transformers accelerate sentencepiece")
        return None

    model_name = str(model or "").strip()
    if not model_name:
        print("❌ --model is required for --llm-backend transformers_local.")
        return None

    device_pref = (transformers_device or "auto").strip().lower()
    dtype_pref = (transformers_dtype or "auto").strip().lower()

    dtype: Any = "auto"
    if dtype_pref in {"float16", "fp16"}:
        dtype = torch.float16
    elif dtype_pref in {"bfloat16", "bf16"}:
        dtype = torch.bfloat16
    elif dtype_pref in {"float32", "fp32"}:
        dtype = torch.float32

    load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if token:
        load_kwargs["token"] = token
    if transformers_attn_implementation:
        load_kwargs["attn_implementation"] = transformers_attn_implementation

    use_device_map = False
    target_device = "cpu"
    if device_pref == "auto":
        if torch.cuda.is_available():
            use_device_map = True
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            target_device = "mps"
        else:
            target_device = "cpu"
    elif device_pref in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            print("❌ Requested CUDA for transformers_local, but CUDA is not available.")
            return None
        use_device_map = True
    elif device_pref == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            print("❌ Requested MPS for transformers_local, but MPS is not available.")
            return None
        target_device = "mps"
    elif device_pref == "cpu":
        target_device = "cpu"
    else:
        print(f"❌ Unknown --transformers-device value: {transformers_device}")
        return None

    if use_device_map:
        load_kwargs["device_map"] = "auto"

    print(f"🧪 Loading experimental local Transformers model: {model_name}")
    print(f"   device={device_pref} dtype={dtype_pref}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model_obj = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if not use_device_map:
            model_obj.to(target_device)
        model_obj.eval()
    except Exception as exc:
        print(f"❌ Could not load local Transformers model '{model_name}': {exc}")
        return None

    def _prompt_from_messages(messages: list[dict[str, str]]) -> str:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        rendered = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            rendered.append(f"{role.upper()}: {content}")
        rendered.append("ASSISTANT:")
        return "\n\n".join(rendered)

    def _inputs_for(messages: list[dict[str, str]]):
        prompt = _prompt_from_messages(messages)
        inputs = tokenizer(prompt, return_tensors="pt")
        if not use_device_map:
            inputs = {key: val.to(target_device) for key, val in inputs.items()}
        return inputs

    def _generation_kwargs(temperature: float, max_tokens: int, **kw) -> dict[str, Any]:
        temp = float(temperature)
        return {
            "max_new_tokens": int(max_tokens),
            "do_sample": temp > 0,
            "temperature": max(temp, 1e-5),
            "top_p": float(kw.get("top_p", os.getenv("TRANSFORMERS_LOCAL_TOP_P", "0.9"))),
            "repetition_penalty": float(
                kw.get("repetition_penalty", os.getenv("TRANSFORMERS_LOCAL_REPETITION_PENALTY", "1.08"))
            ),
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

    class _TransformersLocalChatCompletions:
        def create(self, model, messages, temperature=0.2, max_tokens=1500, **kw):
            inputs = _inputs_for(messages)
            input_len = int(inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                output = model_obj.generate(
                    **inputs,
                    **_generation_kwargs(temperature, max_tokens, **kw),
                )
            generated = output[0][input_len:]
            content = tokenizer.decode(generated, skip_special_tokens=True).strip()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            )

    class _TransformersLocalChat:
        def __init__(self):
            self.completions = _TransformersLocalChatCompletions()

    class _TransformersLocalWrapper:
        def __init__(self):
            self.chat = _TransformersLocalChat()
            self._hf_raw = None
            self.endpoint_mode = "transformers_local"
            self.model_name = model_name

        def stream_chat_completion(self, messages, temperature=0.2, max_tokens=900, **kw):
            import threading

            inputs = _inputs_for(messages)
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generation_kwargs = {
                **inputs,
                **_generation_kwargs(temperature, max_tokens, **kw),
                "streamer": streamer,
            }
            thread = threading.Thread(target=model_obj.generate, kwargs=generation_kwargs)
            thread.start()
            for token in streamer:
                if token:
                    yield token
            thread.join()

    return _TransformersLocalWrapper()


def _get_hf_client(
    model: str,
    endpoint_url: str = "",
    endpoint_mode: str = "chat_completions",
    handler_parameters: Optional[dict[str, Any]] = None,
    backend: str = "hf_api",
    gguf_model_path: str | Path = GGUF_MODEL_DEFAULT,
    llama_n_ctx: int = 4096,
    llama_n_gpu_layers: int = -1,
    llama_chat_format: str = "llama-3",
    llama_n_threads: int = 0,
    llama_verbose: bool = False,
    transformers_device: str = "auto",
    transformers_dtype: str = "auto",
    transformers_attn_implementation: str = "",
) -> Optional[Any]:
    backend = (backend or "hf_api").strip().lower()
    if backend == "llama_cpp":
        return _get_llama_cpp_client(
            gguf_model_path=gguf_model_path,
            llama_n_ctx=llama_n_ctx,
            llama_n_gpu_layers=llama_n_gpu_layers,
            llama_chat_format=llama_chat_format,
            llama_n_threads=llama_n_threads,
            llama_verbose=llama_verbose,
        )
    if backend == "transformers_local":
        return _get_transformers_local_client(
            model=model,
            transformers_device=transformers_device,
            transformers_dtype=transformers_dtype,
            transformers_attn_implementation=transformers_attn_implementation,
        )
    token = os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_TOKEN", "")
    if not token:
        print("⚠️  HF_TOKEN not set.")
        print("   export HF_TOKEN='hf_...'  or add it to .env")
        print("   Get a free token at: https://huggingface.co/settings/tokens")
        return None
    from types import SimpleNamespace
    import urllib.request
    import urllib.error

    endpoint_url = (endpoint_url or "").strip().rstrip("/")
    endpoint_mode = (endpoint_mode or "").strip() or "chat_completions"
    handler_parameters = dict(handler_parameters or {})
    # When endpoint_url is set:
    # - endpoint_mode=chat_completions: call <endpoint_url>/v1/chat/completions (OpenAI-compatible)
    # - endpoint_mode=handler_structured: POST to <endpoint_url> with inputs={system,context,question}
    # For public Inference API (no endpoint_url), we use HF's chat_completion.
    #
    # NOTE: handler_structured does not require huggingface_hub because it posts directly using urllib.
    client = None
    if not endpoint_url or endpoint_mode == "chat_completions":
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
            return None
        client = InferenceClient(token=token)

    def _endpoint_chat_completions(
        endpoint: str,
        model_name: str,
        messages: list,
        temperature: float,
        max_tokens: int,
    ):
        url = f"{endpoint}/v1/chat/completions"
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if model_name:
            payload["model"] = model_name

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            raise RuntimeError(
                f"Client error '{exc.code} {exc.reason}' for url '{url}': {err_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Endpoint request failed for url '{url}': {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from endpoint: {raw[:500]}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices in endpoint response: {data}")
        msg = (choices[0].get("message") or {})
        content = (msg.get("content") or "")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=str(content)))],
        )

    def _endpoint_handler_structured(
        endpoint: str,
        system: str,
        context: str,
        question: str,
        parameters: dict[str, Any],
    ) -> str:
        # This matches common HF custom handler schemas that accept a structured `inputs` dict
        # and return `generated_text`.
        url = endpoint
        payload: dict[str, Any] = {
            "inputs": {
                "system": str(system),
                "context": str(context),
                "question": str(question),
            },
            "parameters": dict(parameters or {}),
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            raise RuntimeError(
                f"Client error '{exc.code} {exc.reason}' for url '{url}': {err_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Endpoint request failed for url '{url}': {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from endpoint: {raw[:500]}") from exc

        if isinstance(data, dict) and isinstance(data.get("generated_text"), str):
            return str(data["generated_text"]).strip()
        if (
            isinstance(data, list)
            and data
            and isinstance(data[0], dict)
            and isinstance(data[0].get("generated_text"), str)
        ):
            return str(data[0]["generated_text"]).strip()

        raise RuntimeError(
            f"Unexpected handler response shape (no generated_text): {type(data).__name__}"
        )

    # Wrap in a namespace so the call site (client.chat.completions.create)
    # stays identical to the OpenAI SDK.
    class _ChatCompletions:
        def __init__(self, hf, endpoint: str):
            self._hf = hf
            self._endpoint = endpoint

        def create(self, model, messages, temperature=0.2, max_tokens=1500, **kw):
            # For endpoints, use the OpenAI-compatible route.
            if self._endpoint:
                return _endpoint_chat_completions(
                    endpoint=self._endpoint,
                    model_name=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            if not self._hf:
                raise RuntimeError(
                    "huggingface_hub is required for public Inference API calls (no --endpoint-url)."
                )

            # For the public HF Inference API, use chat_completion.
            return self._hf.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
    class _Chat:
        def __init__(self, hf, endpoint: str):
            self.completions = _ChatCompletions(hf, endpoint)
    class _Wrapper:
        def __init__(self, hf, endpoint: str, endpoint_mode: str, handler_parameters: dict[str, Any]):
            self.chat = _Chat(hf, endpoint)
            # Exposed for streaming via query_stream(). Endpoints are treated as non-streaming
            # in this script (query_stream() will fall back to non-streamed completion).
            self._hf_raw = None if endpoint else hf
            self.endpoint_mode = endpoint_mode
            self._endpoint = endpoint
            self._handler_parameters = dict(handler_parameters or {})

        def handler_structured_generate(self, system: str, context: str, question: str) -> str:
            if not self._endpoint:
                raise RuntimeError("handler_structured_generate requires --endpoint-url")

            # `params` become the handler payload's `parameters` dict.
            # Defaults are aligned with the script's usual generation settings.
            params = {
                "use_system_role": bool(self._handler_parameters.get("use_system_role", False)),
                "do_sample": bool(self._handler_parameters.get("do_sample", False)),
                "temperature": float(self._handler_parameters.get("temperature", 0.2)),
                "top_p": float(self._handler_parameters.get("top_p", 1.0)),
                "repetition_penalty": float(self._handler_parameters.get("repetition_penalty", 1.05)),
                "no_repeat_ngram_size": int(self._handler_parameters.get("no_repeat_ngram_size", 0)),
                "max_new_tokens": int(self._handler_parameters.get("max_new_tokens", 900)),
                "max_input_tokens": int(self._handler_parameters.get("max_input_tokens", 4096)),
                "debug": bool(self._handler_parameters.get("debug", False)),
            }
            return _endpoint_handler_structured(
                endpoint=self._endpoint,
                system=system,
                context=context,
                question=question,
                parameters=params,
            )
    return _Wrapper(client, endpoint_url, endpoint_mode=endpoint_mode, handler_parameters=handler_parameters)


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
        help=f"Model name for hf_api or transformers_local backend (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--llm-backend",
        type=str,
        choices=["hf_api", "llama_cpp", "transformers_local"],
        default=os.getenv("LLM_BACKEND", "hf_api"),
        help=(
            "LLM backend: hf_api for Hugging Face API/endpoints, "
            "llama_cpp for local GGUF files, "
            "transformers_local for experimental non-quantized local HF models"
        ),
    )
    parser.add_argument(
        "--gguf-model-path",
        type=str,
        default=os.getenv("GGUF_MODEL_PATH", str(GGUF_MODEL_DEFAULT)),
        help=f"Path to a quantized GGUF model for --llm-backend llama_cpp (default: {GGUF_MODEL_DEFAULT})",
    )
    parser.add_argument(
        "--llama-n-ctx",
        type=int,
        default=int(os.getenv("LLAMA_CPP_N_CTX", "4096")),
        help="llama.cpp context window size for GGUF backend (default: 4096)",
    )
    parser.add_argument(
        "--llama-n-gpu-layers",
        type=int,
        default=int(os.getenv("LLAMA_CPP_N_GPU_LAYERS", "-1")),
        help="llama.cpp GPU layers to offload. Use -1 for all supported layers (default: -1)",
    )
    parser.add_argument(
        "--llama-chat-format",
        type=str,
        default=os.getenv("LLAMA_CPP_CHAT_FORMAT", "llama-3"),
        help="llama.cpp chat_format for GGUF backend (default: llama-3)",
    )
    parser.add_argument(
        "--llama-n-threads",
        type=int,
        default=int(os.getenv("LLAMA_CPP_N_THREADS", "0")),
        help="llama.cpp CPU thread count. 0 lets llama.cpp choose (default: 0)",
    )
    parser.add_argument(
        "--llama-verbose",
        action="store_true",
        default=os.getenv("LLAMA_CPP_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"},
        help="Enable verbose llama.cpp logging",
    )
    parser.add_argument(
        "--transformers-device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default=os.getenv("TRANSFORMERS_LOCAL_DEVICE", "auto"),
        help="Device for --llm-backend transformers_local (default: auto)",
    )
    parser.add_argument(
        "--transformers-dtype",
        type=str,
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        default=os.getenv("TRANSFORMERS_LOCAL_DTYPE", "auto"),
        help="Torch dtype for --llm-backend transformers_local (default: auto)",
    )
    parser.add_argument(
        "--transformers-attn-implementation",
        type=str,
        default=os.getenv("TRANSFORMERS_LOCAL_ATTN_IMPLEMENTATION", ""),
        help="Optional transformers_local attn_implementation, e.g. sdpa",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default="",
        help=(
            "Optional Hugging Face Inference Endpoint base URL (OpenAI-compatible). "
            "If set, requests are sent to <endpoint-url>/v1/chat/completions instead of the public Inference API."
        ),
    )

    parser.add_argument(
        "--endpoint-mode",
        type=str,
        choices=["chat_completions", "handler_structured"],
        default="chat_completions",
        help=(
            "How to call the endpoint when --endpoint-url is set. "
            "chat_completions uses /v1/chat/completions; handler_structured POSTs to the endpoint root with inputs={system,context,question}."
        ),
    )
    # Handler parameter passthrough (only used for --endpoint-mode handler_structured).
    # These map directly into the JSON payload under the `parameters` key.
    parser.add_argument("--endpoint-max-new-tokens", type=int, default=900)
    parser.add_argument("--endpoint-temperature", type=float, default=0.2)
    parser.add_argument("--endpoint-top-p", type=float, default=1.0)
    parser.add_argument("--endpoint-do-sample", action="store_true")
    parser.add_argument("--endpoint-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--endpoint-no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--endpoint-max-input-tokens", type=int, default=4096)
    parser.add_argument("--endpoint-use-system-role", action="store_true")
    parser.add_argument("--endpoint-debug", action="store_true")
    args = parser.parse_args()
    args.llm_backend = (args.llm_backend or "hf_api").strip().lower()

    if args.build:
        build_index()
        return

    if args.query or args.chat:
        client = get_chroma_client()
        if not _check_index(client):
            return

        endpoint_handler_parameters: dict[str, Any] = {
            "max_new_tokens": int(args.endpoint_max_new_tokens),
            "temperature": float(args.endpoint_temperature),
            "top_p": float(args.endpoint_top_p),
            "do_sample": bool(args.endpoint_do_sample),
            "repetition_penalty": float(args.endpoint_repetition_penalty),
            "no_repeat_ngram_size": int(args.endpoint_no_repeat_ngram_size),
            "max_input_tokens": int(args.endpoint_max_input_tokens),
            "use_system_role": bool(args.endpoint_use_system_role),
            "debug": bool(args.endpoint_debug),
        }

        llm_client = _get_hf_client(
            args.model,
            endpoint_url=args.endpoint_url,
            endpoint_mode=args.endpoint_mode,
            handler_parameters=endpoint_handler_parameters,
            backend=args.llm_backend,
            gguf_model_path=args.gguf_model_path,
            llama_n_ctx=args.llama_n_ctx,
            llama_n_gpu_layers=args.llama_n_gpu_layers,
            llama_chat_format=args.llama_chat_format,
            llama_n_threads=args.llama_n_threads,
            llama_verbose=args.llama_verbose,
            transformers_device=args.transformers_device,
            transformers_dtype=args.transformers_dtype,
            transformers_attn_implementation=args.transformers_attn_implementation,
        )

        if not llm_client:
            return

        if args.context:
            os.environ["SHOW_CONTEXT"] = "1"

        if args.query:
            print(f"\n🔬 Query: {args.query}")
            if args.llm_backend == "llama_cpp":
                print(f"   GGUF model: {args.gguf_model_path}")
            elif args.llm_backend == "transformers_local":
                print(f"   Local Transformers model: {args.model}")
            else:
                print(f"   Model: {args.model}")
            print(f"   Backend: {args.llm_backend}\n")
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
