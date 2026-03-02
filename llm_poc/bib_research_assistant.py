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
    Extract variable → section_title mapping from embedded Reactable JSON blobs
    in the data dictionary HTML files.

    Returns: { table_id: { variable_name: section_title } }
    """
    print("📄 Parsing HTML files for section context...")
    sections: dict = {}
    html_files = list(HTML_DIR.glob("*.html"))

    for html_path in html_files:
        # Derive table_id from filename: bib_ageofwonder_survey_mod02_dr23.html
        # → BiB_AgeOfWonder.survey_mod02_dr23 (best-effort, used for join)
        stem = html_path.stem  # e.g. bib_ageofwonder_survey_mod02_dr23

        try:
            raw = html_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Find all <script> blocks and look for the Reactable data blob with variable data
        script_blocks = re.findall(r"<script[^>]*>(.*?)</script>", raw, re.DOTALL)
        for block in script_blocks:
            if '"variable":' not in block or '"closer_title":' not in block:
                continue
            try:
                data = json.loads(block)
                inner = data.get("x", {}).get("tag", {}).get("attribs", {}).get("data", {})
                variables = inner.get("variable", [])
                titles    = inner.get("closer_title", [])
                if variables and titles and len(variables) == len(titles):
                    sections[stem] = {
                        var: (title or "").strip()
                        for var, title in zip(variables, titles)
                    }
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue

    print(f"   ✅ Parsed section context from {len(sections)} HTML files")
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

        # Look up section from HTML (best-effort)
        # html stem would be like bib_{project_lower}_{table_name}
        project_lower = project.lower().replace("_", "")
        html_stem_guess = f"bib_{project_lower}_{table_nm}"
        section = table_name_to_sections.get(html_stem_guess, {}).get(variable, "")

        # Build rich text for embedding
        parts = [
            f"Table: {table_id}",
            f"Variable: {variable}",
            f"Label: {label}",
        ]
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

Context retrieved from the BiB knowledge base is provided below. Use it to ground your answer."""


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


def query(question: str, client: chromadb.ClientAPI, llm_client: Any,
          model: str = DEFAULT_MODEL, show_context: bool = False) -> str:
    """Run a RAG query: retrieve context → call HuggingFace LLM → return answer."""

    context = retrieve_context(question, client)

    if show_context:
        print("\n── Retrieved Context ──────────────────────────────────────────")
        print(context[:3000])
        print("──────────────────────────────────────────────────────────────\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
        max_tokens=1500,
    )
    return response.choices[0].message.content


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
        print(f"✅ Index ready — {cols['bib_papers']} papers | "
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
        def __init__(self, hf): self.chat = _Chat(hf)
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
