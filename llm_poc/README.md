# BiB Research Assistant

An AI-powered research assistant for the **Born in Bradford (BiB)** longitudinal cohort dataset. It combines a local vector database of 26,000+ variables, 289 tables, and 500 paper abstracts with a HuggingFace language model to answer natural-language questions about the dataset.

---

## Running Locally

### Prerequisites

- Python 3.10+ with a virtual environment at `BiB/.venv`
- A free HuggingFace token — get one at https://huggingface.co/settings/tokens

---

### Step 1 — Install dependencies

From the repo root (`BiB/`):

```bash
.venv/bin/pip install -r BornInBradford-datadict/llm_poc/requirements_llm_poc.txt
```

---

### Step 2 — Add your HuggingFace token

Create (or confirm) the file `BornInBradford-datadict/llm_poc/.env` contains:

```
HF_TOKEN=hf_your_token_here
```

---

### Step 3 — Build the vector index *(one-time, ~2–5 minutes)*

Only needed on first run, or after the source data changes.

```bash
cd BornInBradford-datadict/llm_poc
../../.venv/bin/python bib_research_assistant.py --build
```

This reads the CSVs and HTML files, embeds everything locally using `all-MiniLM-L6-v2`, and saves the index to `.chroma_db/`. To do a clean rebuild from scratch at any time run `bash build_index.sh` instead.

---

### Step 4 — Start the server

```bash
bash BornInBradford-datadict/llm_poc/start.sh
```

Or equivalently:

```bash
cd BornInBradford-datadict/llm_poc
../../.venv/bin/python server.py
```

`start.sh` will automatically build the index if it doesn't exist yet.

You should see:

```
✅ Index ready — 500 papers | 26104 variables | 289 tables
🌐 Server running at: http://127.0.0.1:5050
```

Then open your browser:

| URL | What you get |
|-----|-------------|
| `http://localhost:5050/` | Data dictionary website with floating 🔬 chat button (bottom-right of every page) |
| `http://localhost:5050/assistant` | Full-screen chat interface with suggested questions |

Press **Ctrl+C** to stop the server.

#### Options

```bash
# Different port
bash start.sh --port 8080

# Faster/smaller model
bash start.sh --model "HuggingFaceH4/zephyr-7b-beta"
```

---

### Step 5 — (Optional) Command-line interface

If you prefer the terminal over the browser:

```bash
cd BornInBradford-datadict/llm_poc

# Interactive chat session
../../.venv/bin/python bib_research_assistant.py --chat

# Single query and exit
../../.venv/bin/python bib_research_assistant.py \
  --query "What variables measure anxiety in Age of Wonder?"

# Show the retrieved context chunks alongside the answer
../../.venv/bin/python bib_research_assistant.py \
  --query "What variables measure anxiety in Age of Wonder?" --context

# Use a different model
../../.venv/bin/python bib_research_assistant.py \
  --model "HuggingFaceH4/zephyr-7b-beta" --chat
```

---

## How It Works

### Architecture

```
User question
      │
      ▼
┌─────────────────────────────────────────────┐
│            Retrieval (ChromaDB)             │
│  ┌─────────────┐ ┌───────────┐ ┌─────────┐ │
│  │  bib_papers │ │bib_vars   │ │bib_tables│ │
│  │  500 papers │ │26,104 vars│ │289 tables│ │
│  └─────────────┘ └───────────┘ └─────────┘ │
└─────────────────────────────────────────────┘
      │  top-k relevant chunks
      ▼
┌─────────────────────────────────────────────┐
│       HuggingFace LLM (Qwen 2.5-72B)        │
│  System prompt + retrieved context + query  │
└─────────────────────────────────────────────┘
      │
      ▼
   Answer grounded in BiB knowledge base
```

### Knowledge Sources

| Source | Content | Count |
|--------|---------|-------|
| `papers/bib_papers_metadata.json` | Title + abstract for BiB publications | 500 papers |
| `docs/csv/all_variables_meta.csv` | Variable names, labels, types, topics, completeness | 26,104 variables |
| `docs/csv/all_tables.csv` | Table IDs, projects, entity types, row counts | 289 tables |
| `docs/*.html` | `closer_title` section groupings parsed from Reactable JSON | 326 HTML files |

### Indexing (`--build`)

1. **HTML parsing** — Each data dictionary HTML file contains an embedded Reactable JSON blob. The indexer extracts `variable → closer_title` (section heading) mappings to enrich variable records with human-readable context that isn't in the CSVs.

2. **Embedding** — All text is embedded using ChromaDB's default model (`all-MiniLM-L6-v2`, runs locally, no API needed) and stored in three collections: `bib_papers`, `bib_variables`, `bib_tables`.

3. **Persistence** — The index is saved to `.chroma_db/` and reused on every subsequent query.

### Querying (RAG)

For each question:

1. The question is embedded and used to retrieve the top-5 papers, top-10 variables, and top-5 tables by semantic similarity.
2. The retrieved chunks are formatted as a markdown context block.
3. The context + question are sent to the HuggingFace LLM with a BiB-specific system prompt that instructs it to cite variable names, reference papers, note completeness, and respect privacy rules (no `BiBPersonID` in results).
4. The LLM answer is returned.

### Web Server (`server.py`)

Flask serves two things:

- **Static docs site** — proxies all files from `docs/` with a floating chat widget (`/widget-static/chat-widget.js`) injected before `</body>` in every HTML page. Researchers can browse the data dictionary and ask questions without leaving the page.

- **`/assistant`** — a standalone full-screen chat page with suggested starter questions.

- **`POST /api/chat`** — JSON API consumed by both interfaces:
  ```json
  // Request
  { "question": "What anxiety variables exist in Age of Wonder?" }

  // Response
  { "answer": "The RCADS scale variables rcad_ga, rcad_ga_t... " }
  ```

---

## Files

```
llm_poc/
├── bib_research_assistant.py   # Core RAG engine + CLI
├── server.py                   # Flask web server
├── start.sh                    # ← run this to launch everything
├── build_index.sh              # Wipe + rebuild index from scratch
├── requirements_llm_poc.txt    # Python dependencies
├── .env                        # HF_TOKEN (not in git)
├── .chroma_db/                 # Built vector index (not in git)
└── static/
    ├── chat-widget.js          # Floating chat widget (auto-injected into every page)
    └── chat-widget.css         # Widget styles
```

---

## Models

Default: `Qwen/Qwen2.5-72B-Instruct` (best free-tier quality on HuggingFace).

| Model | Notes |
|-------|-------|
| `Qwen/Qwen2.5-72B-Instruct` | Default. Best quality on free tier. |
| `meta-llama/Llama-3.1-8B-Instruct` | Faster. Requires accepting Meta licence on HF. |
| `HuggingFaceH4/zephyr-7b-beta` | Reliable fallback, no sign-up required. |

Switch model via `--model` flag or the `DEFAULT_MODEL` constant in `bib_research_assistant.py`.

---

## Rebuilding the Index

Run after any changes to the source CSVs or HTML files:

```bash
cd BornInBradford-datadict/llm_poc
bash build_index.sh
```

This wipes `.chroma_db/` and rebuilds all three collections from scratch.


## Improvements

Fine-tuning: what it would and wouldn't help
What fine-tuning on the PDFs would improve:

The model's fluency in BiB/epidemiology terminology and writing style
Responses that sound more like a cohort study researcher
Familiarity with BiB-specific concepts (BiBPersonID, wave names, Bradford context)
What fine-tuning wouldn't reliably improve:

Factual recall of specific variables, table IDs, or paper findings — models can still hallucinate these after fine-tuning. That's exactly what RAG handles better.
Fine-tuning teaches style and domain language. RAG injects specific facts. They're complementary, not alternatives.

Practical constraints
Fine-tuning Qwen 2.5-72B requires multiple A100 GPUs — not feasible locally or on the free HF tier
25 local PDFs is a very small corpus; you'd want at minimum hundreds of full papers
You'd then need to host the fine-tuned model yourself
A realistic middle ground is LoRA/PEFT fine-tuning on a smaller model (e.g. LLaMA 3.1-8B) using a cloud service like AutoTrain or Google Colab A100 — achievable but still non-trivial.

The higher-impact improvement available right now
Currently the system only indexes paper abstracts (500 papers × ~200 words). The 25 local PDFs contain full methods, results, and discussion sections — extracting those would give the LLM far richer factual context with no model changes needed.

This would be the highest ROI improvement: extract full text from the local PDFs and add them to the bib_papers ChromaDB collection.

## Restart port

cd /Users/dawud.izza/Desktop/BiB/BornInBradford-datadict/llm_poc
lsof -ti :5050 | xargs kill -9 2>/dev/null; sleep 1
/Users/dawud.izza/Desktop/BiB/.venv/bin/python server.py