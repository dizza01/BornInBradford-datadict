# BiB Research Assistant

An AI-powered research assistant for the **Born in Bradford (BiB)** longitudinal cohort dataset. It combines a local vector database of 26,000+ variables, 289 tables, and 500 paper abstracts, 100+ BiB research papers with a HuggingFace language model to answer natural-language questions about the dataset.

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
│  ┌─────────────┐ ┌───────────┐ ┌─────────┐  │
│  │  bib_papers │ │bib_vars   │ │bib_csv  │  │
│  │             │ │           │ │         │  │
│  └─────────────┘ └───────────┘ └─────────┘  │
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


## Retrieval Evaluation (PDF triples)

We evaluated retrieval quality using generated PDF retrieval triples (`question`, `source_chunk_id`) from `eval/dataset_generator/generate_pdf_retrieval_triples.py`.

- **Goal**: check whether the known source chunk for each question is retrieved in the top-*k* results.
- **Dataset size**: 300 queries.
- **Metrics**:
  - **Recall@k**: fraction of queries where the true chunk appears in top-1 / 3 / 5 / 10.
  - **MRR**: mean reciprocal rank of the first correct hit (higher is better).
  - **Avg/Median hit rank**: how early correct chunks appear when they are retrieved.

### Triple generation methodology

The evaluation triples were generated automatically from indexed PDF full-text chunks using `eval/dataset_generator/generate_pdf_retrieval_triples.py`.

1. **Load candidate chunks**
  - Pull all records from Chroma collection `bib_papers` where `metadata.source == "pdf_fulltext"`.
  - Normalize whitespace and drop empty chunks.

2. **Sample chunks**
  - Target set size: 300 triples (`--sample-size 300`).
  - Candidate pool is expanded with `--candidate-multiplier` (e.g., 2.0) so unsuitable chunks can be skipped while still reaching target size.
  - Sampling is random with a fixed seed (`--seed 42`) for reproducibility.

3. **Generate grounded QA per chunk**
  - For each sampled chunk, send chunk text + metadata (`pdf_file`, `title`, `year`, `chunk`) to the configured HuggingFace model.
  - Prompt requires exactly one grounded JSON output with:
    - `question`
    - `answer`
    - `evidence` (short verbatim span)
    - `question_type` (`definition|method|result|dataset|theory`)
  - Model can return `{"skip": true}` for unsuitable chunks (references, headers/footers, acknowledgements, etc.).

4. **Quality checks before acceptance**
  - Keep only records with non-empty `question` and `answer`.
  - If `evidence` is provided, it must appear in the source chunk text.
  - Normalize unknown question types to `dataset`.

5. **Write JSONL triples**
  - Each accepted record includes the retrievable target `source_chunk_id`, plus `query_id`, `question`, `answer`, `question_type`, and source metadata.
  - Output file: `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl`.

### Quick end-to-end workflow (copy-paste)

```bash
cd BornInBradford-datadict/llm_poc

# 1) Generate triples from indexed PDF chunks
../../.venv/bin/python eval/dataset_generator/generate_pdf_retrieval_triples.py \
  --sample-size 300 \
  --overwrite

# 2) Run retrieval evaluation on the generated triples
../../.venv/bin/python eval/run_pdf_retrieval_eval.py \
  --triples eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl \
  --top-k 1 3 5 10 \
  --run-name retrieval_baseline_300

# 3) Run faithfulness evaluation on the same triples
../../.venv/bin/python eval/run_faithfulness_eval.py \
  --triples eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl \
  --max-queries 300 \
  --answer-model Qwen/Qwen2.5-72B-Instruct \
  --external-judge-model meta-llama/Llama-3.1-70B-Instruct \
  --qwen-judge-model Qwen/Qwen2.5-72B-Instruct \
  --judge-retries 2 \
  --run-name faithfulness_baseline_300
```

Outputs from this workflow:
- Triples: `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl`
- Retrieval run JSON + comparison row(s): `eval/results/retrieval_metrics/`
- Faithfulness run JSON + comparison row(s): `eval/results/llm_faithfulness_metrics/`

### Modes benchmarked

| Mode | Description |
|---|---|
| `dense` | Pure semantic vector search against ChromaDB using `all-MiniLM-L6-v2` embeddings. Fast baseline with no keyword matching. |
| `hybrid` | Dense retrieval + BM25 sparse retrieval fused with Reciprocal Rank Fusion (RRF). Combines semantic understanding with exact keyword matching. |
| `hybrid_rerank` | Hybrid fusion followed by a lightweight lexical reranker (unigram + bigram overlap) applied to the top-N fused candidates. Best overall quality. |

### Results (300 queries, top-k = 1 / 3 / 5 / 10)

| Mode | R@1 | R@3 | R@5 | R@10 | MRR | Avg rank | Median rank |
|---|---|---|---|---|---|---|---|
| `dense` | 0.203 | 0.380 | 0.433 | 0.523 | 0.304 | 3.01 | 2.00 |
| `hybrid` | 0.310 | 0.507 | 0.617 | 0.743 | 0.439 | 3.06 | 2.00 |
| `hybrid_rerank` | **0.420** | **0.653** | **0.727** | **0.807** | **0.552** | **2.38** | **1.00** |

`hybrid_rerank` is the recommended mode: it retrieves the correct chunk in the top-1 position for 42% of queries and within the top-10 for 81%, compared to 52% for pure dense retrieval. The median hit rank of 1 means the correct chunk is most often the very first result returned.


## Improvements
## Faithfulness Evaluation (LLM component)

To evaluate answer faithfulness (not just retrieval), use:

```bash
cd BornInBradford-datadict/llm_poc
../../.venv/bin/python eval/run_faithfulness_eval.py \
  --triples eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl \
  --max-queries 300 \
  --answer-model Qwen/Qwen2.5-72B-Instruct \
  --external-judge-model meta-llama/Llama-3.1-70B-Instruct \
  --qwen-judge-model Qwen/Qwen2.5-72B-Instruct \
  --judge-retries 2 \
  --run-name faithfulness_baseline
```

Method used:

- Generate RAG answers with the baseline answer model (Qwen by default).
- Judge each answer against retrieved context with a **primary external judge** (must be non-Qwen).
- Run a **Qwen judge sensitivity check** on the same answers/context.
- Retry judge calls on malformed JSON/API errors (`--judge-retries`, `--judge-retry-delay`) to reduce dropped queries.
- Claim-level verdicts are now split into:
  - `supported`
  - `contradicted` (explicitly conflicts with context)
  - `not_found` (no evidence in context)
  - `unclear`
- Report:
  - primary metrics from external judge (`unfaithful_claim_rate`, `contradicted_claim_rate`, `not_found_claim_rate`, `query_faithfulness_rate`, `claim_support_rate`, `contradiction_weighted_hallucination`)
  - Qwen judge sensitivity metrics
  - agreement between judges (`faithful_label_agreement`, `unfaithful_presence_agreement`)

The script writes a JSON report to `eval/faithfulness_eval_results.json` (or `eval/results/llm_faithfulness_metrics/<run-name>_<timestamp>.json` when `--run-name` is provided).
When `--run-name` is set, it also appends a comparison row set (external + Qwen judge) to `eval/results/llm_faithfulness_metrics/comparison.csv`.


## Retrieval quality
Retrieval quality is measured by running each query from the generated evaluation triples against the retriever and checking whether the known source chunk for that query appears in the top results. For each retrieval mode (dense, hybrid, hybrid_rerank), the script computes Recall@k (whether the gold chunk is found within top 1/3/5/10), MRR (higher weight when the gold chunk appears earlier), and hit-rank summaries (average/median rank of the first correct hit). In practice, this is conducted over a fixed query set (e.g., 300 queries), fixed retrieval settings, and logged to JSON plus eval/results/retrieval_metrics/comparison.csv, so modes can be directly compared under the same conditions.


## Faithfulness of generated answers
Faithfulness is measured after retrieval by generating a RAG answer per query, then judging each answer claim against the retrieved context only (not external knowledge). The primary metric uses an external non-Qwen judge; a Qwen judge is run as sensitivity analysis, and agreement between judges is reported. Each claim is labeled as supported, contradicted, or not_found, then aggregated into project metrics: claim_support_rate, contradicted_claim_rate, not_found_claim_rate, unfaithful_claim_rate, query_faithfulness_rate (answer-level faithfulness), and contradiction_weighted_hallucination (heavier penalty for contradictions). Results are written to JSON and appended to eval/results/llm_faithfulness_metrics/comparison.csv for baseline-vs-dense and run-to-run comparison.


### Faithfulness metrics (how they are calculated)

These metrics are computed per evaluation run (`run_name`) and judge (`judge_type`) over all evaluated queries.

- `run_name`: Identifier for the experiment configuration (for example, `faithfulness_baseline_300` vs `faithfulness_dense_300`).
- `retrieval_mode_clean`: Retrieval setup used to produce RAG context (`default` = baseline pipeline, `dense` = dense-only retrieval).
- `judge_type`: Which judge produced labels (`external` = primary non-Qwen judge, `qwen` = sensitivity judge).

For the scored metrics:

- `contradicted_claim_rate`  
  Claim-level contradiction frequency:
  \[
  \text{contradicted\_claim\_rate} = \frac{\#(\text{claims labeled contradicted})}{\#(\text{all judged claims})}
  \]

- `unfaithful_claim_rate`  
  Claim-level unfaithfulness frequency. In this project this combines non-supported labels (primarily `contradicted + not_found`):
  \[
  \text{unfaithful\_claim\_rate} = \frac{\#(\text{unfaithful claims})}{\#(\text{all judged claims})}
  \]
  This is always greater than or equal to `contradicted_claim_rate`.

- `query_faithfulness_rate`  
  Query-level faithfulness (stricter than claim-level): fraction of answers judged faithful at whole-answer level (typically meaning no unfaithful claims in that answer):
  \[
  \text{query\_faithfulness\_rate} = \frac{\#(\text{queries marked faithful})}{\#(\text{evaluated queries})}
  \]

These values are aggregated separately for each retrieval mode and judge so results can be compared directly across configurations.



#### Compact worked example (one run)

Suppose one faithfulness run has:

- `n_queries = 300`
- `total_claims = 1200`
- `supported = 1048`
- `contradicted = 12`
- `not_found = 36`
- `faithful_queries = 219`

Then:

\[
\text{contradicted\_claim\_rate} = \frac{12}{1200} = 0.010
\]

\[
\text{not\_found\_claim\_rate} = \frac{36}{1200} = 0.030
\]

\[
\text{unfaithful\_claim\_rate} = \frac{12 + 36}{1200} = \frac{48}{1200} = 0.040
\]

\[
\text{claim\_support\_rate} = \frac{1048}{1200} = 0.873
\]

\[
\text{query\_faithfulness\_rate} = \frac{219}{300} = 0.730
\]

So this run would report approximately:
- `contradicted_claim_rate = 0.010`
- `not_found_claim_rate = 0.030`
- `unfaithful_claim_rate = 0.040`
- `claim_support_rate = 0.873`
- `query_faithfulness_rate = 0.730`


## Phase 2 eval plan (concrete)

This plan adds a compact headline set of retrieval + generation metrics and states exactly which dataset each metric uses.

### Target metrics and dataset mapping

| Area | Metric | Can use current `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl`? | Dataset to use |
|---|---|---|---|
| Retrieval | `Recall@10` | Yes | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` |
| Retrieval | `nDCG@10` | Yes (single-gold relevance) | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` |
| Retrieval | `Precision@5` | Yes (binary relevance vs `source_chunk_id`) | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` |
| Retrieval | `latency p95` | Yes (dataset-independent timing over same query set) | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` |
| Generation | `query_faithfulness_rate` | Yes | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` + faithfulness judge outputs |
| Generation | `contradiction_weighted_hallucination` | Yes | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` + faithfulness judge outputs |
| Generation | `Entity F1` | No (not enough structured gold entities) | New gold entity benchmark |
| Generation | `abstention accuracy` | No (triples are answerable by design) | New negative/abstention benchmark |

### Retrieval metrics (implemented from current triples)

- Run retrieval modes on `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` (same as current retrieval eval).
- Add two summary metrics to `eval/run_pdf_retrieval_eval.py`:
  - `nDCG@10` (with one relevant chunk per query, derived from rank of `source_chunk_id`)
  - `Precision@5` (fraction of top-5 items equal to the known relevant set; with one gold chunk this is 0 or 0.2 per query)
- Add latency capture:
  - measure per-query retrieval time in milliseconds around the retrieval call
  - report `p50` and `p95` per mode in JSON + `results/retrieval_metrics/comparison.csv`

### Generation metrics already available from current triples

- `query_faithfulness_rate` and `contradiction_weighted_hallucination` are already produced by `eval/run_faithfulness_eval.py` using the same triples file.
- Keep reporting both external-judge and Qwen-sensitivity values in `eval/results/llm_faithfulness_metrics/comparison.csv`.

### Dataset creation for metrics not supported by current triples

#### 1) Entity F1 benchmark (new)

Create `eval/entity_f1_benchmark.jsonl` with one record per question:

```json
{
  "query_id": "...",
  "question": "...",
  "gold_variables": ["rcad_ga", "rcad_sep"],
  "gold_tables": ["BiB_AgeOfWonder.survey_mod232_derived_dr24"],
  "gold_studies": ["Age of Wonder"],
  "gold_papers": ["paper title or id"]
}
```

How to generate efficiently (hybrid automatic + manual QA):

1. Auto-seed candidates from existing triples + retrieval outputs (variable/table/study/paper mentions).
2. Normalize via dictionary registries (`all_variables_meta.csv`, `all_tables.csv`, paper metadata).
3. Manually verify a sampled subset, refine rules, then freeze benchmark version.
4. Compute entity-level Precision/Recall/F1 against model-extracted entities.

#### 2) Abstention benchmark (new)

Create `eval/evaluation_datasets/variable_abstention/abstention_benchmark.jsonl` with explicit negative labels:

```json
{
  "query_id": "...",
  "question": "Is gestation_at_recruitment_weeks a BiB variable?",
  "should_abstain": true,
  "reason": "variable not in registry"
}
```

Quick copy-paste commands:

```bash
cd BornInBradford-datadict/llm_poc

# Generate both slices (variables/tables + papers) and unified report
../../.venv/bin/python eval/dataset_generator/generate_combined_abstention_benchmarks.py \
  --target both \
  --overwrite

# Generate variables/tables slice only
../../.venv/bin/python eval/dataset_generator/generate_combined_abstention_benchmarks.py \
  --target variables_tables \
  --n-examples-vt 500 \
  --overwrite

# Generate paper slice only
../../.venv/bin/python eval/dataset_generator/generate_combined_abstention_benchmarks.py \
  --target papers \
  --n-examples-paper 400 \
  --overwrite

# Optional direct generators (if you do not want the combined runner)
../../.venv/bin/python eval/dataset_generator/generate_variable_abstention_benchmark.py --overwrite
../../.venv/bin/python eval/dataset_generator/generate_paper_abstention_benchmark.py --overwrite
```

Default outputs from these commands:
- Variables/tables dataset: `eval/evaluation_datasets/variable_abstention/abstention_benchmark.jsonl`
- Paper dataset: `eval/evaluation_datasets/paper_abstention/paper_abstention_benchmark.jsonl`
- Combined report: `eval/evaluation_datasets/combined_abstention/combined_abstention_benchmark_report.json`

Evaluate abstention metrics and append to comparison CSV:

```bash
cd BornInBradford-datadict/llm_poc

../../.venv/bin/python eval/run_abstention_eval.py \
  --run-name abstention_baseline \
  --model Qwen/Qwen2.5-72B-Instruct
```

Abstention evaluation outputs:
- JSON report: `eval/results/abstention_metrics/abstention_eval_results.json`
- Comparison CSV: `eval/results/abstention_metrics/comparison.csv`

How to generate efficiently:

1. Auto-generate negatives by controlled perturbations:
   - non-existent variable names
   - wrong study/table combinations
   - cross-domain out-of-scope questions
2. Keep a matched positive set for balance (for example, 50/50).
3. Manually review at least 10–20% to ensure negatives are truly unanswerable.
4. Report:
   - `abstention_accuracy`
   - `true_abstain_rate`
   - `false_answer_rate` on negative items.

### Minimal reporting schema (recommended)

Write one row per run per mode to CSV with:

- retrieval: `recall_at_10`, `ndcg_at_10`, `precision_at_5`, `latency_p95_ms`
- generation: `query_faithfulness_rate`, `contradiction_weighted_hallucination`, `entity_f1`, `abstention_accuracy`
- metadata: `run_name`, `model`, `retrieval_mode`, `n_queries`, `dataset_version`, `timestamp`

This keeps Phase 2 results comparable across baseline, dense-only, and future fine-tuned model runs.


## Evaluation Metric Matrix (Implemented / Partial / Planned)

Status legend:
- **Implemented**: metric is computed and logged in current scripts/outputs.
- **Partial**: proxy/limited version exists; useful but not full target definition.
- **Planned**: not yet implemented in current pipeline.

### Retrieval

| Metric | Status | Dataset used | Current source | Notes / next step |
|---|---|---|---|---|
| Recall@k / Hit Rate@k | **Implemented** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | `eval/run_pdf_retrieval_eval.py`, `eval/results/retrieval_metrics/comparison.csv` | Includes `recall_at_1/3/5/10`. |
| MRR | **Implemented** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | `eval/run_pdf_retrieval_eval.py`, retrieval comparison CSV | Rank quality of first relevant chunk. |
| nDCG@10 | **Implemented** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | `eval/run_pdf_retrieval_eval.py`, retrieval comparison CSV | Binary relevance with single gold chunk. |
| Context Precision | **Partial** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | `precision_at_5` in retrieval comparison CSV | Current proxy uses single gold chunk. Full precision needs multi-relevant labels. |
| Context Recall | **Partial** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | `recall_at_k` in retrieval comparison CSV | Current recall is against one known gold chunk, not all relevant chunks. |
| Retrieval latency p50/p95 | **Implemented** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | `eval/run_pdf_retrieval_eval.py`, retrieval comparison CSV | Per-query timing captured around retrieval calls. |
| TTFT (time-to-first-token) | **Planned** | N/A | N/A | Not explicitly tracked yet. |

### Faithfulness

| Metric | Status | Dataset used | Current source | Notes / next step |
|---|---|---|---|---|
| Faithfulness (claim + query level) | **Implemented** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | `eval/run_faithfulness_eval.py`, `eval/results/llm_faithfulness_metrics/comparison.csv` | Includes claim rates and query-level faithful labeling. |
| query_faithfulness_rate | **Implemented** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | `eval/run_faithfulness_eval.py`, faithfulness comparison CSV | Answer-level faithfulness headline metric. |
| contradiction_weighted_hallucination | **Implemented** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | `eval/run_faithfulness_eval.py`, faithfulness comparison CSV | Heavier penalty for contradictions than unsupported claims. |
| Answer Relevance | **Implemented** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | `run_faithfulness_eval.py` generation quality summary + faithfulness comparison CSV | Token-overlap proxy on question vs answer. |
| Completeness | **Implemented** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | `run_faithfulness_eval.py` generation quality summary + faithfulness comparison CSV | Reference-answer coverage proxy. |
| Conciseness | **Implemented** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | `run_faithfulness_eval.py` generation quality summary + faithfulness comparison CSV | Length-ratio style score. |
| Answer Correctness | **Implemented** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | `run_faithfulness_eval.py` (`answer_correctness_token_f1`, `answer_correctness_exact_match_rate`) | Exact match is strict; token F1 usually more informative. |
| Token efficiency | **Implemented** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | `run_faithfulness_eval.py` generation quality summary + faithfulness comparison CSV | Includes `answer_tokens_per_retrieved_token`. |
| End-to-end latency (faithfulness run) | **Partial** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` | Faithfulness output JSON + comparison CSV | Includes `latency_total_p95_ms` and average stage latencies; TTFT not tracked separately. |
| Entity F1 | **Planned** | New benchmark required (`entity_f1_benchmark.jsonl`) | N/A | Requires new gold entity benchmark and extractor/evaluator wiring. |

### Abstention

| Metric | Status | Dataset used | Current source | Notes / next step |
|---|---|---|---|---|
| Abstention accuracy (overall) | **Implemented** | `eval/evaluation_datasets/variable_abstention/abstention_benchmark.jsonl` + `eval/evaluation_datasets/paper_abstention/paper_abstention_benchmark.jsonl` | `eval/run_abstention_eval.py`, `eval/results/abstention_metrics/comparison.csv` | Evaluates answer-vs-abstain classification on balanced answerable/unanswerable slices. |
| True abstain rate / recall on unanswerable | **Implemented** | Same as above | `eval/run_abstention_eval.py`, abstention comparison CSV | Computed from confusion matrix on should_abstain labels. |
| False answer rate | **Implemented** | Same as above | `eval/run_abstention_eval.py`, abstention comparison CSV | Fraction answered when ground truth is abstain. |
| False abstain rate | **Implemented** | Same as above | `eval/run_abstention_eval.py`, abstention comparison CSV | Fraction abstained when ground truth is answerable. |
| Precision/Recall/F1 (abstain class) | **Implemented** | Same as above | `eval/run_abstention_eval.py`, abstention comparison CSV | Supports strict classifier mode (`model_strict`). |
| Per-slice abstention metrics | **Implemented** | Variable and paper abstention slices | `eval/run_abstention_eval.py` (`segment_type=slice`) | Separate rows for `variable_abstention` and `paper_abstention`. |
| Per-source-scope abstention metrics | **Implemented** | Records with `source_scope` labels in abstention datasets | `eval/run_abstention_eval.py` (`segment_type=source_scope`) | Breaks out `in_index_snapshot` vs `outside_index_snapshot` where present. |
| Per-reason-type abstention metrics | **Implemented** | Records with reason/generation_type metadata in abstention datasets | `eval/run_abstention_eval.py` (`segment_type=reason_type`) | Includes categories such as out-of-scope and non-existent/missing. |
| Adversarial / no-answer reporting | **Implemented** | Out-of-scope subset from abstention datasets | `eval/run_abstention_eval.py` (`segment_type=adversarial_no_answer`) | Uses records tagged `outside_index_snapshot` / `out_of_scope`. |
| Abstention calibration (confidence quality) | **Planned** | `eval/results/abstention_metrics/abstention_eval_results.json` with classifier confidence fields | N/A | Confidence is captured in strict mode, but calibration metrics (ECE/Brier) are not yet computed. |

### Strategy coverage (current)

| Strategy | Status | Current state |
|---|---|---|
| Golden dataset | **Implemented** | `eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl` (question/answer/source chunk). |
| LLM-as-judge | **Implemented** | External judge primary + Qwen sensitivity judge in faithfulness eval. |
| Separate failure modes | **Implemented** | Retrieval and generation metrics are reported separately. |
| Chunk-level vs document-level bottleneck tracking | **Planned** | Not yet explicitly instrumented. |
| Adversarial/no-answer cases | **Planned** | Pending abstention benchmark dataset. |

