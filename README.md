# Born in Bradford Data Dictionary

A comprehensive interactive data dictionary documenting all datasets, variables, and data structures from the **Born in Bradford (BiB)** longitudinal birth cohort study.

## 📖 About

Born in Bradford is a large-scale research program following families in Bradford, UK, to understand how genetic, nutritional, environmental, and social factors affect health and development from pregnancy through childhood and beyond. This repository contains the data dictionary documentation for the BiB cohort and related sub-studies.

## 🗂️ Repository Contents

This repository contains **321 HTML documentation pages** (~170 MB total) organized as an interactive bookdown website, providing detailed information about:

- **Variable definitions** and metadata
- **Data collection methods** and timepoints
- **Data linkage** information
- **Coding schemes** and value labels
- **Quality control** information

### 📊 Main Data Categories

The data dictionary covers the following major research areas and sub-studies:

#### **Core Cohort Data**
- **BiB Cohort Information** - Participant demographics, ethnicity, family relationships, ID linkage
- **BiB Baseline** - Maternal and paternal baseline surveys, food frequency questionnaires, exercise data
- **BiBBS (Born in Bradford's Better Start)** - Newer cohort recruitment and baseline data
- **BiB4All** - Extended cohort geographic and dental data

#### **Health & Clinical Data**
- **Pregnancy** - Maternity records, ultrasound scans, fetal renal data, blood pressure
- **Biosamples** - Blood tests, cord blood, GTT, biochemistry, haematology
- **Biobank** - Current biobank sample inventory
- **Congenital Anomalies** - Yorkshire & Humber CAR and GP linkage data
- **Dental** - Dental surveys, extractions, PLATOON study data

#### **Child Development & Growth**
- **BiB 1000** - Longitudinal questionnaires at 6m, 12m, 18m, 24m, 36m
- **Child Growth** - Anthropometry, bioimpedance, NCMP data, primary care records
- **Maternal Measurements** - Research measurement data
- **Starting School** - BPVS, letter identification, CKAT assessments
- **Primary School Years** - SDQ data, child quizzes, executive function tasks
- **Growing Up** - Adult and child surveys, DXA scans, blood pressure, renal studies
- **Age of Wonder** - Recent school visits, surveys, anthropometry (2023-2024 releases)

#### **Environmental Data**
- **Geographic Information** - Air quality, built environment, green space, food environment, NDVI, walkability
- **BREATHES** - Air quality survey data (Phase 1 & 2)

#### **Specialized Studies**
- **ALL IN** - Questionnaires on childcare, immunizations, household data (12m, 24m)
- **MeDALL** - Allergy questionnaires, skin prick tests, green space data
- **Education Records** - EYFSP, Key Stage assessments, phonics, contextual data

#### **Omics & Biomarkers**
- **Metabolomics** - MS and NMR metabolomics (mother, baby, child samples)
- **Proteomics** - Next generation sequencing, QPCR projects
- **Glycomics** - Total plasma glycome analyses (HILIC-UHPLC-FLR)
- **Genotyping** - Linkage to genotyping, exome sequencing, DNA methylation data

#### **Mental Health & Wellbeing**
- **SDQs** - Strength and Difficulties Questionnaires from multiple timepoints and studies
- **COVID-19 Surveys** - Phase 1, 2, and 3 survey data

## 🚀 How to Use

### Viewing the Data Dictionary

#### Quick Start - View in Browser

From the repository root directory, run:

```bash
# macOS
open docs/index.html

# Linux
xdg-open docs/index.html

# Windows
start docs/index.html

# Or use the full path
open /path/to/BornInBradford-datadict/docs/index.html
```

#### Clone and Browse

If you haven't cloned the repository yet:

```bash
git clone https://github.com/dizza01/BornInBradford-datadict.git
cd BornInBradford-datadict
open docs/index.html  # macOS
# or
xdg-open docs/index.html  # Linux
# or
start docs/index.html  # Windows
```

The data dictionary will open in your default web browser with full navigation and search functionality.

### Navigation

- Use the **left sidebar** to navigate between data modules
- Use the **search function** to find specific variables or tables
- Each table includes:
  - Variable names
  - Variable labels/descriptions
  - Value labels (for categorical variables)
  - Data types
  - Collection timepoints

### Finding Specific Information

**By Research Area**: Navigate using the main category pages (e.g., `bib_baseline.html`, `bib_pregnancy.html`)

**By Sub-study**: Look for specific study names (e.g., `bib_1000`, `bib_medall`, `bib_ageofwonder`)

**By Data Type**: Search for specific data types:
- Survey/questionnaire data (main tables with `_main.html` suffix)
- Geographic data (`bib_geographic` section)
- Clinical measurements (`bib_biosamples`, `bib_childgrowth`)
- Omics data (`bib_metabolomics`, `bib_proteomics`, `bib_glycomics`)

## 📋 Data Structure

### Naming Conventions

Data tables follow a hierarchical naming structure:

```
{cohort}_{study}_{subcategory}_{table}
```

**Examples:**
- `bib_baseline_base_m_survey` - BiB baseline maternal survey
- `bib_1000_bib1000_12m_main` - BiB 1000 study 12-month main questionnaire
- `bibbs_cohortinfo_pregnancy` - BiBBS cohort pregnancy information

### Data Relationships & Linkage

> **📊 [VIEW COMPLETE DATA MODEL](DATA_MODEL.md)** - Comprehensive Entity-Relationship diagrams showing all table relationships, identifiers, and linkage patterns.

#### Key Identifiers

The cohort uses various linking identifiers to connect tables:
- **Person-level IDs** - Unique identifiers for individuals (mothers, fathers, children)
- **Pregnancy IDs** - Link pregnancy-related data
- **Family relationship IDs** - Connect family members
- **Study-specific IDs** - Link to sub-study data (BiB 1000, MeDALL, etc.)
- **Property & LSOA codes** - Geographic linkage

#### Relationship Documentation

The data dictionary documents table relationships through:

1. **ID Linkage Section** ([bib_cohortinfo_id_linkage.html](docs/bib_cohortinfo_id_linkage.html))
   - Comprehensive guide to identifier types and how they link tables
   - Shows which IDs to use for joining datasets

2. **Related Pairs** ([bib_cohortinfo_related_pairs.html](docs/bib_cohortinfo_related_pairs.html))
   - Parent-child relationships
   - Sibling relationships  
   - Family linkage information

3. **Data Linkage Pages** - Specific sections for:
   - Genotyping data linkage
   - Exome sequencing data linkage
   - DNA methylation data linkage
   - GP record linkage to other datasets

4. **Hierarchical Structure** - Data is organized by:
   - **Person-level** → Individual participant data
   - **Pregnancy-level** → Linked to mothers via pregnancy ID
   - **Property-level** → Geographic data linked to residential addresses
   - **LSOA-level** → Area-level geographic and demographic data

5. **Temporal Relationships** - Longitudinal data linkage:
   - Same participants across timepoints (6m, 12m, 18m, 24m, 36m)
   - Baseline → Follow-up study linkage
   - Cross-sectional sub-studies linked to main cohort

## 🔍 Use Cases

This data dictionary is essential for:

1. **Data Analysts** - Understanding variable definitions before analysis
2. **Researchers** - Planning research proposals and identifying available data
3. **Collaborators** - Exploring what data exists within the BiB cohort
4. **Data Managers** - Reference for data structure and coding schemes
5. **Students** - Learning about longitudinal cohort study data organization

## 📦 Project Structure

```
BornInBradford-datadict/
├── docs/                      # 326 HTML documentation files
│   ├── index.html            # Main entry point
│   ├── bib_*.html            # BiB cohort documentation
│   ├── bibbs_*.html          # BiBBS cohort documentation
│   ├── bib4all_*.html        # BiB4All documentation
│   ├── covid19_*.html        # COVID-19 survey documentation
│   ├── csv/                  # Machine-readable metadata exports
│   │   ├── all_variables_meta.csv   # 26 104 variables with labels, types, stats
│   │   └── all_tables.csv           # 289 table definitions
│   └── libs/                 # Supporting JavaScript/CSS libraries
├── llm_poc/                   # LLM-powered research assistant
│   ├── bib_research_assistant.py   # RAG engine (ChromaDB + HuggingFace)
│   ├── server.py                   # Flask web server
│   ├── requirements_llm_poc.txt    # Python dependencies
│   ├── build_index.sh              # Rebuild vector index
│   └── start.sh                    # Start the assistant server
├── papers/                    # BiB research papers (PDFs)
├── datadict.Rproj            # R project file
├── README.md                 # This file
└── .nojekyll                 # Prevents Jekyll processing
```

## 🤖 LLM Research Assistant

The `llm_poc/` directory contains a Retrieval-Augmented Generation (RAG) assistant that lets researchers ask natural-language questions about the BiB dataset — e.g. *"which variables measure anxiety in children?"* or *"what rcad_ga columns are available?"*.

### Architecture

| Component | Detail |
|---|---|
| Vector store | ChromaDB (local, `.chroma_db/`) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| LLM | `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference API |
| Collections | `bib_variables` (26 104 vars), `bib_tables` (289 tables), `bib_papers` (500 abstracts + PDF full-text chunks) |
| Web UI | Flask server on port 5050, served alongside the static bookdown site |

### Variable Index Enrichment — HTML Label Extraction

Every HTML file in `docs/` is generated by the R **reactable** package and embeds its data as a JSON blob inside a `<script>` tag.  That blob contains three parallel arrays:

```json
{
  "variable":     ["rcad_ga",  "rcad_sep",  ...],
  "label":        ["RCADS-25 General anxiety. Raw score",
                   "RCADS-25 Separation anxiety. Raw score", ...],
  "closer_title": ["Mental health", "Mental health", ...]
}
```

| Array | Content |
|---|---|
| `variable[]` | Machine variable name (e.g. `rcad_ga`) |
| `label[]` | Full human-readable description (e.g. *"RCADS-25 General anxiety. Raw score"*) |
| `closer_title[]` | Topic / section heading (e.g. *"Mental health"*) |

Previously `parse_html_sections()` only extracted `closer_title` and discarded `label`, which caused the RAG system to confuse short abbreviations — e.g. `rcad_ga` (RCADS General Anxiety) was retrieved alongside `dental_ga` (General Anaesthesia) because the embedding had no distinguishing text.

The fix (applied to `bib_research_assistant.py`) now extracts all three arrays for all 326 HTML files and stores `{"section": closer_title, "description": label}` per variable.  During indexing, `build_variables_collection()` emits a `Description:` line in each variable's embedding text whenever the HTML label carries additional detail beyond the CSV label, giving every variable a semantically unambiguous fingerprint.

### Quick start

```bash
cd llm_poc
bash build_index.sh   # build / rebuild ChromaDB vector index (~5 min first run)
bash start.sh         # start Flask server on http://localhost:5050
```

Set `HF_TOKEN` in your environment (or a `.env` file) before starting.

---

## 🛠️ Technical Details

- **Generated using**: R bookdown package (v0.42)
- **Framework**: GitBook (v2.6.7)
- **Interactive tables**: reactable package
- **Total size**: ~170 MB (50 MB Git history + 120 MB working files)
- **HTML files**: 326 pages

## 📝 Contributing

This data dictionary is maintained by the Born in Bradford research team. For questions about:
- **Data access**: Contact the BiB Data Access Committee
- **Variable definitions**: Refer to study protocols or contact study leads
- **Technical issues**: Open an issue in this repository

## 📚 Related Resources

- [Born in Bradford Website](https://borninbradford.nhs.uk/)
- Data access requests and collaboration information
- Study protocols and publications

## ⚖️ License & Data Access

This data dictionary is publicly available for reference. However, **access to the actual data** requires:
- Approval from the BiB Data Access Committee
- Appropriate ethical approvals
- Signed data sharing agreements

The data dictionary helps researchers understand what data exists before applying for access.

## 📧 Contact

For more information about Born in Bradford or data access, please visit the official Born in Bradford website.

---

---------

abstention update prompt

Use this prompt with your coding agent:

---

You are updating a Python evaluation script to support three ways of calling an LLM:

1. Hugging Face Inference API with a model id (`--model-api-mode hf_api`), using `huggingface_hub.InferenceClient` (already implemented via `_get_hf_client`).
2. Hugging Face Inference Endpoint with the **standard** toolkit / custom `handler.py` (`--model-api-mode hf_endpoint`), using `InferenceClient.text_generation` (already implemented via `_get_hf_endpoint_client`).
3. Hugging Face **vLLM Engine** endpoint that exposes **OpenAI‑style HTTP APIs** (`/v1/chat/completions` and `/v1/completions`) with a `model` field.

You must modify the script below so that:

- `--model-api-mode` gains a new option: `hf_vllm`.
- When `--model-api-mode hf_vllm` is used, and `--model-endpoint-url` is provided, the script calls the model using the OpenAI‑style JSON format against:
  - `/v1/chat/completions` if available (preferred), or
  - `/v1/completions` as a fallback
  on that endpoint URL.
- The vLLM client object must expose the **same interface** used by `_predict_abstain_strict`, i.e.:

  ```python
  if hasattr(llm_client, "chat"):
      response = llm_client.chat.completions.create(...)
      raw_text = response.choices[0].message.content
  else:
      raw = llm_client.generate(prompt, ...)
      raw_text = str(raw)
  ```

  So your vLLM wrapper must define:
  - `llm_client.chat.completions.create(model, messages, temperature, max_tokens, **kw)` (OpenAI‑style),
  - OR at least `llm_client.chat` with a `.completions.create(...)` method that returns a namespace with `choices[0].message.content`.

- Use `requests` and the environment variable `HF_TOKEN` (or `HUGGINGFACE_TOKEN`) as Bearer auth for the vLLM endpoint.
- The vLLM wrapper should:
  - Take a `base_url` and a `model_name` (use `args.model_endpoint_model` if non‑empty, else `args.model`).
  - For `chat.completions.create(...)`:
    - POST to `f"{base_url.rstrip('/')}/v1/chat/completions"`,
    - JSON body:

      ```json
      {
        "model": "<model_name>",
        "messages": [...],
        "max_tokens": <int>,
        "temperature": <float>
      }
      ```

    - Return a `SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=...))])`.

  - Optionally add a `.generate(prompt, temperature, max_tokens, **kw)` method that uses `/v1/completions`:

    ```json
    {
      "model": "<model_name>",
      "prompt": "<prompt>",
      "max_tokens": <int>,
      "temperature": <float>
    }
    ```

    and returns the generated text (e.g. `choices[0].text`).

- On HTTP errors for vLLM, raise a clear `RuntimeError` including status code and response text.

- The rest of the script’s logic (metrics, CSV writing, JSON parsing) must remain unchanged.

The current script is run_abstention_eval_updated.py; modify it to satisfy the above requirements, adding any necessary imports (e.g. `requests`) and the new `_get_vllm_endpoint_client` helper, and then integrating it into `main()` based on `args.model_api_mode`:





---------
eval improvement 
Here’s a prompt you can give to a coding LLM to patch your evaluation script.

---

You are updating a Python evaluation script that computes answer faithfulness using LLM judges. The relevant pieces are `_judge_faithfulness` and `_summarize` in `eval/run_faithfulness_eval_updated.py` (or equivalent).

Current behavior:

- `_judge_faithfulness` parses the judge’s TSV/JSON output into:
  ```python
  {
      "claims": [...],
      "counts": {
          "total": total_claims,
          "supported": supported,
          "contradicted": contradicted,
          "not_found": not_found,
          "unclear": unclear,
      },
      "overall": {
          "faithful": faithful,  # recomputed
          "notes": notes_from_judge,
      },
      "raw_judge_response": raw,
  }
  ```
- `faithful` is currently computed as:
  ```python
  total_claims = len(normalized_claims)
  faithful = bool(total_claims > 0 and contradicted == 0 and not_found == 0)
  ```
  So answers with only `unclear` claims are treated as faithful.
- `_summarize` later aggregates these `counts` and `overall["faithful"]` values.

I want you to patch the script with these changes:

1. **Stricter definition of faithfulness in `_judge_faithfulness`**

   - Keep the existing parsing of claims and counts.
   - After computing `supported`, `contradicted`, `not_found`, `unclear`, and `total_claims`, change the `faithful` computation to:

   ```python
   # New definition:
   # - At least one supported claim
   # - No contradicted or not_found claims
   # - Unclear claims are allowed but counted as "risky"

   total_claims = len(normalized_claims)
   has_supported = supported > 0
   has_contradicted_or_not_found = (contradicted + not_found) > 0

   faithful = bool(
       total_claims > 0
       and has_supported
       and not has_contradicted_or_not_found
   )
   ```

   - Keep the existing `notes` coming from the parsed judge output, but do **not** overwrite `faithful` with the judge’s OVERALL flag; we want to rely on our own rule.

2. **Add a “risky” indicator at the per-answer level**

   Still inside `_judge_faithfulness`, compute:

   ```python
   risky = bool(
       (contradicted + not_found + unclear) > 0
   )
   ```

   And include it in the return dict:

   ```python
   return {
       "claims": normalized_claims,
       "counts": {
           "total": total_claims,
           "supported": supported,
           "contradicted": contradicted,
           "not_found": not_found,
           "unclear": unclear,
       },
       "overall": {
           "faithful": faithful,
           "risky": risky,
           "notes": str((parsed.get("overall") or {}).get("notes", "")).strip(),
       },
       "raw_judge_response": raw,
   }
   ```

   Make sure this new `risky` field is added in both external and Qwen judge outputs and propagated through to `per_query` entries.

3. **Adjust `_summarize` to surface “risky answer” metrics**

   `_summarize(judgments: list[dict[str, Any]])` currently computes:

   ```python
   total_claims = sum(j["counts"]["total"] for j in judgments)
   ...
   faithful_queries = sum(1 for j in judgments if j["overall"]["faithful"])
   contradicted_queries = ...
   not_found_queries = ...
   unfaithful_queries = ...
   ```

   Extend it to:

   - Count how many answers are “risky”:

     ```python
     risky_queries = sum(
         1
         for j in judgments
         if bool((j.get("overall") or {}).get("risky", False))
     )
     ```

   - In the returned dict, add these fields:

     ```python
     "risky_query_rate": _safe_div(risky_queries, n_queries),
     "risky_queries": risky_queries,
     ```

   Do not change the existing fields; just add the new ones.

4. **(Optional but preferred) Add a comment clarifying semantics**

   In `_judge_faithfulness`, near the new logic, add a short comment explaining:

   ```python
   # Faithfulness definition:
   # - faithful = at least one supported claim, and no contradicted/not_found claims.
   # - unclear claims do not by themselves make an answer unfaithful, but they
   #   contribute to "risky" = True, which is summarized separately.
   ```

5. **Ensure backward compatibility**

   - Do not rename existing keys (`claims`, `counts`, `overall["faithful"]`, etc.).
   - Only **add** `overall["risky"]` and the two new summary metrics in `_summarize`.
   - Make sure any code that already accesses `judgments["external"]["overall"]["faithful"]` or `judgments["qwen"]["overall"]["faithful"]` continues to work unchanged.

Here is the current implementation of `_judge_faithfulness` and `_summarize`; apply the modifications in-place:

```python
<PASTE JUST THE CURRENT DEFINITIONS OF _judge_faithfulness AND _summarize HERE>
```

Return the patched versions of these two functions only, with all changes applied exactly as requested.