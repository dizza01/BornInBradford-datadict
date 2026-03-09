# Evaluation folder

This folder contains the minimal evaluation setup for the BiB research assistant.

The purpose of the evaluation is to measure whether changes to the system actually improve performance. That includes changes to:

- the LLM
- the prompt
- retrieval settings
- indexing
- grounding rules

The evaluation is designed to be lightweight. It does **not** require full gold written answers. Instead, it uses a **query-to-relevant-items benchmark**.

## What this evaluation does

For each benchmark question, the evaluator:

1. sends the question to the current RAG system
2. stores the retrieved context and the model's answer
3. extracts predicted variables, tables, and study labels from the answer
4. compares those predictions against the gold labels in the benchmark file
5. computes simple scores that can be compared over time

This makes it possible to tell whether a new version is better than the current baseline.

## What is in this folder

- `benchmark_minimal.json` — the benchmark questions and gold labels
- `run_benchmark.py` — the evaluator script
- `runs/` — saved raw model outputs for each benchmark run
- `reports/` — saved summary metrics for each benchmark run

## Benchmark schema

Each benchmark item should contain:

- `id` — stable evaluation item ID
- `question` — the user query
- `task_type` — one of `variable_discovery`, `table_lookup`, `study_identification`, `paper_linking`, `abstention`
- `gold_variables` — canonical variable IDs expected in a good answer
- `gold_tables` — relevant table IDs
- `gold_study_context` — correct study or wave labels
- `gold_papers` — relevant paper IDs or titles if needed
- `should_abstain` — `true` when the correct behaviour is to say the evidence is not available
- `notes` — optional annotation guidance

## Metrics currently scored

The evaluator currently measures:

- variable precision / recall / F1
- table precision / recall / F1
- study-context accuracy
- abstention accuracy
- hallucinated variable rate

These are the most useful early metrics for this project because they focus on grounding and hallucination rather than prose style.

## How to run it

From [llm_poc](llm_poc):

```bash
../../.venv/bin/python eval/run_benchmark.py
```

Optional examples:

```bash
# Run only the first 5 items
../../.venv/bin/python eval/run_benchmark.py --max-items 5

# Use a different model
../../.venv/bin/python eval/run_benchmark.py --model "HuggingFaceH4/zephyr-7b-beta"
```

## Output files

Each run writes:

- a detailed run file in `runs/` containing questions, answers, extracted predictions, and retrieved context
- a summary file in `reports/` containing the metric totals and scores

This means every evaluation run is saved and can be compared later.

## Current limitation

If a benchmark item still contains placeholder gold labels such as `<canonical_variable_id_1>`, the evaluator will skip scoring for that part of the item.

So the script is ready now, but the benchmark becomes truly meaningful only after the gold labels are replaced with real values from the canonical registry.

## Suggested workflow

1. Build 25-50 benchmark items.
2. Replace placeholders with real canonical labels.
3. Freeze the benchmark in git.
4. Run the current baseline once.
5. Save that result as the reference point.
6. Re-run the exact same benchmark after every model, prompt, or retrieval change.

That gives a simple and credible way to measure whether later improvements worked.
