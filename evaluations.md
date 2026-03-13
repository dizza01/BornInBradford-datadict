# LLM Evaluation Framework

This document outlines a measurable evaluation approach for the BiB research assistant, especially if the goal is to compare the current system against a fine-tuned model.

## Recommended Evaluation Design

The strongest approach is a **fixed benchmark with a frozen evaluation protocol**.

### 1. Build a held-out test set

Create a benchmark of around **100–300 questions** that are not used in tuning.

Include a mix of:

- **Variable discovery**  
  e.g. *What anxiety variables exist in Age of Wonder?*
- **Study or wave identification**  
  e.g. *Which study collected this variable?*
- **Table recommendation**  
  e.g. *Which tables contain maternal BMI?*
- **Covariate suggestion**  
  e.g. *What confounders should be considered for child obesity?*
- **Evidence linking**  
  e.g. *What BiB paper discusses EPDS?*
- **Negative / abstention cases**  
  e.g. *Is gestation_at_recruitment_weeks a BiB variable?*

Negative questions are important for measuring hallucination.

---

## Retrieval and Generation Should Be Evaluated Separately

Because this is a RAG system, do not evaluate only the final answer.

### A. Retrieval metrics

Measure whether the correct items were retrieved at all.

Useful metrics:

- **Recall@k**
- **MRR** (Mean Reciprocal Rank)
- **nDCG@k**

Examples:

- Was the correct variable in the top 10?
- Was the correct table in the top 5?
- Was the correct paper in the top 5?

### B. Answer metrics

Then score the generated answer itself.

---

## Most Useful Measurable Metrics for This System

### 1. Variable grounding accuracy

For each answer, check:

- Are all mentioned variables real?
- Do they exist in the metadata registry?

Suggested metrics:

- **% of answers with zero invented variables**
- **Hallucinated variable rate per answer**

This is one of the most important metrics for BiB.

### 2. Exact match on key entities

For questions with known gold answers, compare:

- variable names
- table IDs
- study / wave labels
- paper titles

Use:

- **Precision**
- **Recall**
- **F1**

Example:

If the gold set is `{rcad_ga, rcad_sep, rcad_soc}` and the model returns `{rcad_ga, rcad_sep, dental_ga}`, score this using entity-level precision and recall.

### 3. Study-context accuracy

Now that the registry includes derived `study_context`, measure whether the model assigns the correct study label, for example:

- `BiB 1000 (12m)`
- `Age of Wonder`
- `Growing Up`
- `BiBBS`

Metric:

- **Study-context accuracy**

### 4. Abstention correctness

The model should say *not found* when the answer is not supported.

Measure:

- **True abstain rate** on impossible questions
- **False answer rate** on impossible questions

This is essential for measuring hallucination reduction.

### 5. Faithfulness / groundedness

Assess whether claims are supported by the retrieved context.

Possible metrics:

- Human annotation: **supported / partially supported / unsupported**
- A simpler summary metric: **% unsupported claims per answer**

### 6. Answer usefulness

Use a small blinded human evaluation with 2–3 raters.

Score each answer from 1–5 on:

- correctness
- completeness
- usefulness for a researcher
- clarity

Report:

- mean score
- inter-rater agreement

---

## Best Comparison for Fine-Tuning

If the goal is to evaluate fine-tuning, keep everything else fixed.

### Compare:

1. **Baseline RAG**
2. **Fine-tuned model + same RAG**
3. optionally **Fine-tuned model without RAG** for a sanity check

### Freeze:

- same retrieval index
- same benchmark questions
- same prompt structure
- same decoding settings

This makes model differences interpretable.

---

## Recommended Primary Success Metrics

If the goal is to show measurable benefit from fine-tuning, use these as headline metrics:

- **Entity F1** for variables, tables, and studies
- **Hallucinated variable rate**
- **Abstention accuracy**
- **Faithfulness rate**
- **Human usefulness score**

These are much more meaningful than BLEU or ROUGE for this application.

---

## Important Interpretation

For this system, fine-tuning may improve:

- style
- fluency
- domain phrasing
- consistency of answer structure

But it may not improve:

- retrieval quality
- grounding to the metadata
- variable validity

Unless the fine-tuning data is explicitly built around grounded QA.

---

## Example Results Table

| Metric | Baseline RAG | Fine-tuned RAG |
|---|---:|---:|
| Retrieval Recall@10 | 0.82 | 0.82 |
| Variable Entity F1 | 0.71 | 0.78 |
| Hallucinated Variable Rate | 18% | 7% |
| Study-context Accuracy | 74% | 83% |
| Abstention Accuracy | 61% | 79% |
| Faithfulness | 80% | 88% |
| Human Usefulness (1–5) | 3.7 | 4.2 |

---

## Best Next Step

Before fine-tuning, create a **gold evaluation set**.

Without that, there is no credible way to measure improvement.

A good benchmark schema should include:

- question
- gold variables
- gold tables
- gold study context
- gold papers (if applicable)
- whether abstention is the correct outcome
- optional human quality rating fields
