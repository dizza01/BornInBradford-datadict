#!/usr/bin/env python3
"""Run the BiB benchmark against the current RAG system and score it.

Usage:
  ../../.venv/bin/python eval/run_benchmark.py
  ../../.venv/bin/python eval/run_benchmark.py --benchmark eval/benchmark_minimal.json --max-items 10
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
LLM_POC_DIR = SCRIPT_DIR.parent
DATADICT_DIR = LLM_POC_DIR.parent
RUNS_DIR = SCRIPT_DIR / "runs"
REPORTS_DIR = SCRIPT_DIR / "reports"
DEFAULT_BENCHMARK = SCRIPT_DIR / "benchmark_minimal.json"

import sys
sys.path.insert(0, str(LLM_POC_DIR))

from bib_research_assistant import (  # noqa: E402
    DEFAULT_MODEL,
    _check_index,
    _get_hf_client,
    get_chroma_client,
    query,
    retrieve_context,
)
from server import _get_variable_registry  # noqa: E402


ABSTAIN_PATTERNS = [
    r"\bnot found\b",
    r"\bno evidence\b",
    r"\bnot supported\b",
    r"\bnot available\b",
    r"\bnot in (?:the )?(?:data|dataset|registry|metadata)\b",
    r"\bunable to find\b",
    r"\bcould not find\b",
    r"\bcannot find\b",
    r"\bcan't find\b",
    r"\bdo not see evidence\b",
    r"\bno matching variable\b",
    r"\bno such variable\b",
    r"\bi cannot confirm\b",
    r"\bcan't confirm\b",
]


@dataclass
class MetricCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, gold: set[str], pred: set[str]) -> None:
        self.tp += len(gold & pred)
        self.fp += len(pred - gold)
        self.fn += len(gold - pred)

    def to_dict(self) -> dict[str, float | int]:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }


@dataclass
class RegistryLookups:
    variables: dict[str, str]
    tables: dict[str, str]
    study_contexts: list[str]


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Benchmark file must contain a JSON array of items.")
    return data


def build_registry_lookups() -> RegistryLookups:
    registry = _get_variable_registry()
    variable_map: dict[str, str] = {}
    table_map: dict[str, str] = {}
    study_contexts: set[str] = set()

    for row in registry["rows"]:
        variable = str(row.get("variable", "")).strip()
        table = str(row.get("table", "")).strip()
        study = str(row.get("study_context", "")).strip()
        if variable:
            variable_map[variable.lower()] = variable
        if table:
            table_map[table.lower()] = table
        if study:
            study_contexts.add(study)

    ordered_contexts = sorted(study_contexts, key=lambda s: (-len(s), s.lower()))
    return RegistryLookups(
        variables=variable_map,
        tables=table_map,
        study_contexts=ordered_contexts,
    )


def contains_placeholder(values: list[str]) -> bool:
    return any(isinstance(v, str) and "<" in v and ">" in v for v in values)


def normalise_str_list(values: list[Any]) -> list[str]:
    out: list[str] = []
    for value in values or []:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def token_candidates(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_.]*\b", text)}


def extract_variables(answer_text: str, lookups: RegistryLookups) -> list[str]:
    candidates = token_candidates(answer_text)
    found = [lookups.variables[token] for token in candidates if token in lookups.variables]
    return sorted(set(found), key=str.lower)


def extract_tables(answer_text: str, lookups: RegistryLookups) -> list[str]:
    candidates = token_candidates(answer_text)
    found = [lookups.tables[token] for token in candidates if token in lookups.tables]
    return sorted(set(found), key=str.lower)


def extract_study_contexts(answer_text: str, lookups: RegistryLookups) -> list[str]:
    haystack = answer_text.lower()
    found: list[str] = []
    for label in lookups.study_contexts:
        if label.lower() in haystack:
            found.append(label)
    return found


def detect_abstention(answer_text: str) -> bool:
    haystack = answer_text.lower()
    return any(re.search(pattern, haystack, re.I) for pattern in ABSTAIN_PATTERNS)


def score_exact_set(gold: set[str], pred: set[str]) -> bool:
    return gold == pred


def score_overlap_set(gold: set[str], pred: set[str]) -> bool:
    if not gold and not pred:
        return True
    if not gold:
        return False
    return bool(gold & pred)


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "model"


def ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_item(
    item: dict[str, Any],
    client: Any,
    llm_client: Any,
    model: str,
    lookups: RegistryLookups,
) -> dict[str, Any]:
    question = str(item["question"])
    context = retrieve_context(question, client)
    answer = query(question, client, llm_client, model=model)

    predicted_variables = extract_variables(answer, lookups)
    predicted_tables = extract_tables(answer, lookups)
    predicted_study_context = extract_study_contexts(answer, lookups)
    abstained = detect_abstention(answer)

    gold_variables = normalise_str_list(item.get("gold_variables", []))
    gold_tables = normalise_str_list(item.get("gold_tables", []))
    gold_study_context = normalise_str_list(item.get("gold_study_context", []))
    gold_papers = normalise_str_list(item.get("gold_papers", []))
    should_abstain = bool(item.get("should_abstain", False))

    skipped_metrics: list[str] = []
    if contains_placeholder(gold_variables):
        skipped_metrics.append("variables")
    if contains_placeholder(gold_tables):
        skipped_metrics.append("tables")
    if contains_placeholder(gold_study_context):
        skipped_metrics.append("study_context")

    return {
        "eval_id": item.get("id", ""),
        "question": question,
        "task_type": item.get("task_type", ""),
        "timestamp": iso_now(),
        "model": model,
        "should_abstain": should_abstain,
        "gold_variables": gold_variables,
        "gold_tables": gold_tables,
        "gold_study_context": gold_study_context,
        "gold_papers": gold_papers,
        "context": context,
        "answer_text": answer,
        "predicted_variables": predicted_variables,
        "predicted_tables": predicted_tables,
        "predicted_study_context": predicted_study_context,
        "predicted_papers": [],
        "abstained": abstained,
        "skipped_metrics": skipped_metrics,
    }


def summarize(results: list[dict[str, Any]], benchmark_path: Path) -> dict[str, Any]:
    variable_counts = MetricCounts()
    table_counts = MetricCounts()
    study_total = 0
    study_correct = 0
    abstain_total = 0
    abstain_correct = 0
    hallucinated_variables = 0
    answers_with_hallucinations = 0
    skipped = Counter()

    for result in results:
        skipped.update(result.get("skipped_metrics", []))

        gold_vars = set(normalise_str_list(result.get("gold_variables", [])))
        pred_vars = set(normalise_str_list(result.get("predicted_variables", [])))
        if "variables" not in result.get("skipped_metrics", []):
            variable_counts.add(gold_vars, pred_vars)
            hallucinations = len(pred_vars - gold_vars)
            hallucinated_variables += hallucinations
            if hallucinations > 0:
                answers_with_hallucinations += 1

        gold_tables = set(normalise_str_list(result.get("gold_tables", [])))
        pred_tables = set(normalise_str_list(result.get("predicted_tables", [])))
        if "tables" not in result.get("skipped_metrics", []):
            table_counts.add(gold_tables, pred_tables)

        gold_studies = set(normalise_str_list(result.get("gold_study_context", [])))
        pred_studies = set(normalise_str_list(result.get("predicted_study_context", [])))
        if "study_context" not in result.get("skipped_metrics", []):
            study_total += 1
            if score_overlap_set(gold_studies, pred_studies):
                study_correct += 1

        abstain_total += 1
        if bool(result.get("should_abstain")) == bool(result.get("abstained")):
            abstain_correct += 1

    study_accuracy = study_correct / study_total if study_total else 0.0
    abstention_accuracy = abstain_correct / abstain_total if abstain_total else 0.0
    hallucinated_var_rate = hallucinated_variables / len(results) if results else 0.0
    hallucination_answer_rate = answers_with_hallucinations / len(results) if results else 0.0

    return {
        "benchmark_file": str(benchmark_path),
        "evaluated_items": len(results),
        "skipped_metric_counts": dict(skipped),
        "variable_metrics": variable_counts.to_dict(),
        "table_metrics": table_counts.to_dict(),
        "study_context_accuracy": {
            "correct": study_correct,
            "total": study_total,
            "accuracy": round(study_accuracy, 4),
        },
        "abstention_accuracy": {
            "correct": abstain_correct,
            "total": abstain_total,
            "accuracy": round(abstention_accuracy, 4),
        },
        "hallucinated_variable_rate_per_item": round(hallucinated_var_rate, 4),
        "answers_with_hallucinated_variables_rate": round(hallucination_answer_rate, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal benchmark against the current BiB RAG system.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK, help="Path to benchmark JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-items", type=int, default=0, help="Optional cap on number of benchmark items to run")
    parser.add_argument("--output-prefix", default="", help="Optional prefix for output filenames")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_path = args.benchmark.resolve()
    if not benchmark_path.exists():
        print(f"❌ Benchmark file not found: {benchmark_path}")
        return 1

    benchmark = load_benchmark(benchmark_path)
    if args.max_items > 0:
        benchmark = benchmark[: args.max_items]
    if not benchmark:
        print("❌ Benchmark is empty.")
        return 1

    ensure_dirs()

    lookups = build_registry_lookups()
    client = get_chroma_client()
    if not _check_index(client):
        return 1

    llm_client = _get_hf_client(args.model)
    if llm_client is None:
        return 1

    print(f"▶ Running {len(benchmark)} benchmark items with model: {args.model}")
    results: list[dict[str, Any]] = []
    for idx, item in enumerate(benchmark, start=1):
        eval_id = item.get("id", f"item_{idx}")
        question = str(item.get("question", "")).strip()
        print(f"[{idx}/{len(benchmark)}] {eval_id}: {question}")
        try:
            result = run_item(item, client, llm_client, args.model, lookups)
            results.append(result)
        except Exception as exc:
            print(f"   ❌ Failed: {exc}")
            results.append({
                "eval_id": eval_id,
                "question": question,
                "task_type": item.get("task_type", ""),
                "timestamp": iso_now(),
                "model": args.model,
                "error": str(exc),
                "gold_variables": normalise_str_list(item.get("gold_variables", [])),
                "gold_tables": normalise_str_list(item.get("gold_tables", [])),
                "gold_study_context": normalise_str_list(item.get("gold_study_context", [])),
                "gold_papers": normalise_str_list(item.get("gold_papers", [])),
                "predicted_variables": [],
                "predicted_tables": [],
                "predicted_study_context": [],
                "predicted_papers": [],
                "abstained": False,
                "skipped_metrics": ["variables", "tables", "study_context"],
                "should_abstain": bool(item.get("should_abstain", False)),
                "answer_text": "",
                "context": "",
            })

    summary = summarize(results, benchmark_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{safe_slug(args.output_prefix)}_" if args.output_prefix else ""
    model_slug = safe_slug(args.model)
    run_path = RUNS_DIR / f"{prefix}{timestamp}_{model_slug}.json"
    report_path = REPORTS_DIR / f"{prefix}{timestamp}_{model_slug}_summary.json"

    with run_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n✅ Evaluation complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved run details: {run_path}")
    print(f"Saved summary:     {report_path}")
    if summary.get("skipped_metric_counts"):
        print("\n⚠ Some metrics were skipped because benchmark items still contain placeholder gold labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
