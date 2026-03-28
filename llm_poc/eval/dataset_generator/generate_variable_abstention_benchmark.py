"""Generate a safeguarded abstention benchmark from BiB registries.

The benchmark includes a balanced mix of answerable (should_abstain=false)
and unanswerable (should_abstain=true) questions. Questions are generated from
registry-backed entities and validated before writing.

Usage examples
--------------

    ../../.venv/bin/python eval/dataset_generator/generate_variable_abstention_benchmark.py

    ../../.venv/bin/python eval/dataset_generator/generate_variable_abstention_benchmark.py \
        --n-examples 600 \
        --positive-ratio 0.5 \
        --dataset-version abstention_benchmark_v1 \
        --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "dataset_generator" else SCRIPT_DIR
DATASETS_DIR = EVAL_DIR / "evaluation_datasets"

ROOT_DIR = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "docs" / "csv").exists() and (parent / "papers").exists()
    ),
    None,
)
if ROOT_DIR is None:
    raise RuntimeError("Could not locate datadict root (expected docs/csv and papers folders).")

LLM_POC_DIR = ROOT_DIR / "llm_poc"
if str(LLM_POC_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_POC_DIR))

from bib_research_assistant import CHROMA_DIR, get_chroma_client

DEFAULT_VARIABLES_CSV = ROOT_DIR / "docs" / "csv" / "all_variables_meta.csv"
DEFAULT_TABLES_CSV = ROOT_DIR / "docs" / "csv" / "all_tables.csv"
DEFAULT_OUTPUT = DATASETS_DIR / "variable_abstention" / "abstention_benchmark.jsonl"
DEFAULT_REPORT = DATASETS_DIR / "variable_abstention" / "abstention_benchmark_report.json"
DEFAULT_QA_SAMPLE = DATASETS_DIR / "variable_abstention" / "abstention_benchmark_qa_sample.jsonl"


@dataclass(frozen=True)
class VariableRecord:
    variable: str
    table_id: str
    project: str


@dataclass(frozen=True)
class TableRecord:
    table_id: str
    project_name: str


def _load_index_snapshot() -> dict[str, Any]:
    client = get_chroma_client()

    try:
        variables_col = client.get_collection("bib_variables")
        variables_rows = variables_col.get(include=["metadatas"])
    except Exception as exc:
        raise RuntimeError(
            "Could not read 'bib_variables' collection from Chroma index. "
            "Build index first with: python bib_research_assistant.py --build"
        ) from exc

    try:
        tables_col = client.get_collection("bib_tables")
        tables_rows = tables_col.get(include=["metadatas"])
    except Exception as exc:
        raise RuntimeError(
            "Could not read 'bib_tables' collection from Chroma index. "
            "Build index first with: python bib_research_assistant.py --build"
        ) from exc

    var_metas = variables_rows.get("metadatas", []) or []
    table_metas = tables_rows.get("metadatas", []) or []

    indexed_variables: set[str] = set()
    indexed_variable_pairs: set[tuple[str, str]] = set()
    for meta in var_metas:
        meta = meta or {}
        variable = str(meta.get("variable", "")).strip()
        table_id = str(meta.get("table_id", "")).strip()
        if variable:
            indexed_variables.add(variable)
        if variable and table_id:
            indexed_variable_pairs.add((variable, table_id))

    indexed_tables: set[str] = set()
    for meta in table_metas:
        meta = meta or {}
        table_id = str(meta.get("table_id", "")).strip()
        if table_id:
            indexed_tables.add(table_id)

    snapshot_material = "\n".join(
        sorted(
            [f"var::{name}" for name in indexed_variables]
            + [f"pair::{variable}::{table_id}" for variable, table_id in indexed_variable_pairs]
            + [f"table::{table_id}" for table_id in indexed_tables]
        )
    )
    snapshot_id = hashlib.md5(snapshot_material.encode("utf-8")).hexdigest()[:16]

    return {
        "snapshot_id": snapshot_id,
        "chroma_dir": str(CHROMA_DIR),
        "indexed_variables": indexed_variables,
        "indexed_variable_pairs": indexed_variable_pairs,
        "indexed_tables": indexed_tables,
        "collection_counts": {
            "bib_variables": variables_col.count(),
            "bib_tables": tables_col.count(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate abstention benchmark with safeguards.")
    parser.add_argument("--variables-csv", type=Path, default=DEFAULT_VARIABLES_CSV)
    parser.add_argument("--tables-csv", type=Path, default=DEFAULT_TABLES_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--qa-sample-output", type=Path, default=DEFAULT_QA_SAMPLE)
    parser.add_argument("--n-examples", type=int, default=500)
    parser.add_argument("--positive-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-version", type=str, default="abstention_benchmark_v1")
    parser.add_argument("--qa-sample-fraction", type=float, default=0.15)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-attempt-multiplier", type=int, default=25)
    parser.add_argument(
        "--disable-index-snapshot-filter",
        action="store_true",
        help="Do not restrict in-scope generation to entities present in current Chroma index snapshot.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _split_from_group(group_key: str, train_ratio: float, dev_ratio: float) -> str:
    digest = hashlib.md5(group_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + dev_ratio:
        return "dev"
    return "test"


def _load_variables(path: Path) -> list[VariableRecord]:
    rows: list[VariableRecord] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            variable = str(row.get("variable", "")).strip()
            table_id = str(row.get("table_id", "")).strip()
            project = str(row.get("project", "")).strip()
            if not variable or not table_id:
                continue
            key = (variable, table_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append(VariableRecord(variable=variable, table_id=table_id, project=project))
    return rows


def _load_tables(path: Path) -> list[TableRecord]:
    rows: list[TableRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            table_id = str(row.get("table_id", "")).strip()
            project_name = str(row.get("project_name", "")).strip()
            if not table_id or table_id in seen:
                continue
            seen.add(table_id)
            rows.append(TableRecord(table_id=table_id, project_name=project_name))
    return rows


def _perturb_identifier(text: str, rng: random.Random) -> str:
    if not text:
        return text
    chars = list(text)
    if len(chars) >= 3 and rng.random() < 0.5:
        idx = rng.randint(1, len(chars) - 2)
        chars[idx] = chars[idx - 1]
    else:
        idx = rng.randint(0, max(0, len(chars) - 1))
        substitute = rng.choice("abcdefghijklmnopqrstuvwxyz0123456789")
        chars[idx] = substitute
    if rng.random() < 0.4:
        suffix = rng.choice(["_x", "_v2", "_tmp", "_legacy"])
        chars.extend(list(suffix))
    return "".join(chars)


def _build_registries(
    variables: list[VariableRecord],
    tables: list[TableRecord],
) -> dict[str, Any]:
    variable_names = {v.variable for v in variables}
    table_ids = {t.table_id for t in tables}
    projects = {t.project_name for t in tables if t.project_name}

    variable_to_tables: dict[str, set[str]] = defaultdict(set)
    variable_to_projects: dict[str, set[str]] = defaultdict(set)
    table_to_project: dict[str, str] = {}

    for v in variables:
        variable_to_tables[v.variable].add(v.table_id)
        if v.project:
            variable_to_projects[v.variable].add(v.project)

    for t in tables:
        table_to_project[t.table_id] = t.project_name

    return {
        "variable_names": variable_names,
        "table_ids": table_ids,
        "projects": projects,
        "variable_to_tables": variable_to_tables,
        "variable_to_projects": variable_to_projects,
        "table_to_project": table_to_project,
    }


def _question_record(
    *,
    query_id: str,
    question: str,
    should_abstain: bool,
    reason: str,
    generation_type: str,
    group_key: str,
    split: str,
    dataset_version: str,
    source_scope: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "question": question,
        "should_abstain": should_abstain,
        "reason": reason,
        "generation_type": generation_type,
        "group_key": group_key,
        "split": split,
        "dataset_version": dataset_version,
        "source_scope": source_scope,
        "metadata": metadata,
    }


def _validate_record(record: dict[str, Any], registries: dict[str, Any]) -> bool:
    generation_type = record["generation_type"]
    should_abstain = bool(record["should_abstain"])
    metadata = record.get("metadata", {})

    variable_names = registries["variable_names"]
    table_ids = registries["table_ids"]
    projects = registries["projects"]
    variable_to_tables = registries["variable_to_tables"]
    table_to_project = registries["table_to_project"]

    if generation_type in {"var_exists_positive", "var_exists_negative"}:
        variable = str(metadata.get("variable", ""))
        return (variable in variable_names) != should_abstain

    if generation_type in {"table_exists_positive", "table_exists_negative"}:
        table_id = str(metadata.get("table_id", ""))
        return (table_id in table_ids) != should_abstain

    if generation_type in {"var_table_pair_positive", "var_table_pair_negative"}:
        variable = str(metadata.get("variable", ""))
        table_id = str(metadata.get("table_id", ""))
        is_true_pair = variable in variable_to_tables and table_id in variable_to_tables[variable]
        return is_true_pair != should_abstain

    if generation_type in {"table_project_positive", "table_project_negative"}:
        table_id = str(metadata.get("table_id", ""))
        project_name = str(metadata.get("project_name", ""))
        exists = table_id in table_to_project and table_to_project.get(table_id) == project_name
        return exists != should_abstain

    if generation_type == "out_of_scope_negative":
        return should_abstain and str(metadata.get("domain", "")) in {"non_bib", "external"}

    return False


def _generate_positive(
    rng: random.Random,
    variables: list[VariableRecord],
    tables: list[TableRecord],
    registries: dict[str, Any],
) -> tuple[str, str, str, dict[str, Any], str]:
    template = rng.choice(
        [
            "var_exists_positive",
            "table_exists_positive",
            "var_table_pair_positive",
            "table_project_positive",
        ]
    )

    if template == "var_exists_positive":
        v = rng.choice(variables)
        question = f"Is {v.variable} a BiB variable in the datasphere?"
        reason = "variable exists in registry"
        metadata = {"variable": v.variable, "table_id": v.table_id, "project_name": v.project}
        group_key = f"variable::{v.variable}"
        return question, reason, template, metadata, group_key

    if template == "table_exists_positive":
        t = rng.choice(tables)
        question = f"Does the table {t.table_id} exist in the BiB datasphere?"
        reason = "table exists in registry"
        metadata = {"table_id": t.table_id, "project_name": t.project_name}
        group_key = f"table::{t.table_id}"
        return question, reason, template, metadata, group_key

    if template == "var_table_pair_positive":
        v = rng.choice(variables)
        question = f"Is variable {v.variable} collected in table {v.table_id}?"
        reason = "variable-table mapping exists in registry"
        metadata = {"variable": v.variable, "table_id": v.table_id, "project_name": v.project}
        group_key = f"variable::{v.variable}"
        return question, reason, template, metadata, group_key

    t = rng.choice(tables)
    project = t.project_name
    question = f"Is table {t.table_id} part of project {project}?"
    reason = "table-project mapping exists in registry"
    metadata = {"table_id": t.table_id, "project_name": project}
    group_key = f"table::{t.table_id}"
    return question, reason, "table_project_positive", metadata, group_key


def _generate_negative(
    rng: random.Random,
    variables: list[VariableRecord],
    tables: list[TableRecord],
    registries: dict[str, Any],
) -> tuple[str, str, str, dict[str, Any], str]:
    template = rng.choice(
        [
            "var_exists_negative",
            "table_exists_negative",
            "var_table_pair_negative",
            "table_project_negative",
            "out_of_scope_negative",
        ]
    )

    variable_names = registries["variable_names"]
    table_ids = registries["table_ids"]

    if template == "var_exists_negative":
        base = rng.choice(variables)
        candidate = _perturb_identifier(base.variable, rng)
        retries = 0
        while candidate in variable_names and retries < 20:
            candidate = _perturb_identifier(base.variable, rng)
            retries += 1
        question = f"Is {candidate} a BiB variable in the datasphere?"
        reason = "variable not in registry"
        metadata = {"variable": candidate, "derived_from": base.variable, "table_id": base.table_id}
        group_key = f"variable::{base.variable}"
        return question, reason, template, metadata, group_key

    if template == "table_exists_negative":
        base = rng.choice(tables)
        candidate = _perturb_identifier(base.table_id, rng)
        retries = 0
        while candidate in table_ids and retries < 20:
            candidate = _perturb_identifier(base.table_id, rng)
            retries += 1
        question = f"Does the table {candidate} exist in the BiB datasphere?"
        reason = "table not in registry"
        metadata = {"table_id": candidate, "derived_from": base.table_id}
        group_key = f"table::{base.table_id}"
        return question, reason, template, metadata, group_key

    if template == "var_table_pair_negative":
        v = rng.choice(variables)
        max_tries = 50
        selected_table = ""
        for _ in range(max_tries):
            t = rng.choice(tables)
            if t.table_id != v.table_id:
                selected_table = t.table_id
                break
        if not selected_table:
            selected_table = v.table_id + "_x"
        question = f"Is variable {v.variable} collected in table {selected_table}?"
        reason = "variable-table mapping does not exist"
        metadata = {"variable": v.variable, "table_id": selected_table, "true_table_id": v.table_id}
        group_key = f"variable::{v.variable}"
        return question, reason, template, metadata, group_key

    if template == "table_project_negative":
        t = rng.choice(tables)
        wrong_project = ""
        project_names = [x.project_name for x in tables if x.project_name and x.project_name != t.project_name]
        if project_names:
            wrong_project = rng.choice(project_names)
        else:
            wrong_project = t.project_name + "_x"
        question = f"Is table {t.table_id} part of project {wrong_project}?"
        reason = "table belongs to different project"
        metadata = {"table_id": t.table_id, "project_name": wrong_project, "true_project_name": t.project_name}
        group_key = f"table::{t.table_id}"
        return question, reason, template, metadata, group_key

    out_questions = [
        "Is the UK Biobank data dictionary part of the BiB datasphere registry?",
        "Does BiB include a table called nhanes_lab_results_2019?",
        "Is variable icd10_primary_diagnosis from NHS HES automatically in the BiB registry?",
        "Can you find a BiB table for NOAA daily temperature summaries?",
        "Does the BiB datasphere include a variable named census_block_population_density_2021?",
    ]
    question = rng.choice(out_questions)
    reason = "question is out of scope for BiB registries"
    metadata = {"domain": "non_bib"}
    group_key = f"oos::{_normalize_text(question)}"
    return question, reason, "out_of_scope_negative", metadata, group_key


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _make_qa_sample(
    records: list[dict[str, Any]],
    fraction: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not records:
        return []
    n_target = max(1, math.ceil(len(records) * fraction))
    by_bucket: dict[tuple[bool, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_bucket[(bool(record["should_abstain"]), str(record["generation_type"]))].append(record)

    sample: list[dict[str, Any]] = []
    buckets = list(by_bucket.keys())
    rng.shuffle(buckets)

    for bucket in buckets:
        bucket_rows = by_bucket[bucket]
        rng.shuffle(bucket_rows)
        take = max(1, round(n_target / max(1, len(buckets))))
        sample.extend(bucket_rows[:take])

    if len(sample) < n_target:
        remaining = [r for r in records if r not in sample]
        rng.shuffle(remaining)
        sample.extend(remaining[: n_target - len(sample)])

    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in sample:
        qid = str(record["query_id"])
        if qid in seen_ids:
            continue
        seen_ids.add(qid)
        deduped.append(record)

    return deduped[:n_target]


def main() -> None:
    args = parse_args()

    if args.n_examples <= 0:
        raise ValueError("--n-examples must be > 0")
    if not (0.0 < args.positive_ratio < 1.0):
        raise ValueError("--positive-ratio must be between 0 and 1")
    if min(args.train_ratio, args.dev_ratio, args.test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")
    if abs((args.train_ratio + args.dev_ratio + args.test_ratio) - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to 1")
    if not (0.0 < args.qa_sample_fraction <= 1.0):
        raise ValueError("--qa-sample-fraction must be in (0, 1]")

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Use --overwrite.")

    variables = _load_variables(args.variables_csv)
    tables = _load_tables(args.tables_csv)
    if not variables:
        raise RuntimeError("No variables loaded from variables CSV")
    if not tables:
        raise RuntimeError("No tables loaded from tables CSV")

    original_variable_count = len(variables)
    original_table_count = len(tables)
    index_snapshot: dict[str, Any] | None = None

    if not args.disable_index_snapshot_filter:
        index_snapshot = _load_index_snapshot()
        indexed_tables = index_snapshot["indexed_tables"]
        indexed_variables = index_snapshot["indexed_variables"]
        indexed_pairs = index_snapshot["indexed_variable_pairs"]

        tables = [table for table in tables if table.table_id in indexed_tables]
        variables = [
            variable
            for variable in variables
            if variable.variable in indexed_variables
            and (variable.variable, variable.table_id) in indexed_pairs
        ]

        if not variables:
            raise RuntimeError(
                "No variables remain after applying index snapshot filter. "
                "Rebuild index with: python bib_research_assistant.py --build"
            )
        if not tables:
            raise RuntimeError(
                "No tables remain after applying index snapshot filter. "
                "Rebuild index with: python bib_research_assistant.py --build"
            )

    registries = _build_registries(variables, tables)

    rng = random.Random(args.seed)
    n_positive = int(round(args.n_examples * args.positive_ratio))
    n_negative = args.n_examples - n_positive

    records: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_pairs: set[tuple[str, bool]] = set()

    max_attempts = args.n_examples * max(1, args.max_attempt_multiplier)
    attempts = 0
    counters = Counter()

    while (counters["positive"] < n_positive or counters["negative"] < n_negative) and attempts < max_attempts:
        attempts += 1
        want_positive = counters["positive"] < n_positive and (
            counters["negative"] >= n_negative or rng.random() < args.positive_ratio
        )

        if want_positive:
            question, reason, generation_type, metadata, group_key = _generate_positive(
                rng=rng,
                variables=variables,
                tables=tables,
                registries=registries,
            )
            should_abstain = False
            label_key = "positive"
        else:
            question, reason, generation_type, metadata, group_key = _generate_negative(
                rng=rng,
                variables=variables,
                tables=tables,
                registries=registries,
            )
            should_abstain = True
            label_key = "negative"

        normalized_question = _normalize_text(question)
        pair_key = (normalized_question, should_abstain)
        if normalized_question in seen_questions or pair_key in seen_pairs:
            continue

        split = _split_from_group(group_key, args.train_ratio, args.dev_ratio)

        record = _question_record(
            query_id=f"abs_{len(records) + 1:05d}",
            question=question,
            should_abstain=should_abstain,
            reason=reason,
            generation_type=generation_type,
            group_key=group_key,
            split=split,
            dataset_version=args.dataset_version,
            source_scope=(
                "outside_index_snapshot"
                if generation_type == "out_of_scope_negative"
                else "in_index_snapshot"
            ),
            metadata=metadata,
        )

        if not _validate_record(record, registries):
            continue

        records.append(record)
        seen_questions.add(normalized_question)
        seen_pairs.add(pair_key)
        counters[label_key] += 1
        counters[f"type::{generation_type}"] += 1
        counters[f"split::{split}"] += 1

    if len(records) < args.n_examples:
        raise RuntimeError(
            f"Generated only {len(records)} records after {attempts} attempts; requested {args.n_examples}."
        )

    _write_jsonl(args.output, records)

    qa_sample = _make_qa_sample(records, args.qa_sample_fraction, rng)
    _write_jsonl(args.qa_sample_output, qa_sample)

    report = {
        "dataset_version": args.dataset_version,
        "index_snapshot_filter_enabled": not args.disable_index_snapshot_filter,
        "index_snapshot_id": (index_snapshot or {}).get("snapshot_id", ""),
        "n_examples": len(records),
        "n_positive": sum(1 for r in records if not r["should_abstain"]),
        "n_negative": sum(1 for r in records if r["should_abstain"]),
        "attempts": attempts,
        "unique_questions": len({_normalize_text(r["question"]) for r in records}),
        "registry_filtering": {
            "variables_before_filter": original_variable_count,
            "variables_after_filter": len(variables),
            "tables_before_filter": original_table_count,
            "tables_after_filter": len(tables),
        },
        "split_distribution": {
            "train": sum(1 for r in records if r["split"] == "train"),
            "dev": sum(1 for r in records if r["split"] == "dev"),
            "test": sum(1 for r in records if r["split"] == "test"),
        },
        "generation_type_distribution": dict(
            sorted(
                Counter(str(r["generation_type"]) for r in records).items(),
                key=lambda item: item[0],
            )
        ),
        "qa_sample_size": len(qa_sample),
        "qa_sample_fraction": args.qa_sample_fraction,
        "sources": {
            "variables_csv": str(args.variables_csv),
            "tables_csv": str(args.tables_csv),
            "chroma_dir": (index_snapshot or {}).get("chroma_dir", ""),
            "index_collection_counts": (index_snapshot or {}).get("collection_counts", {}),
        },
    }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("════════════════════════════════════════════════════")
    print(f"Dataset version:     {args.dataset_version}")
    print(f"Output dataset:      {args.output}")
    print(f"Output report:       {args.report_output}")
    print(f"QA sample:           {args.qa_sample_output}")
    print(f"Total records:       {len(records)}")
    print(f"Positive records:    {report['n_positive']}")
    print(f"Negative records:    {report['n_negative']}")
    if not args.disable_index_snapshot_filter:
        print(f"Index snapshot id:   {report['index_snapshot_id']}")
        print(
            "Registry filtered:   "
            f"variables {original_variable_count}->{len(variables)}, "
            f"tables {original_table_count}->{len(tables)}"
        )
    print(
        "Split distribution: "
        f"train={report['split_distribution']['train']}, "
        f"dev={report['split_distribution']['dev']}, "
        f"test={report['split_distribution']['test']}"
    )
    print("Generation types:")
    for generation_type, count in report["generation_type_distribution"].items():
        print(f"  - {generation_type}: {count}")


if __name__ == "__main__":
    main()
