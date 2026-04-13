"""Generate a paper-specific abstention benchmark from indexed BiB papers.

This module creates answerable and unanswerable questions focused on the paper
layer of the index (`bib_papers`), including both abstract entries and PDF
full-text chunks where available.

Usage examples
--------------

    ../../.venv/bin/python eval/dataset_generator/generate_paper_abstention_benchmark.py

    ../../.venv/bin/python eval/dataset_generator/generate_paper_abstention_benchmark.py \
        --n-examples 400 \
        --positive-ratio 0.5 \
        --dataset-version paper_abstention_benchmark_v1 \
        --overwrite
"""

from __future__ import annotations

import argparse
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

DEFAULT_OUTPUT = DATASETS_DIR / "paper_abstention" / "paper_abstention_benchmark.jsonl"
DEFAULT_REPORT = DATASETS_DIR / "paper_abstention" / "paper_abstention_benchmark_report.json"
DEFAULT_QA_SAMPLE = DATASETS_DIR / "paper_abstention" / "paper_abstention_benchmark_qa_sample.jsonl"


@dataclass(frozen=True)
class PaperRecord:
    title: str
    year: str
    doi: str
    journal: str
    authors: str
    has_pdf_chunk: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-specific abstention benchmark.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--qa-sample-output", type=Path, default=DEFAULT_QA_SAMPLE)
    parser.add_argument("--n-examples", type=int, default=400)
    parser.add_argument("--positive-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-version", type=str, default="paper_abstention_benchmark_v1")
    parser.add_argument("--qa-sample-fraction", type=float, default=0.15)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-attempt-multiplier", type=int, default=25)
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


def _load_indexed_papers() -> dict[str, Any]:
    client = get_chroma_client()
    try:
        papers_col = client.get_collection("bib_papers")
    except Exception as exc:
        raise RuntimeError(
            "Could not read 'bib_papers' collection from Chroma index. "
            "Build index first with: python bib_research_assistant.py --build"
        ) from exc

    rows = papers_col.get(include=["metadatas"])
    metas = rows.get("metadatas", []) or []

    by_title: dict[str, dict[str, Any]] = {}
    for meta in metas:
        meta = meta or {}
        title = str(meta.get("title", "")).strip()
        if not title:
            continue
        key = _normalize_text(title)
        existing = by_title.get(key)
        source = str(meta.get("source", "")).strip().lower()
        is_pdf = source == "pdf_fulltext"

        if not existing:
            by_title[key] = {
                "title": title,
                "year": str(meta.get("year", "")).strip(),
                "doi": str(meta.get("doi", "")).strip(),
                "journal": str(meta.get("journal", "")).strip(),
                "authors": str(meta.get("authors", "")).strip(),
                "has_pdf_chunk": is_pdf,
            }
        else:
            if is_pdf:
                existing["has_pdf_chunk"] = True
            if not existing["doi"] and str(meta.get("doi", "")).strip():
                existing["doi"] = str(meta.get("doi", "")).strip()
            if not existing["year"] and str(meta.get("year", "")).strip():
                existing["year"] = str(meta.get("year", "")).strip()
            if not existing["journal"] and str(meta.get("journal", "")).strip():
                existing["journal"] = str(meta.get("journal", "")).strip()
            if not existing["authors"] and str(meta.get("authors", "")).strip():
                existing["authors"] = str(meta.get("authors", "")).strip()

    papers = [
        PaperRecord(
            title=value["title"],
            year=value["year"],
            doi=value["doi"],
            journal=value["journal"],
            authors=value["authors"],
            has_pdf_chunk=bool(value["has_pdf_chunk"]),
        )
        for value in by_title.values()
    ]

    if not papers:
        raise RuntimeError("No papers found in bib_papers index snapshot.")

    snapshot_material = "\n".join(
        sorted(
            f"{_normalize_text(p.title)}::{p.year}::{p.doi}::{int(p.has_pdf_chunk)}"
            for p in papers
        )
    )
    snapshot_id = hashlib.md5(snapshot_material.encode("utf-8")).hexdigest()[:16]

    return {
        "papers": papers,
        "snapshot_id": snapshot_id,
        "chroma_dir": str(CHROMA_DIR),
        "collection_count": papers_col.count(),
        "n_unique_titles": len(papers),
        "n_titles_with_pdf": sum(1 for p in papers if p.has_pdf_chunk),
    }


def _mutate_title(title: str, rng: random.Random) -> str:
    words = title.split()
    if not words:
        return title
    if len(words) >= 4 and rng.random() < 0.5:
        idx = rng.randint(1, len(words) - 2)
        words[idx] = words[idx - 1]
    else:
        idx = rng.randint(0, len(words) - 1)
        token = words[idx]
        if len(token) > 3:
            token = token[:-1] + rng.choice("abcdefghijklmnopqrstuvwxyz")
        else:
            token = token + rng.choice(["X", "Z", "2027"])
        words[idx] = token
    if rng.random() < 0.35:
        words.append(rng.choice(["trial", "analysis", "dataset", "consortium"]))
    return " ".join(words)


def _mutate_doi(doi: str, rng: random.Random) -> str:
    if not doi:
        return "10.9999/bib." + str(rng.randint(1000, 9999))
    chars = list(doi)
    idx = rng.randint(0, len(chars) - 1)
    chars[idx] = rng.choice("abcdefghijklmnopqrstuvwxyz0123456789")
    mutated = "".join(chars)
    if mutated == doi:
        mutated += "x"
    return mutated


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


def _generate_positive(rng: random.Random, papers: list[PaperRecord]) -> tuple[str, str, str, dict[str, Any], str]:
    template = rng.choice(
        [
            "paper_title_exists_positive",
            "paper_title_year_positive",
            "paper_doi_exists_positive",
            "paper_has_pdf_positive",
        ]
    )

    if template == "paper_title_exists_positive":
        p = rng.choice(papers)
        question = f"Is there a BiB indexed paper titled '{p.title}'?"
        reason = "paper title exists in index"
        metadata = {"title": p.title}
        return question, reason, template, metadata, f"paper::{_normalize_text(p.title)}"

    if template == "paper_title_year_positive":
        candidates = [p for p in papers if p.year]
        p = rng.choice(candidates if candidates else papers)
        year = p.year if p.year else "unknown"
        question = f"Was the BiB paper '{p.title}' published in {year}?"
        reason = "paper title-year pair exists in index metadata"
        metadata = {"title": p.title, "year": year}
        return question, reason, template, metadata, f"paper::{_normalize_text(p.title)}"

    if template == "paper_doi_exists_positive":
        candidates = [p for p in papers if p.doi]
        p = rng.choice(candidates if candidates else papers)
        doi = p.doi if p.doi else ""
        question = f"Is DOI '{doi}' associated with a BiB indexed paper?" if doi else f"Does the paper '{p.title}' have an indexed DOI record?"
        reason = "paper DOI exists in index metadata" if doi else "paper metadata has DOI field absent; still index-grounded paper"
        metadata = {"title": p.title, "doi": doi}
        return question, reason, template, metadata, f"paper::{_normalize_text(p.title)}"

    candidates = [p for p in papers if p.has_pdf_chunk]
    p = rng.choice(candidates if candidates else papers)
    question = f"Does the indexed paper '{p.title}' have full-text PDF chunks in the BiB index?"
    reason = "paper has pdf_fulltext chunks in index" if p.has_pdf_chunk else "fallback paper selected"
    metadata = {"title": p.title, "has_pdf_chunk": p.has_pdf_chunk}
    return question, reason, "paper_has_pdf_positive", metadata, f"paper::{_normalize_text(p.title)}"


def _generate_negative(
    rng: random.Random,
    papers: list[PaperRecord],
    title_set: set[str],
    title_year_set: set[tuple[str, str]],
    doi_set: set[str],
    pdf_title_set: set[str],
) -> tuple[str, str, str, dict[str, Any], str]:
    template = rng.choice(
        [
            "paper_title_exists_negative",
            "paper_title_year_negative",
            "paper_doi_exists_negative",
            "paper_has_pdf_negative",
            "paper_out_of_scope_negative",
        ]
    )

    if template == "paper_title_exists_negative":
        base = rng.choice(papers)
        mutated = _mutate_title(base.title, rng)
        retries = 0
        while _normalize_text(mutated) in title_set and retries < 20:
            mutated = _mutate_title(base.title, rng)
            retries += 1
        question = f"Is there a BiB indexed paper titled '{mutated}'?"
        reason = "paper title not in index snapshot"
        metadata = {"title": mutated, "derived_from": base.title}
        return question, reason, template, metadata, f"paper::{_normalize_text(base.title)}"

    if template == "paper_title_year_negative":
        base = rng.choice(papers)
        wrong_year = str(rng.randint(1995, 2029))
        retries = 0
        while (_normalize_text(base.title), wrong_year) in title_year_set and retries < 20:
            wrong_year = str(rng.randint(1995, 2029))
            retries += 1
        question = f"Was the BiB paper '{base.title}' published in {wrong_year}?"
        reason = "title-year pair not present in index metadata"
        metadata = {"title": base.title, "year": wrong_year, "true_year": base.year}
        return question, reason, template, metadata, f"paper::{_normalize_text(base.title)}"

    if template == "paper_doi_exists_negative":
        base = rng.choice(papers)
        mutated_doi = _mutate_doi(base.doi, rng)
        retries = 0
        while mutated_doi in doi_set and retries < 20:
            mutated_doi = _mutate_doi(base.doi, rng)
            retries += 1
        question = f"Is DOI '{mutated_doi}' associated with a BiB indexed paper?"
        reason = "doi not in index metadata"
        metadata = {"doi": mutated_doi, "derived_from": base.doi, "title": base.title}
        return question, reason, template, metadata, f"paper::{_normalize_text(base.title)}"

    if template == "paper_has_pdf_negative":
        candidates = [p for p in papers if not p.has_pdf_chunk]
        if not candidates:
            candidates = papers
        p = rng.choice(candidates)
        question = f"Does the indexed paper '{p.title}' have full-text PDF chunks in the BiB index?"
        reason = "paper lacks pdf_fulltext chunks in index metadata"
        metadata = {"title": p.title, "has_pdf_chunk": p.has_pdf_chunk}
        return question, reason, template, metadata, f"paper::{_normalize_text(p.title)}"

    out_questions = [
        "Is there a BiB research paper titled 'Genome-wide atlas of Martian microbiome shifts'?",
        "Does the BiB data dictionary index the Lancet paper 'Global burden of disease in penguins 2030'?",
        "Is DOI '10.5555/nonbib.424242' linked to a Born in Bradford research paper?",
        "Is there full-text PDF content for the paper 'NHANES Nutrition Trial in Icelandic Teens' in the BiB data dictionary?",
        "Does the BiB data dictionary include a 2027 Nature paper on quantum cardiology?",
    ]
    question = rng.choice(out_questions)
    reason = "question is out of scope for the BiB data dictionary paper index"
    metadata = {"domain": "non_bib_papers"}
    return question, reason, "paper_out_of_scope_negative", metadata, f"oos::{_normalize_text(question)}"


def _validate_record(
    record: dict[str, Any],
    title_set: set[str],
    title_year_set: set[tuple[str, str]],
    doi_set: set[str],
    pdf_title_set: set[str],
) -> bool:
    generation_type = record["generation_type"]
    should_abstain = bool(record["should_abstain"])
    metadata = record.get("metadata", {})

    if generation_type in {"paper_title_exists_positive", "paper_title_exists_negative"}:
        title = _normalize_text(str(metadata.get("title", "")))
        exists = title in title_set
        return exists != should_abstain

    if generation_type == "paper_title_year_positive":
        title = _normalize_text(str(metadata.get("title", "")))
        year = str(metadata.get("year", "")).strip()
        return (title, year) in title_year_set and not should_abstain

    if generation_type == "paper_title_year_negative":
        # Paper exists but wrong year claimed → model can answer "no" → should_abstain=False
        title = _normalize_text(str(metadata.get("title", "")))
        year = str(metadata.get("year", "")).strip()
        title_exists = title in title_set
        year_matches = (title, year) in title_year_set
        return title_exists and not year_matches and not should_abstain

    if generation_type in {"paper_doi_exists_positive", "paper_doi_exists_negative"}:
        doi = str(metadata.get("doi", "")).strip()
        if not doi:
            return not should_abstain
        exists = doi in doi_set
        return exists != should_abstain

    if generation_type == "paper_has_pdf_positive":
        title = _normalize_text(str(metadata.get("title", "")))
        return title in pdf_title_set and not should_abstain

    if generation_type == "paper_has_pdf_negative":
        # Paper exists but has no PDF chunks → model can answer "no" → should_abstain=False
        title = _normalize_text(str(metadata.get("title", "")))
        title_exists = title in title_set
        has_pdf = title in pdf_title_set
        return title_exists and not has_pdf and not should_abstain

    if generation_type == "paper_out_of_scope_negative":
        return should_abstain

    return False


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _make_qa_sample(records: list[dict[str, Any]], fraction: float, rng: random.Random) -> list[dict[str, Any]]:
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
        rows = by_bucket[bucket]
        rng.shuffle(rows)
        take = max(1, round(n_target / max(1, len(buckets))))
        sample.extend(rows[:take])

    if len(sample) < n_target:
        remaining = [r for r in records if r not in sample]
        rng.shuffle(remaining)
        sample.extend(remaining[: n_target - len(sample)])

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in sample:
        qid = str(row.get("query_id", ""))
        if qid in seen:
            continue
        seen.add(qid)
        out.append(row)

    return out[:n_target]


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

    snapshot = _load_indexed_papers()
    papers: list[PaperRecord] = snapshot["papers"]

    title_set = {_normalize_text(p.title) for p in papers}
    title_year_set = {(_normalize_text(p.title), p.year) for p in papers if p.year}
    doi_set = {p.doi for p in papers if p.doi}
    pdf_title_set = {_normalize_text(p.title) for p in papers if p.has_pdf_chunk}

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
            question, reason, generation_type, metadata, group_key = _generate_positive(rng=rng, papers=papers)
            should_abstain = False
            label_key = "positive"
        else:
            question, reason, generation_type, metadata, group_key = _generate_negative(
                rng=rng,
                papers=papers,
                title_set=title_set,
                title_year_set=title_year_set,
                doi_set=doi_set,
                pdf_title_set=pdf_title_set,
            )
            # Policy split: mismatch types (wrong year, no PDF) have a definitive "no"
            # answer because the paper EXISTS. True abstentions are reserved for
            # non-existent titles/DOIs and out-of-scope queries.
            if generation_type in {"paper_title_year_negative", "paper_has_pdf_negative"}:
                should_abstain = False
            else:
                should_abstain = True
            label_key = "negative"

        normalized_question = _normalize_text(question)
        pair_key = (normalized_question, should_abstain)
        if normalized_question in seen_questions or pair_key in seen_pairs:
            continue

        split = _split_from_group(group_key, args.train_ratio, args.dev_ratio)

        record = _question_record(
            query_id=f"paper_abs_{len(records) + 1:05d}",
            question=question,
            should_abstain=should_abstain,
            reason=reason,
            generation_type=generation_type,
            group_key=group_key,
            split=split,
            dataset_version=args.dataset_version,
            source_scope=(
                "outside_index_snapshot" if generation_type == "paper_out_of_scope_negative" else "in_index_snapshot"
            ),
            metadata=metadata,
        )

        if not _validate_record(
            record=record,
            title_set=title_set,
            title_year_set=title_year_set,
            doi_set=doi_set,
            pdf_title_set=pdf_title_set,
        ):
            continue

        records.append(record)
        seen_questions.add(normalized_question)
        seen_pairs.add(pair_key)
        counters[label_key] += 1

    if len(records) < args.n_examples:
        raise RuntimeError(
            f"Generated only {len(records)} records after {attempts} attempts; requested {args.n_examples}."
        )

    _write_jsonl(args.output, records)
    qa_sample = _make_qa_sample(records, args.qa_sample_fraction, rng)
    _write_jsonl(args.qa_sample_output, qa_sample)

    report = {
        "dataset_version": args.dataset_version,
        "index_snapshot_filter_enabled": True,
        "index_snapshot_id": snapshot["snapshot_id"],
        "n_examples": len(records),
        "n_positive": sum(1 for r in records if not r["should_abstain"]),
        "n_negative": sum(1 for r in records if r["should_abstain"]),
        "attempts": attempts,
        "unique_questions": len({_normalize_text(r["question"]) for r in records}),
        "split_distribution": {
            "train": sum(1 for r in records if r["split"] == "train"),
            "dev": sum(1 for r in records if r["split"] == "dev"),
            "test": sum(1 for r in records if r["split"] == "test"),
        },
        "generation_type_distribution": dict(
            sorted(Counter(str(r["generation_type"]) for r in records).items(), key=lambda item: item[0])
        ),
        "qa_sample_size": len(qa_sample),
        "qa_sample_fraction": args.qa_sample_fraction,
        "index_snapshot_stats": {
            "n_unique_titles": snapshot["n_unique_titles"],
            "n_titles_with_pdf": snapshot["n_titles_with_pdf"],
            "bib_papers_collection_count": snapshot["collection_count"],
        },
        "sources": {
            "chroma_dir": snapshot["chroma_dir"],
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
    print(f"Index snapshot id:   {report['index_snapshot_id']}")
    print(
        "Index snapshot:      "
        f"titles={snapshot['n_unique_titles']}, "
        f"titles_with_pdf={snapshot['n_titles_with_pdf']}, "
        f"bib_papers_rows={snapshot['collection_count']}"
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
