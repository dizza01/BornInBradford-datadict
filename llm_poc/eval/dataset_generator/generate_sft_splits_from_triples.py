"""Create generic SFT-ready train/dev/test splits from retrieval triples.

This script:
1. Reads triples from a JSONL file.
2. Filters low-quality rows.
3. Splits at document level with an exact 30% train target by row count.
4. Writes train/dev/test JSONL files ready for chat SFT.

Usage:

    ../../.venv/bin/python eval/dataset_generator/generate_sft_splits_from_triples.py

    ../../.venv/bin/python eval/dataset_generator/generate_sft_splits_from_triples.py \
        --input eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl \
        --output-dir eval/evaluation_datasets/triples/train_dev_val \
        --train-ratio 0.30
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "dataset_generator" else SCRIPT_DIR
DATASETS_DIR = EVAL_DIR / "evaluation_datasets"

DEFAULT_INPUT = DATASETS_DIR / "triples" / "pdf_retrieval_triples.jsonl"
DEFAULT_OUTPUT_DIR = DATASETS_DIR / "triples" / "train_dev_val"

SYSTEM_MSG = (
    "You are a QA assistant. Use only the provided context. "
    "If the answer is not present in the context, say so clearly."
)

BAD_ANSWER_PATTERNS = [
    r"\b(as an ai|language model)\b",
    r"\b(i cannot answer|cannot answer|can't answer)\b",
    r"\b(not enough information|insufficient information)\b",
    r"\bunknown\b",
]

QUESTION_BAD_PATTERNS = [
    r"\b(chunk|excerpt|passage|document)\b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create generic SFT train/dev/test splits from triples JSONL.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input triples JSONL path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for train/dev/test JSONL files.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.30,
        help="Target train ratio by row count (default: 0.30).",
    )
    parser.add_argument(
        "--dev-ratio-of-remainder",
        type=float,
        default=0.50,
        help="Fraction of non-train rows assigned to dev (default: 0.50).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--allow-nearest-train",
        action="store_true",
        help="Allow nearest train size if exact document-level target is impossible.",
    )
    parser.add_argument(
        "--min-context-chars",
        type=int,
        default=300,
        help="Minimum source context length to keep a row.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def _normalise_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _derive_doc_id(row: dict[str, Any]) -> str:
    meta = row.get("source_metadata") or {}
    pdf_file = _normalise_text(meta.get("pdf_file"))
    if pdf_file:
        return pdf_file

    title = _normalise_text(meta.get("title"))
    year = _normalise_text(meta.get("year"))
    if title:
        return f"{title}::{year}" if year else title

    chunk_id = _normalise_text(row.get("source_chunk_id"))
    if chunk_id:
        return re.sub(r"_chunk_\d+$", "", chunk_id)

    return _normalise_text(row.get("query_id")) or "unknown_doc"


def _is_low_quality(row: dict[str, Any], min_context_chars: int) -> tuple[bool, str]:
    question = _normalise_text(row.get("question"))
    answer = _normalise_text(row.get("answer"))
    context = _normalise_text(row.get("source_chunk"))
    evidence = _normalise_text(row.get("evidence"))

    if not question:
        return True, "missing_question"
    if not answer:
        return True, "missing_answer"
    if not context:
        return True, "missing_context"

    if len(context) < min_context_chars:
        return True, "short_context"
    if len(question) < 12:
        return True, "short_question"
    if len(answer) < 1:
        return True, "short_answer"
    if len(question) > 500:
        return True, "long_question"
    if len(answer) > 1200:
        return True, "long_answer"

    for pat in BAD_ANSWER_PATTERNS:
        if re.search(pat, answer, flags=re.IGNORECASE):
            return True, "bad_answer_pattern"

    for pat in QUESTION_BAD_PATTERNS:
        if re.search(pat, question, flags=re.IGNORECASE):
            return True, "question_not_natural"

    if evidence and evidence.lower() not in context.lower():
        return True, "evidence_not_in_context"

    return False, "ok"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            rows.append(row)
    return rows


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = (
            _normalise_text(row.get("source_chunk_id")).lower(),
            _normalise_text(row.get("question")).lower(),
            _normalise_text(row.get("answer")).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _subset_sum_docs(
    doc_to_count: dict[str, int],
    target_rows: int,
    rng: random.Random,
) -> tuple[set[str], bool]:
    items = list(doc_to_count.items())
    rng.shuffle(items)

    dp: dict[int, set[str]] = {0: set()}

    for doc_id, count in items:
        updates: dict[int, set[str]] = {}
        for total, selected_docs in dp.items():
            new_total = total + count
            if new_total not in dp and new_total not in updates:
                updates[new_total] = set(selected_docs)
                updates[new_total].add(doc_id)
        dp.update(updates)

    if target_rows in dp:
        return dp[target_rows], True

    closest_total = min(dp.keys(), key=lambda x: (abs(x - target_rows), x))
    return dp[closest_total], False


def _make_sft_record(row: dict[str, Any]) -> dict[str, Any]:
    question = _normalise_text(row.get("question"))
    answer = _normalise_text(row.get("answer"))
    context = _normalise_text(row.get("source_chunk"))

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": answer},
    ]

    text = (
        f"<|im_start|>system\n{SYSTEM_MSG}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>\n"
    )

    return {
        "messages": messages,
        "text": text,
        "question": question,
        "answer": answer,
        "question_type": row.get("question_type"),
        "source_chunk_id": row.get("source_chunk_id"),
        "source_metadata": row.get("source_metadata", {}),
        "evidence": row.get("evidence", ""),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input}")

    if not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio must be between 0 and 1.")

    if not (0 < args.dev_ratio_of_remainder < 1):
        raise ValueError("--dev-ratio-of-remainder must be between 0 and 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "sft_train.jsonl"
    dev_path = args.output_dir / "sft_dev.jsonl"
    test_path = args.output_dir / "sft_test.jsonl"
    stats_path = args.output_dir / "sft_split_stats.json"

    outputs = [train_path, dev_path, test_path, stats_path]
    if not args.overwrite:
        existing = [p for p in outputs if p.exists()]
        if existing:
            names = ", ".join(str(p) for p in existing)
            raise FileExistsError(
                f"Output files already exist: {names}. Use --overwrite to replace them."
            )

    raw_rows = _load_rows(args.input)
    deduped_rows = _deduplicate_rows(raw_rows)

    filtered_rows: list[dict[str, Any]] = []
    dropped_reasons: Counter[str] = Counter()

    for row in deduped_rows:
        low_quality, reason = _is_low_quality(row, min_context_chars=args.min_context_chars)
        if low_quality:
            dropped_reasons[reason] += 1
            continue
        row = dict(row)
        row["_doc_id"] = _derive_doc_id(row)
        filtered_rows.append(row)

    if not filtered_rows:
        raise RuntimeError("All rows were filtered out. Relax quality thresholds and retry.")

    doc_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered_rows:
        doc_to_rows[row["_doc_id"]].append(row)

    doc_to_count = {doc_id: len(rows) for doc_id, rows in doc_to_rows.items()}
    total_rows = len(filtered_rows)
    target_train_rows = round(total_rows * args.train_ratio)

    rng = random.Random(args.seed)

    train_docs, exact_train = _subset_sum_docs(doc_to_count, target_train_rows, rng)
    train_rows = [r for r in filtered_rows if r["_doc_id"] in train_docs]

    if not exact_train and not args.allow_nearest_train:
        actual = len(train_rows)
        raise RuntimeError(
            "Could not hit exact train target with document-level split. "
            f"Target rows={target_train_rows}, nearest possible={actual}. "
            "Re-run with --allow-nearest-train to accept nearest feasible split."
        )

    remainder_rows = [r for r in filtered_rows if r["_doc_id"] not in train_docs]
    remainder_doc_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in remainder_rows:
        remainder_doc_to_rows[row["_doc_id"]].append(row)

    remainder_doc_to_count = {
        doc_id: len(rows) for doc_id, rows in remainder_doc_to_rows.items()
    }
    target_dev_rows = round(len(remainder_rows) * args.dev_ratio_of_remainder)

    dev_docs, _ = _subset_sum_docs(remainder_doc_to_count, target_dev_rows, rng)
    dev_rows = [r for r in remainder_rows if r["_doc_id"] in dev_docs]
    test_rows = [r for r in remainder_rows if r["_doc_id"] not in dev_docs]

    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    rng.shuffle(test_rows)

    train_records = [_make_sft_record(r) for r in train_rows]
    dev_records = [_make_sft_record(r) for r in dev_rows]
    test_records = [_make_sft_record(r) for r in test_rows]

    _write_jsonl(train_path, train_records)
    _write_jsonl(dev_path, dev_records)
    _write_jsonl(test_path, test_records)

    stats = {
        "input_path": str(args.input),
        "output_dir": str(args.output_dir),
        "raw_rows": len(raw_rows),
        "deduped_rows": len(deduped_rows),
        "filtered_rows": total_rows,
        "dropped_reasons": dict(dropped_reasons),
        "documents_after_filter": len(doc_to_rows),
        "train_ratio_requested": args.train_ratio,
        "train_target_rows": target_train_rows,
        "train_rows": len(train_records),
        "train_docs": len(train_docs),
        "exact_train_target_achieved": exact_train,
        "dev_rows": len(dev_records),
        "dev_docs": len(dev_docs),
        "test_rows": len(test_records),
        "test_docs": len(set(r["_doc_id"] for r in test_rows)),
        "seed": args.seed,
    }

    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print("Generated generic SFT splits")
    print(f"  Train: {train_path} ({len(train_records)} rows)")
    print(f"  Dev:   {dev_path} ({len(dev_records)} rows)")
    print(f"  Test:  {test_path} ({len(test_records)} rows)")
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()
