"""Make PDF retrieval triple questions self-contained.

This script rewrites ambiguous questions like "...in the study" / "according to the study"
by injecting the paper title/year that already exists in each triple's `source_metadata`.

It is intentionally conservative:
- Only rewrites when there is a clear ambiguous reference to "the study/paper".
- Does not add new fields or change any other keys.
- Keeps JSONL format (one JSON object per line).

Usage
-----

    ../../.venv/bin/python eval/evaluation_datasets/triples/make_questions_self_contained.py \
        eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl \
        eval/evaluation_datasets/triples/train_dev_val/sft_train.jsonl \
        eval/evaluation_datasets/triples/train_dev_val/sft_dev.jsonl \
        eval/evaluation_datasets/triples/train_dev_val/sft_test.jsonl

"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_AMBIGUOUS_RE = re.compile(
    r"(?i)(\baccording to\b.*\b(the|this)\s+(study|paper)\b|\b(in|within|from)\s+the\s+(study|paper)\b|\bthis\s+study\b)"
)


def _normalize_question_text(text: str) -> str:
    """Conservative whitespace/punctuation cleanup.

    Only removes trivial formatting artifacts (extra spaces, space before
    punctuation, dangling commas before a question mark).
    """

    s = (text or "").strip()
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\s+([\?\!\:\;\.\,])", r"\1", s)
    s = re.sub(r",\?", "?", s)
    # In case we end up with "..., ?".
    s = re.sub(r"\,\s*\?", "?", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s


def _extract_title_year(rec: dict[str, Any]) -> tuple[str, str]:
    meta = rec.get("source_metadata") or {}
    title = (meta.get("title") or "").strip()
    year = (meta.get("year") or "").strip()

    if title and year:
        return title, year

    # Fallback: parse from source_chunk, which usually begins with
    # "Title: ... Year: ...".
    chunk = (rec.get("source_chunk") or "").strip()
    if chunk:
        m_title = re.search(r"(?i)\btitle:\s*(.*?)\s+year:\s*([0-9]{4})\b", chunk)
        if m_title:
            title = title or m_title.group(1).strip()
            year = year or m_title.group(2).strip()

    return title, year


def _question_already_self_contained(question: str, title: str, year: str) -> bool:
    q = (question or "").lower()
    if year and year in q:
        return True
    if title:
        # Avoid doing expensive fuzzy matching. If the question already contains
        # a chunk of the title (first 6 words), treat it as self-contained.
        title_words = [w for w in re.split(r"\s+", title.lower()) if w]
        prefix = " ".join(title_words[:6]).strip()
        if prefix and prefix in q:
            return True
    return False


def _de_ambiguous_question(question: str) -> str:
    # Remove common suffixes like "in the study".
    q = (question or "").strip()

    q = re.sub(r"(?i)\s+(in|within|from)\s+the\s+(study|paper)\s*\?\s*$", "?", q)
    q = re.sub(r"(?i)\s+(in|within|from)\s+the\s+(study|paper)\s*$", "", q)

    # If question contains "according to the study" internally, keep wording
    # but remove the ambiguous referent.
    q = re.sub(r"(?i)\baccording to\s+(the|this)\s+(study|paper)\b", "", q)
    q = re.sub(r"\s{2,}", " ", q).strip()

    # Ensure it still ends with '?'
    if q and not q.endswith("?"):
        q = q + "?"
    return _normalize_question_text(q)


def rewrite_question(rec: dict[str, Any]) -> tuple[bool, str]:
    question = (rec.get("question") or "").strip()
    if not question:
        return False, question

    # Always normalize punctuation (safe / idempotent).
    normalized = _normalize_question_text(question)
    normalization_changed = normalized != question
    question = normalized

    if not _AMBIGUOUS_RE.search(question):
        return normalization_changed, question

    title, year = _extract_title_year(rec)
    if not title and not year:
        return normalization_changed, question

    if _question_already_self_contained(question, title, year):
        return normalization_changed, question

    cleaned = _de_ambiguous_question(question)

    # Prefix keeps it unambiguous even if cleaned question is short.
    if title and year:
        prefix = f'In the paper "{title}" ({year}), '
    elif title:
        prefix = f'In the paper "{title}", '
    else:
        prefix = f"In the {year} paper, "

    new_q = _normalize_question_text((prefix + cleaned).strip())
    return True, new_q


def process_file(path: Path) -> dict[str, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []

    changed = 0
    processed = 0

    for line in lines:
        if not line.strip():
            continue
        processed += 1
        rec = json.loads(line)
        did_change, new_q = rewrite_question(rec)
        if did_change:
            rec["question"] = new_q
            changed += 1
        out_lines.append(json.dumps(rec, ensure_ascii=False))

    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return {"processed": processed, "changed": changed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="JSONL triple files to rewrite in-place")
    args = ap.parse_args()

    total_processed = 0
    total_changed = 0

    for p in args.paths:
        path = Path(p)
        stats = process_file(path)
        print(f"{path}: processed={stats['processed']} changed={stats['changed']}")
        total_processed += stats["processed"]
        total_changed += stats["changed"]

    print(f"TOTAL: processed={total_processed} changed={total_changed}")


if __name__ == "__main__":
    main()
