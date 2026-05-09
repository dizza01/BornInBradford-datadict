"""Expand short gold answers in JSONL triples into concise full sentences.

Goal
----
Some triples contain very short gold answers (e.g., "26%", "Poisson regression").
Those can be poor reference answers for answer correctness / conciseness metrics.

This script rewrites only *short* answers into a one-sentence gold answer that
references the question subject, while keeping the factual content identical to
what is already present in the triple (answer/evidence/source_chunk).

Design principles
-----------------
- Conservative: do not add new numbers or claims.
- Minimal: keep answers short; do not paste the entire question.
- Deterministic: rule-based rewrites, no model calls.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _simple_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").strip().lower())


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _strip_trailing_punct(text: str) -> str:
    return (text or "").strip().rstrip(". ")


def _ensure_period(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    if t.endswith((".", "?", "!")):
        return t
    return t + "."


def _needs_expansion(answer: str) -> bool:
    ans = _normalize_space(answer)
    if not ans:
        return False

    toks = _simple_tokens(ans)
    # Keep longer answers as-is.
    if len(toks) >= 12:
        return False
    # Keep sentence-like answers as-is.
    if ans.endswith(".") and len(toks) >= 6:
        return False
    # Otherwise, treat as too-short / fragment.
    return True


def _question_to_sentence(question: str, value: str) -> str:
    q_raw = _normalize_space(question)
    q = q_raw.rstrip("?")
    v = _strip_trailing_punct(_normalize_space(value))

    if not q:
        return _ensure_period(v)

    # If the question starts with "In the paper ...", try to use the sub-question after the first comma.
    q_lower = q.lower()
    if q_lower.startswith("in the paper") and "," in q:
        tail = q.split(",", 1)[1].strip()
        if tail.lower().startswith(("what ", "which ", "how many ")):
            q = tail.rstrip("?")
            q_lower = q.lower()

    # Common templates.
    m = re.match(r"^What\s+(is|are|was|were)\s+(.+)$", q, flags=re.IGNORECASE)
    if m:
        verb = m.group(1).lower()
        subject = m.group(2).strip()
        subj_lower = subject.lower()
        # Avoid "The the ..." and "The a/an ...".
        if subj_lower.startswith("the "):
            subject = "The " + subject[4:]
        elif subj_lower.startswith("a "):
            subject = "A " + subject[2:]
        elif subj_lower.startswith("an "):
            subject = "An " + subject[3:]
        else:
            subject = "The " + subject
        return _ensure_period(f"{subject} {verb} {v}")

    m = re.match(r"^What\s+(.+?)\s+(was|were)\s+not\s+available\b(.*)$", q, flags=re.IGNORECASE)
    if m:
        subject = m.group(1).strip()
        verb = m.group(2).lower()
        rest = (m.group(3) or "").strip()
        if subject.lower().startswith("the "):
            subject = "The " + subject[4:]
        else:
            subject = "The " + subject
        rest_txt = (" " + rest) if rest else ""
        return _ensure_period(f"{subject} that was not available{rest_txt} {verb} {v}")

    m = re.match(r"^What\s+(.+?)\s+(was|were)\s+used\b.*$", q, flags=re.IGNORECASE)
    if m:
        subject = m.group(1).strip()
        verb = m.group(2).lower()
        if subject.lower().startswith("the "):
            subject = "The " + subject[4:]
        else:
            subject = "The " + subject
        return _ensure_period(f"{subject} {verb} {v}")

    if q_lower.startswith("how many "):
        # Prefer specific patterns to avoid duplicating the question tail.
        m = re.match(r"^How many\s+(.+?)\s+were\s+there\s+(.+)$", q, flags=re.IGNORECASE)
        if m:
            subject = m.group(1).strip()
            tail = m.group(2).strip()
            return _ensure_period(f"There were {v} {subject} {tail}")

        m = re.match(r"^How many\s+(.+?)\s+were\s+identified\s+(.+)$", q, flags=re.IGNORECASE)
        if m:
            subject = m.group(1).strip()
            tail = m.group(2).strip()
            return _ensure_period(f"There were {v} {subject} identified {tail}")

        subject = q[9:].strip().rstrip(".")
        return _ensure_period(f"There were {v} {subject}")

    if q.lower().startswith("what percentage"):
        return _ensure_period(f"The reported percentage was {v}")

    if q.lower().startswith("what proportion"):
        return _ensure_period(f"The reported proportion was {v}")

    if q_lower.startswith("which "):
        subject = q[6:].strip()

        # Prefer: "The <noun> that <verb phrase> was <value>." when possible.
        # This reads well for questions like:
        # - "Which socioeconomic indicator showed ...?"
        # - "Which European country had ...?"
        subj = subject
        if subj.lower().startswith("the "):
            subj = subj[4:].strip()

        split = re.split(r"\b(had|has|have|showed|shows|was|were|is|are)\b", subj, maxsplit=1, flags=re.IGNORECASE)
        if len(split) == 3:
            head = split[0].strip()
            verb = split[1].strip()
            tail = split[2].strip()
            if head and tail:
                return _ensure_period(f"The {head} that {verb} {tail} was {v}")

        return _ensure_period(f"The {subj} was {v}")

    # Last resort: short and non-committal framing.
    return _ensure_period(f"The reported value was {v}")


def _maybe_enrich_value(row: dict[str, Any]) -> str:
    """Prefer using the existing answer verbatim.

    Very conservative enrichment: if the answer is a single numeric token and the
    evidence contains a longer expression that includes that token (e.g. adds a
    CI right next to it), use that longer expression.

    This is optional and intentionally narrow to avoid introducing ambiguity.
    """

    answer = _normalize_space(str(row.get("answer", "")))
    evidence = _normalize_space(str(row.get("evidence", "")))
    if not answer or not evidence:
        return answer

    ans_toks = _simple_tokens(answer)
    if len(ans_toks) != 1:
        return answer

    # Only try when the answer looks numeric-ish (e.g., 26%, 239, -1.000)
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", answer):
        return answer

    if answer not in evidence:
        return answer

    # Capture the answer token plus any immediate parenthetical (e.g. CI) after it.
    # Example: "−1.000 (−1.798, −0.201)" or "0.74 (0.61 to 0.90)".
    # NOTE: This intentionally does not attempt to interpret what the parentheses mean.
    m = re.search(
        re.escape(answer) + r"\s*(\([^\)]{0,80}\))?",
        evidence,
    )
    if not m:
        return answer

    candidate = (answer + (" " + m.group(1) if m.group(1) else "")).strip()
    # Use the candidate only if it adds info beyond the raw answer.
    return candidate if len(candidate) > len(answer) else answer


def expand_file(input_path: Path, output_path: Path) -> dict[str, int]:
    changed = 0
    total = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1

            answer = str(row.get("answer", ""))
            if _needs_expansion(answer):
                value = _maybe_enrich_value(row) or answer
                row["answer"] = _question_to_sentence(str(row.get("question", "")), value)
                changed += 1
            else:
                # Normalise punctuation even for answers we keep unchanged.
                row["answer"] = _ensure_period(str(row.get("answer", "")))

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"total": total, "changed": changed}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("eval/evaluation_datasets/triples/pdf_retrieval_triples.jsonl"),
        help="Input JSONL triples file.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("eval/evaluation_datasets/triples/pdf_retrieval_triples_expanded.jsonl"),
        help="Output JSONL triples file with expanded gold answers.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    stats = expand_file(args.input, args.output)
    print(json.dumps({"input": str(args.input), "output": str(args.output), **stats}, indent=2))


if __name__ == "__main__":
    main()
