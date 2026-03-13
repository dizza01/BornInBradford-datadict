"""Generate retrieval-evaluation triples from existing PDF chunks in ChromaDB.

This script samples documents from the `bib_papers` collection where
`metadata.source == "pdf_fulltext"`, asks the configured HuggingFace model to
create a grounded `(question, answer)` pair from each sampled chunk, and writes
JSONL records containing the generated pair plus the original source chunk.

Usage examples
--------------

    ../../.venv/bin/python eval/generate_pdf_retrieval_triples.py

    ../../.venv/bin/python eval/generate_pdf_retrieval_triples.py \
        --sample-size 100 \
        --output eval/pdf_retrieval_triples.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bib_research_assistant import DEFAULT_MODEL, _get_hf_client, get_chroma_client

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "pdf_retrieval_triples.jsonl"
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
ALLOWED_QUESTION_TYPES = {"definition", "method", "result", "dataset", "theory"}

GENERATION_PROMPT = """You are building a retrieval evaluation set for a research assistant.

You will receive one source chunk from a PDF. Create exactly one grounded question-answer pair that is answerable from that chunk alone.

Rules:
- Use only information explicitly present in the chunk.
- Write a natural researcher-style question.
- The question should be specific enough that this chunk is a strong retrieval target.
- The answer should be short, factual, and fully supported by the chunk.
- Do not mention "chunk", "document", "excerpt", or "passage" in the question.
- Do not ask for information not present in the chunk.
- If the chunk is mostly references, citations, acknowledgements, page headers/footers, or otherwise unsuitable, return skip=true.
- Avoid copying long phrases directly from the text into the question. Paraphrase the question naturally.

Return valid JSON only, with one of these shapes:
{"skip": true, "reason": "..."}
OR
{
  "skip": false,
  "question": "...",
  "answer": "...",
    "evidence": "short verbatim span from the chunk supporting the answer",
    "question_type": "one of: definition | method | result | dataset | theory"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grounded PDF retrieval triples from ChromaDB PDF chunks.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Number of PDF chunks to sample and attempt to convert into triples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for chunk sampling.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name to use for generation (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2200,
        help="Maximum number of chunk characters sent to the generator model.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=2.0,
        help="Multiplier for candidate chunk pool (default: 2.0 => sample 2x target chunks before generation).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Maximum chunk attempts. Default 0 means all sampled candidates are tried.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Pause between generation calls to avoid bursty API traffic.",
    )
    return parser.parse_args()


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_generation_response(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = JSON_BLOCK_RE.search(cleaned)
        if not match:
            raise
        return json.loads(match.group(0))


def _load_pdf_chunks() -> list[dict[str, Any]]:
    client = get_chroma_client()
    collection = client.get_collection("bib_papers")
    results = collection.get(
        where={"source": "pdf_fulltext"},
        include=["documents", "metadatas"],
    )

    ids = results.get("ids", []) or []
    docs = results.get("documents", []) or []
    metas = results.get("metadatas", []) or []

    chunks: list[dict[str, Any]] = []
    for chunk_id, document, metadata in zip(ids, docs, metas):
        if not document:
            continue
        document = re.sub(r"\s+", " ", document).strip()
        if not document:
            continue
        chunks.append(
            {
                "id": chunk_id,
                "document": document,
                "metadata": metadata or {},
            }
        )
    return chunks


def _sample_chunks(chunks: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if not chunks:
        return []
    if sample_size >= len(chunks):
        return list(chunks)
    rng = random.Random(seed)
    return rng.sample(chunks, sample_size)


def _normalise_question_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text if text in ALLOWED_QUESTION_TYPES else "dataset"


def _generate_triple(
    llm_client: Any,
    model: str,
    chunk: dict[str, Any],
    max_chars: int,
) -> dict[str, Any] | None:
    metadata = chunk["metadata"]
    source_text = chunk["document"][:max_chars]

    user_prompt = (
        f"Chunk ID: {chunk['id']}\n"
        f"PDF file: {metadata.get('pdf_file', '')}\n"
        f"Title: {metadata.get('title', '')}\n"
        f"Year: {metadata.get('year', '')}\n"
        f"Chunk number: {metadata.get('chunk', '')}\n\n"
        f"Source chunk:\n{source_text}"
    )

    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GENERATION_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=350,
    )
    raw_text = response.choices[0].message.content or ""
    parsed = _parse_generation_response(raw_text)

    if parsed.get("skip"):
        return None

    question = str(parsed.get("question", "")).strip()
    answer = str(parsed.get("answer", "")).strip()
    evidence = str(parsed.get("evidence", "")).strip()
    question_type = _normalise_question_type(parsed.get("question_type", ""))
    if not question or not answer:
        return None
    if evidence and evidence not in chunk["document"]:
        return None

    return {
        "query_id": f"pdf_{chunk['id']}",
        "question": question,
        "answer": answer,
        "question_type": question_type,
        "source_chunk_id": chunk["id"],
        "source_chunk": chunk["document"],
        "chunk_index": metadata.get("chunk"),
        "chunk_length": len(chunk["document"]),
        "source_metadata": metadata,
        "evidence": evidence,
        "generation_model": model,
    }


def main() -> None:
    args = parse_args()

    if args.output.exists() and not args.overwrite:
        print(f"❌ Output already exists: {args.output}")
        print("   Re-run with --overwrite to replace it.")
        sys.exit(1)

    chunks = _load_pdf_chunks()
    if not chunks:
        print("❌ No PDF chunks found in collection 'bib_papers'.")
        print("   Build the index first with: python bib_research_assistant.py --build")
        sys.exit(1)

    candidate_size = int(max(args.sample_size, round(args.sample_size * args.candidate_multiplier)))
    sampled = _sample_chunks(chunks, candidate_size, args.seed)
    max_attempts = args.max_attempts or len(sampled)

    llm_client = _get_hf_client(args.model)
    if not llm_client:
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"📄 PDF chunks available: {len(chunks)}")
    print(f"🎯 Target triples: {args.sample_size}")
    print(f"🧪 Candidate chunks: {len(sampled)}")
    print(f"🤖 Generation model: {args.model}")
    print(f"💾 Writing to: {args.output}\n")

    generated: list[dict[str, Any]] = []
    attempts = 0

    for chunk in sampled:
        if len(generated) >= args.sample_size or attempts >= max_attempts:
            break
        attempts += 1

        try:
            triple = _generate_triple(
                llm_client=llm_client,
                model=args.model,
                chunk=chunk,
                max_chars=args.max_chars,
            )
        except Exception as exc:
            print(f"⚠️  Skipping {chunk['id']} after generation error: {exc}")
            time.sleep(args.sleep_seconds)
            continue

        if not triple:
            print(f"↷ Skipped unsuitable chunk: {chunk['id']}")
            time.sleep(args.sleep_seconds)
            continue

        generated.append(triple)
        print(
            f"✅ {len(generated):>3}/{args.sample_size} | "
            f"{triple['source_chunk_id']} | {triple['question'][:90]}"
        )
        time.sleep(args.sleep_seconds)

    with args.output.open("w", encoding="utf-8") as fh:
        for record in generated:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n════════════════════════════════════════════════════")
    print(f"Generated triples: {len(generated)}")
    print(f"Attempts:          {attempts}")
    print(f"Output:            {args.output}")
    if generated:
        qtype_counts = Counter(g["question_type"] for g in generated)
        qtype_summary = ", ".join(
            f"{qtype}={count} ({count / len(generated):.1%})"
            for qtype, count in sorted(qtype_counts.items())
        )
        print(f"Question types:    {qtype_summary}")
    if len(generated) < args.sample_size:
        print("⚠️  Fewer triples than requested were produced; some sampled chunks were unsuitable.")
    else:
        print("✅ Generation complete")


if __name__ == "__main__":
    main()
