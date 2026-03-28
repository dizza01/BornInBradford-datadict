"""Run abstention evaluation on variable and paper abstention benchmarks.

This script evaluates answer-vs-abstain behavior on the generated abstention
benchmarks and reports:
- Abstention accuracy (overall)
- True abstain rate (recall on unanswerable)
- False answer rate (answered when should abstain)
- False abstain rate (abstained when answerable)
- Precision / Recall / F1 for the abstain class
- Per-slice metrics (variable_abstention, paper_abstention)
- Per-source-scope metrics (in_index_snapshot vs outside_index_snapshot)
- Per-reason-type metrics
- Adversarial/no-answer metrics (out-of-scope subset)

Usage
-----

    ../../.venv/bin/python eval/run_abstention_eval.py \
        --run-name abstention_baseline \
        --model Qwen/Qwen2.5-72B-Instruct

    ../../.venv/bin/python eval/run_abstention_eval.py \
        --prediction-mode model_strict \
        --max-queries-per-slice 100 \
        --run-name abstention_strict_smoke

        
    ../../.venv/bin/python eval/run_abstention_eval.py \
        --prediction-mode model_strict \
        --model dizza01/qwen2.5-7b-finetunerag-merged-4bit \
        --model-api-mode hf_endpoint \
        --model-endpoint-url https://skqrt4ar5z72zlb7.us-east-1.aws.endpoints.huggingface.cloud \
        --run-name abstention_qwen_finetuned_endpoint 
    
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bib_research_assistant import DEFAULT_MODEL, _get_hf_client, get_chroma_client, retrieve_context

DEFAULT_VARIABLE_DATASET = Path(__file__).resolve().parent / "evaluation_datasets" / "variable_abstention" / "abstention_benchmark.jsonl"
DEFAULT_PAPER_DATASET = Path(__file__).resolve().parent / "evaluation_datasets" / "paper_abstention" / "paper_abstention_benchmark.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "abstention_metrics" / "abstention_eval_results.json"
DEFAULT_COMPARISON_CSV = Path(__file__).resolve().parent / "results" / "abstention_metrics" / "comparison.csv"

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

STRICT_CLASSIFIER_SYSTEM_PROMPT = """You are a strict abstention classifier for a RAG assistant.

Decide whether the assistant SHOULD abstain for the given user question using ONLY the provided retrieved context.

Rules:
- Return should_abstain=true when the context is insufficient, missing the requested entity/fact, or clearly out-of-scope.
- Return should_abstain=false when the context contains enough evidence to provide a factual answer.
- Be conservative: if uncertain, abstain.

Return valid JSON only in this exact schema:
{
    "should_abstain": true or false,
    "reason": "one short reason",
    "confidence": 0.0 to 1.0
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate abstention performance and write summary JSON + comparison CSV.",
    )
    parser.add_argument("--variable-dataset", type=Path, default=DEFAULT_VARIABLE_DATASET)
    parser.add_argument("--paper-dataset", type=Path, default=DEFAULT_PAPER_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON_CSV)
    parser.add_argument("--run-name", type=str, default="abstention_eval")
    parser.add_argument(
        "--prediction-mode",
        type=str,
        choices=["model_strict", "oracle", "always_abstain", "always_answer"],
        default="model_strict",
        help="How predicted abstain labels are produced.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-api-mode",
        type=str,
        choices=["hf_api", "hf_endpoint"],
        default="hf_api",
        help=(
            "How to call classifier model in model_strict mode. "
            "hf_api uses Hugging Face Inference API with --model, "
            "hf_endpoint uses dedicated endpoint URL."
        ),
    )
    parser.add_argument(
        "--model-endpoint-url",
        type=str,
        default="",
        help="Dedicated Hugging Face Inference Endpoint URL for classifier model.",
    )
    parser.add_argument(
        "--model-endpoint-model",
        type=str,
        default="",
        help="Optional model name sent to endpoint chat API. If empty, --model is used.",
    )
    parser.add_argument("--max-queries-per-slice", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--classifier-max-tokens", type=int, default=120)
    parser.add_argument("--classifier-temperature", type=float, default=0.0)
    parser.add_argument("--classifier-retries", type=int, default=2)
    parser.add_argument("--classifier-retry-delay", type=float, default=0.5)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
            rows.append(record)
    return rows


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


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = JSON_BLOCK_RE.search(cleaned)
        if not match:
            raise
        return json.loads(match.group(0))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Could not coerce to bool: {value!r}")


def _predict_abstain_strict(
    *,
    question: str,
    context: str,
    llm_client: Any,
    model: str,
    max_tokens: int,
    temperature: float,
    retries: int,
    retry_delay: float,
) -> tuple[bool, str, float | None]:
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Classify whether the assistant should abstain."
    )

    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": STRICT_CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_text = response.choices[0].message.content or ""
            parsed = _parse_json_response(raw_text)
            should_abstain = _coerce_bool(parsed.get("should_abstain"))
            reason = str(parsed.get("reason", "")).strip()

            confidence_value = parsed.get("confidence")
            confidence: float | None
            if confidence_value is None:
                confidence = None
            else:
                confidence = float(confidence_value)
                if confidence < 0:
                    confidence = 0.0
                if confidence > 1:
                    confidence = 1.0

            return should_abstain, reason, confidence
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries and retry_delay > 0:
                time.sleep(retry_delay * (attempt + 1))

    fallback_pred = True
    fallback_reason = f"strict_classifier_parse_error_default_abstain: {last_error[:180]}"
    return fallback_pred, fallback_reason, None


def _get_hf_endpoint_client(base_url: str) -> Any:
    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for endpoint mode. Install with: pip install huggingface_hub"
        ) from exc

    token = os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_TOKEN", "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. Export HF_TOKEN='hf_...' before running endpoint mode."
        )

    if not base_url.strip():
        raise RuntimeError("--model-endpoint-url is required when --model-api-mode=hf_endpoint")

    client = InferenceClient(model=base_url.strip(), token=token)

    class _Wrapper:
        def __init__(self, hf_client):
            self._hf = hf_client

        def generate(self, prompt, temperature=0.0, max_tokens=1500, **kw):
            return self._hf.text_generation(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **kw
            )

    return _Wrapper(client)


def _reason_type(record: dict[str, Any]) -> str:
    generation_type = str(record.get("generation_type", "")).lower()
    reason = str(record.get("reason", "")).lower()

    if "out_of_scope" in generation_type or "out of scope" in reason:
        return "out_of_scope"
    if (
        "not in" in reason
        or "not present" in reason
        or "not exist" in reason
        or "lacks" in reason
        or "missing" in reason
    ):
        return "non_existent_or_missing"
    if "pair" in generation_type or "mapping" in reason or "different" in reason:
        return "mapping_or_pair_mismatch"
    if "year" in generation_type or "doi" in generation_type:
        return "metadata_mismatch"
    return "other"


def _compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        last_error = ""
        # If llm_client has 'generate', use endpoint mode (text_generation); else, use chat.completions.create
        if hasattr(llm_client, "generate"):
            prompt = f"{STRICT_CLASSIFIER_SYSTEM_PROMPT}\n\n{user_prompt}"
            max_attempts = max(1, retries + 1)
            base_delay = 1.0
            for attempt in range(1, max_attempts + 1):
                try:
                    response = llm_client.generate(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    raw_text = response
                    parsed = _parse_json_response(raw_text)
                    should_abstain = _coerce_bool(parsed.get("should_abstain"))
                    reason = str(parsed.get("reason", "")).strip()
                    confidence_value = parsed.get("confidence")
                    confidence: float | None
                    if confidence_value is None:
                        confidence = None
                    else:
                        confidence = float(confidence_value)
                        if confidence < 0:
                            confidence = 0.0
                        if confidence > 1:
                            confidence = 1.0
                    return should_abstain, reason, confidence
                except Exception as exc:
                    last_error = str(exc)
                    msg = str(exc)
                    if ("503" in msg or "Service Unavailable" in msg) and attempt < max_attempts:
                        wait = base_delay * (2 ** (attempt - 1))
                        print(f"[WARN] 503 Service Unavailable, retrying in {wait:.1f}s (attempt {attempt}/{max_attempts})...")
                        time.sleep(wait)
                        continue
                    if attempt < max_attempts and retry_delay > 0:
                        time.sleep(retry_delay * attempt)
            fallback_pred = True
            fallback_reason = f"strict_classifier_parse_error_default_abstain: {last_error[:180]}"
            return fallback_pred, fallback_reason, None
        else:
            for attempt in range(retries + 1):
                try:
                    response = llm_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": STRICT_CLASSIFIER_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    raw_text = response.choices[0].message.content or ""
                    parsed = _parse_json_response(raw_text)
                    should_abstain = _coerce_bool(parsed.get("should_abstain"))
                    reason = str(parsed.get("reason", "")).strip()
                    confidence_value = parsed.get("confidence")
                    confidence: float | None
                    if confidence_value is None:
                        confidence = None
                    else:
                        confidence = float(confidence_value)
                        if confidence < 0:
                            confidence = 0.0
                        if confidence > 1:
                            confidence = 1.0
                    return should_abstain, reason, confidence
                except Exception as exc:
                    last_error = str(exc)
                    if attempt < retries and retry_delay > 0:
                        time.sleep(retry_delay * (attempt + 1))
            fallback_pred = True
            fallback_reason = f"strict_classifier_parse_error_default_abstain: {last_error[:180]}"
            return fallback_pred, fallback_reason, None
    recall = true_abstain_rate
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "n_examples": n,
        "n_should_abstain": n_should_abstain,
        "n_should_answer": n_should_answer,
        "n_pred_abstain": n_pred_abstain,
        "n_pred_answer": n_pred_answer,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "abstention_accuracy": accuracy,
        "true_abstain_rate": true_abstain_rate,
        "false_answer_rate": false_answer_rate,
        "false_abstain_rate": false_abstain_rate,
        "precision_abstain": precision,
        "recall_abstain": recall,
        "f1_abstain": f1,
    }


def _make_csv_row(
    *,
    timestamp: str,
    run_name: str,
    model: str,
    prediction_mode: str,
    dataset_version: str,
    segment_type: str,
    segment_value: str,
    metrics: dict[str, Any],
    output_file: Path,
) -> dict[str, Any]:
    row = {
        "timestamp": timestamp,
        "run_name": run_name,
        "model": model,
        "prediction_mode": prediction_mode,
        "dataset_version": dataset_version,
        "segment_type": segment_type,
        "segment_value": segment_value,
        "output_file": str(output_file),
    }
    row.update(metrics)
    return row


def _append_comparison_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "timestamp",
        "run_name",
        "model",
        "prediction_mode",
        "dataset_version",
        "segment_type",
        "segment_value",
        "n_examples",
        "n_should_abstain",
        "n_should_answer",
        "n_pred_abstain",
        "n_pred_answer",
        "tp",
        "tn",
        "fp",
        "fn",
        "abstention_accuracy",
        "true_abstain_rate",
        "false_answer_rate",
        "false_abstain_rate",
        "precision_abstain",
        "recall_abstain",
        "f1_abstain",
        "output_file",
    ]

    existing_rows: list[dict[str, Any]] = []
    rewrite_header = False
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            existing_header = reader.fieldnames or []
            rewrite_header = existing_header != fieldnames
            for row in reader:
                existing_rows.append(row)

    keys_to_replace = {
        (str(r["run_name"]), str(r["segment_type"]), str(r["segment_value"]))
        for r in rows
    }
    filtered_existing = [
        r
        for r in existing_rows
        if (str(r.get("run_name", "")), str(r.get("segment_type", "")), str(r.get("segment_value", "")))
        not in keys_to_replace
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in filtered_existing:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
        for row in rows:
            writer.writerow(row)

    if rewrite_header:
        print(f"ℹ️ Rewrote CSV header for updated schema: {csv_path}")


def main() -> None:
    args = parse_args()

    if args.max_queries_per_slice < 0:
        raise ValueError("--max-queries-per-slice must be >= 0")

    variable_rows = _load_jsonl(args.variable_dataset)
    paper_rows = _load_jsonl(args.paper_dataset)

    rng = random.Random(args.seed)
    if args.max_queries_per_slice > 0:
        if len(variable_rows) > args.max_queries_per_slice:
            variable_rows = rng.sample(variable_rows, args.max_queries_per_slice)
        if len(paper_rows) > args.max_queries_per_slice:
            paper_rows = rng.sample(paper_rows, args.max_queries_per_slice)

    for row in variable_rows:
        row["slice_name"] = "variable_abstention"
        row["reason_type"] = _reason_type(row)
    for row in paper_rows:
        row["slice_name"] = "paper_abstention"
        row["reason_type"] = _reason_type(row)

    all_rows = variable_rows + paper_rows

    client = None
    llm_client = None
    model_for_call = args.model
    if args.prediction_mode == "model_strict":
        client = get_chroma_client()
        if args.model_api_mode == "hf_endpoint":
            llm_client = _get_hf_endpoint_client(args.model_endpoint_url)
            model_for_call = args.model_endpoint_model.strip()
        else:
            llm_client = _get_hf_client(args.model)
        if not llm_client:
            raise RuntimeError("Could not initialize HuggingFace client for abstention evaluation")

    start_ts = datetime.now().isoformat(timespec="seconds")
    print(f"🧪 Running abstention eval | mode={args.prediction_mode} | n={len(all_rows)}")

    for idx, row in enumerate(all_rows, start=1):
        if args.prediction_mode == "oracle":
            pred = bool(row.get("should_abstain", False))
            answer_text = ""
        elif args.prediction_mode == "always_abstain":
            pred = True
            answer_text = ""
        elif args.prediction_mode == "always_answer":
            pred = False
            answer_text = ""
        else:
            question = str(row.get("question", ""))
            context = retrieve_context(question, client)
            pred, classifier_reason, classifier_confidence = _predict_abstain_strict(
                question=question,
                context=context,
                llm_client=llm_client,
                model=model_for_call,
                max_tokens=args.classifier_max_tokens,
                temperature=args.classifier_temperature,
                retries=args.classifier_retries,
                retry_delay=args.classifier_retry_delay,
            )
            answer_text = ""
            row["classifier_reason"] = classifier_reason
            row["classifier_confidence"] = classifier_confidence
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        row["pred_should_abstain"] = pred
        row["answer_text"] = answer_text

        if idx % 50 == 0 or idx == len(all_rows):
            print(f"  processed {idx}/{len(all_rows)}")

    dataset_version_set = sorted({str(r.get("dataset_version", "")) for r in all_rows})
    dataset_version = ",".join(v for v in dataset_version_set if v)

    overall_metrics = _compute_metrics(all_rows)
    per_slice_metrics = {
        "variable_abstention": _compute_metrics([r for r in all_rows if r["slice_name"] == "variable_abstention"]),
        "paper_abstention": _compute_metrics([r for r in all_rows if r["slice_name"] == "paper_abstention"]),
    }

    source_scope_values = sorted({str(r.get("source_scope", "")) for r in all_rows})
    per_source_scope_metrics = {
        value: _compute_metrics([r for r in all_rows if str(r.get("source_scope", "")) == value])
        for value in source_scope_values
        if value
    }

    reason_types = sorted({str(r.get("reason_type", "")) for r in all_rows})
    per_reason_type_metrics = {
        value: _compute_metrics([r for r in all_rows if str(r.get("reason_type", "")) == value])
        for value in reason_types
        if value
    }

    adversarial_rows = [
        r
        for r in all_rows
        if (
            str(r.get("source_scope", "")) == "outside_index_snapshot"
            or "out_of_scope" in str(r.get("generation_type", "")).lower()
            or "out of scope" in str(r.get("reason", "")).lower()
        )
    ]
    adversarial_metrics = _compute_metrics(adversarial_rows)

    wrong_examples = [
        {
            "query_id": r.get("query_id", ""),
            "slice_name": r.get("slice_name", ""),
            "question": r.get("question", ""),
            "should_abstain": bool(r.get("should_abstain", False)),
            "pred_should_abstain": bool(r.get("pred_should_abstain", False)),
            "reason": r.get("reason", ""),
            "reason_type": r.get("reason_type", ""),
            "source_scope": r.get("source_scope", ""),
            "answer_text": r.get("answer_text", ""),
        }
        for r in all_rows
        if bool(r.get("should_abstain", False)) != bool(r.get("pred_should_abstain", False))
    ]

    report = {
        "generated_at": start_ts,
        "run_name": args.run_name,
        "prediction_mode": args.prediction_mode,
        "model": args.model,
        "config": {
            "variable_dataset": str(args.variable_dataset),
            "paper_dataset": str(args.paper_dataset),
            "model_api_mode": args.model_api_mode,
            "model_endpoint_url": args.model_endpoint_url,
            "model_endpoint_model": model_for_call,
            "max_queries_per_slice": args.max_queries_per_slice,
            "seed": args.seed,
            "sleep_seconds": args.sleep_seconds,
            "classifier_max_tokens": args.classifier_max_tokens,
            "classifier_temperature": args.classifier_temperature,
            "classifier_retries": args.classifier_retries,
            "classifier_retry_delay": args.classifier_retry_delay,
        },
        "dataset_version": dataset_version,
        "n_examples_total": len(all_rows),
        "overall": overall_metrics,
        "per_slice": per_slice_metrics,
        "per_source_scope": per_source_scope_metrics,
        "per_reason_type": per_reason_type_metrics,
        "adversarial_no_answer": adversarial_metrics,
        "n_incorrect": len(wrong_examples),
        "incorrect_examples_preview": wrong_examples[:25],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_rows: list[dict[str, Any]] = []
    csv_rows.append(
        _make_csv_row(
            timestamp=start_ts,
            run_name=args.run_name,
            model=args.model,
            prediction_mode=args.prediction_mode,
            dataset_version=dataset_version,
            segment_type="overall",
            segment_value="all",
            metrics=overall_metrics,
            output_file=args.output,
        )
    )
    for segment_name, metrics in per_slice_metrics.items():
        csv_rows.append(
            _make_csv_row(
                timestamp=start_ts,
                run_name=args.run_name,
                model=args.model,
                prediction_mode=args.prediction_mode,
                dataset_version=dataset_version,
                segment_type="slice",
                segment_value=segment_name,
                metrics=metrics,
                output_file=args.output,
            )
        )
    for segment_name, metrics in per_source_scope_metrics.items():
        csv_rows.append(
            _make_csv_row(
                timestamp=start_ts,
                run_name=args.run_name,
                model=args.model,
                prediction_mode=args.prediction_mode,
                dataset_version=dataset_version,
                segment_type="source_scope",
                segment_value=segment_name,
                metrics=metrics,
                output_file=args.output,
            )
        )
    for segment_name, metrics in per_reason_type_metrics.items():
        csv_rows.append(
            _make_csv_row(
                timestamp=start_ts,
                run_name=args.run_name,
                model=args.model,
                prediction_mode=args.prediction_mode,
                dataset_version=dataset_version,
                segment_type="reason_type",
                segment_value=segment_name,
                metrics=metrics,
                output_file=args.output,
            )
        )

    csv_rows.append(
        _make_csv_row(
            timestamp=start_ts,
            run_name=args.run_name,
            model=args.model,
            prediction_mode=args.prediction_mode,
            dataset_version=dataset_version,
            segment_type="adversarial_no_answer",
            segment_value="out_of_scope_subset",
            metrics=adversarial_metrics,
            output_file=args.output,
        )
    )

    _append_comparison_csv(args.comparison_csv, csv_rows)

    print("\n════════════════════════════════════════════════════")
    print(f"Run name:             {args.run_name}")
    print(f"Prediction mode:      {args.prediction_mode}")
    print(f"Model:                {args.model}")
    print(f"Model API mode:       {args.model_api_mode}")
    if args.model_api_mode == "hf_endpoint":
        print(f"Endpoint URL:         {args.model_endpoint_url}")
        print(f"Endpoint model arg:   {model_for_call}")
    print(f"Total examples:       {len(all_rows)}")
    print(f"Abstention accuracy:  {overall_metrics['abstention_accuracy']:.4f}")
    print(f"True abstain rate:    {overall_metrics['true_abstain_rate']:.4f}")
    print(f"False answer rate:    {overall_metrics['false_answer_rate']:.4f}")
    print(f"False abstain rate:   {overall_metrics['false_abstain_rate']:.4f}")
    print(f"Abstain F1:           {overall_metrics['f1_abstain']:.4f}")
    print(f"JSON output:          {args.output}")
    print(f"Comparison CSV:       {args.comparison_csv}")


if __name__ == "__main__":
    main()
