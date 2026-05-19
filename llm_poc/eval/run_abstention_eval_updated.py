"""Run abstention evaluation on variable and paper abstention benchmarks.
Usage
-----

    ../../.venv/bin/python eval/run_abstention_eval_updated.py \
        --run-name abstention_baseline \
        --model Qwen/Qwen2.5-72B-Instruct

     ../../.venv/bin/python eval/run_abstention_eval_updated.py \
        --run-name abstention_qwen7B \
        --model Qwen/Qwen2.5-7B-Instruct

    
        
     ../../.venv/bin/python eval/run_abstention_eval_updated.py \
        --run-name abstention_Llama-3.1-70B-Instruct \
        --model meta-llama/Llama-3.1-70B-Instruct

    ../../.venv/bin/python eval/run_abstention_eval_updated.py \
        --prediction-mode model_strict \
        --max-queries-per-slice 100 \
        --run-name abstention_strict_smoke

        

     ../../.venv/bin/python eval/run_abstention_eval_updated.py \
        --prediction-mode model_strict \
        --model dizza01/medalpaca-13b \
        --model-api-mode hf_endpoint \
        --model-endpoint-url https://h46ed7c31gh0orh6.us-east-1.aws.endpoints.huggingface.cloud \
        --run-name abstention_medalpaca-13b

     ../../.venv/bin/python eval/run_abstention_eval_updated.py \
        --prediction-mode model_strict \
        --max-queries-per-slice 100 \
        --model dizza01/qwen2.5-7b-finetunerag-merged-4bit \
        --model-api-mode hf_endpoint \
        --model-endpoint-url https://skqrt4ar5z72zlb7.us-east-1.aws.endpoints.huggingface.cloud \
        --run-name abstention_qwen_finetuned_endpoint_quantised 

    ../../.venv/bin/python eval/run_abstention_eval_updated.py \
        --prediction-mode model_strict \
        --max-queries-per-slice 100 \
        --model dizza01/qwen2.5-7b-finetunerag-merged \
        --model-api-mode hf_endpoint \
        --model-endpoint-url https://woo97muev69lrrvd.us-east-1.aws.endpoints.huggingface.cloud \
        --run-name abstention_qwen_finetuned_endpoint_non_quantised

     ../../.venv/bin/python eval/run_abstention_eval_updated.py \
        --prediction-mode model_strict \
        --max-queries-per-slice 100 \
        --model dizza01/qwen2.5-7b-pdf-merged \
        --model-api-mode hf_endpoint \
        --model-endpoint-url https://eyicswzutfjqodxe.us-east-1.aws.endpoints.huggingface.cloud \
        --run-name abstention_qwen_pdf_finetuned_endpoint_non_quantised   

    ../../.venv/bin/python eval/run_abstention_eval_updated.py \
        --prediction-mode model_strict \
        --max-queries-per-slice 100 \
        --model dizza01/BioMistral-7B-DARE \
        --model-api-mode hf_endpoint \
        --model-endpoint-url https://ylquc2d9j0a43ghh.us-east-1.aws.endpoints.huggingface.cloud \
        --run-name abstention_BioMistral_endpoint_non_quantised   
 


# vLLM Engine (OpenAI-style HTTP API)     # Calls:    <endpoint>/v1/chat/completions if available, else <endpoint>/v1/completions
 ../../.venv/bin/python eval/run_abstention_eval_updated.py \
        --prediction-mode model_strict \
        --model-api-mode hf_vllm \
        --model-endpoint-url https://cfnnphkb5rmchz80.us-east-2.aws.endpoints.huggingface.cloud/v1 \
        --model-endpoint-model dizza01/Qwen2.5-14B-Instruct \
        --max-queries-per-slice 1 \
        --run-name abstention_dizza01_qwen25_14b_vllm
 
               
    ../../.venv/bin/python eval/run_abstention_eval_updated.py \
        --prediction-mode model_strict \
        --model dizza01/qwen2-5-14b-instruct \
        --max-queries-per-slice 10 \
        --model-api-mode hf_endpoint \
        --model-endpoint-url https://fsjtjeq9afi5ktrh.us-east-1.aws.endpoints.huggingface.cloud \
        --run-name abstention_qwen2-5-14b-instruct_chat

    
This script evaluates answer-vs-abstain behavior on generated abstention
benchmarks and reports aggregated and slice-level metrics.
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
from types import SimpleNamespace
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

STRICT_CLASSIFIER_SYSTEM_PROMPT = """You are a strict abstention classifier for a BiB (Born in Bradford) RAG assistant.

The assistant answers questions about the BiB data dictionary: an index of variable and \
table metadata from BiB study documentation, and published Born in Bradford research \
papers. The index is built from HTML/CSV documentation files and a papers metadata JSON — \
it does NOT contain live clinical records, NHS systems, or any data source outside those \
documentation files. All answerable questions are lookups against what has been indexed — \
the fact IS or IS NOT documented in the indexed data.

You are given the user question and retrieved context. The context may contain an \
"Exact Registry Matches" section with direct metadata lookups (highest authority), \
followed by semantic search results.

Decide whether the assistant SHOULD abstain, using ONLY the retrieved context.

━━━ Variable and Table Registry Rules ━━━
- ABSTAIN if the context shows "VARIABLE NOT IN REGISTRY" for the variable in the question.
- ABSTAIN if a variable–table pairing is asked about and the context does NOT confirm \
that exact variable appears in that exact table. If the variable is shown in a DIFFERENT \
table, this is a registry mismatch — ABSTAIN.
- ABSTAIN if a table–project assignment is asked about and the context does NOT confirm \
that exact table belongs to that exact project.
- ABSTAIN if a table name appears in the question but the context shows \
"TABLE NOT IN REGISTRY" for it.
- DO NOT ABSTAIN if the context confirms the variable, table, or mapping exists exactly \
as asked.

━━━ Paper Registry Rules ━━━
- ABSTAIN if the context shows "PAPER TITLE NOT IN REGISTRY" for the title in the question.
- ABSTAIN if a specific publication year is claimed for a paper and the context does NOT \
confirm that year for that title.
- ABSTAIN if the context shows "PAPER DOI NOT IN REGISTRY" for the DOI in the question.
- ABSTAIN if asked whether a paper has full-text PDF chunks and the context shows \
"Has full-text PDF chunks in index: no".
- DO NOT ABSTAIN if the context explicitly confirms the paper title, year, DOI, or \
PDF availability exactly as asked.

━━━ Out-of-Scope Rules ━━━
- ABSTAIN if the question references anything not documented in the BiB data dictionary \
(NHS records, NHANES, UK Biobank, non-BiB external data sources, fabricated entities).
- ABSTAIN if the retrieved context contains no relevant BiB documentation at all.

━━━ General Rules ━━━
- The "Exact Registry Matches" section is the most authoritative evidence — weight it \
above semantic search results.
- A "NOT IN REGISTRY" result is definitive evidence to abstain.
- Be conservative: if the context is ambiguous or incomplete, abstain.

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
        choices=["hf_api", "hf_endpoint", "hf_vllm"],
        default="hf_api",
        help=(
            "How to call classifier model in model_strict mode. "
            "hf_api uses Hugging Face Inference API with --model, "
            "hf_endpoint uses dedicated endpoint URL, "
            "hf_vllm uses an OpenAI-style vLLM HTTP endpoint URL."
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
    parser.add_argument("--classifier-max-tokens", type=int, default=512)
    parser.add_argument("--classifier-temperature", type=float, default=0.0)
    parser.add_argument("--classifier-retries", type=int, default=2)
    parser.add_argument("--classifier-retry-delay", type=float, default=0.5)
    parser.add_argument(
        "--retrieval-depth",
        type=int,
        default=5,
        help="Top-k depth passed to context retrieval (n_results).",
    )
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


def _sanitize_filename_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "run"


def _build_unique_output_path(base_path: Path, run_name: str, timestamp: str) -> Path:
    safe_run_name = _sanitize_filename_component(run_name)
    safe_timestamp = _sanitize_filename_component(timestamp.replace(":", "-"))
    return base_path.with_name(f"{base_path.stem}_{safe_run_name}_{safe_timestamp}{base_path.suffix}")


def _extract_abstain_from_natural_language(text: str) -> bool | None:
    """Last-resort heuristic for fine-tuned models that ignore JSON instructions.

    Scans free-text output for explicit abstain/answer signals.
    Returns True (abstain), False (answer), or None (cannot determine).
    """
    t = text.lower().strip()

    # Strong abstain signals
    abstain_phrases = [
        "should abstain",
        "cannot answer",
        "not enough context",
        "insufficient context",
        "cannot be answered",
        "not supported by",
        "not in the context",
        "not present in",
        "out of scope",
        "unable to answer",
        "no relevant",
        "i don't know",
        "i cannot",
        "context does not",
        "context doesn't",
    ]
    # Strong answer signals
    answer_phrases = [
        "should answer",
        "should not abstain",
        "can be answered",
        "context contains",
        "context supports",
        "context provides",
        "answer is",
        "the answer",
        "according to the context",
        "based on the context",
        "based on the retrieved",
    ]

    abstain_hits = sum(1 for p in abstain_phrases if p in t)
    answer_hits = sum(1 for p in answer_phrases if p in t)

    if abstain_hits > answer_hits:
        return True
    if answer_hits > abstain_hits:
        return False
    return None


# Reinforced system prompt for fine-tuned models that tend to generate prose.
_JSON_ENFORCEMENT_SUFFIX = """

CRITICAL INSTRUCTION: Your entire response MUST be valid JSON. Do NOT write any explanation,
prose, or markdown. Output ONLY the JSON object below and nothing else:
{"should_abstain": true or false, "reason": "one short reason", "confidence": 0.0 to 1.0}
"""


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
) -> tuple[bool, str, float | None, dict[str, Any]]:
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Classify whether the assistant should abstain.\n"
        "Reply with ONLY a JSON object. No prose."
    )

    # Reinforced system prompt with hard JSON constraint.
    system_prompt = STRICT_CLASSIFIER_SYSTEM_PROMPT + _JSON_ENFORCEMENT_SUFFIX

    last_error = ""
    raw_text = ""
    attempts_used = 0
    for attempt in range(retries + 1):
        attempts_used = attempt + 1
        try:
            # Supports OpenAI-style chat completions clients.
            if hasattr(llm_client, "chat"):
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw_text = response.choices[0].message.content or ""
            else:
                # Fallback for endpoint wrappers exposing text generation only.
                prompt = (
                    f"{system_prompt}\n\n"
                    f"{user_prompt}\n\n"
                    'Output JSON only: {"should_abstain": true or false, "reason": "...", "confidence": 0.0}'
                )
                raw = llm_client.generate(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw_text = str(raw)

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

            debug_info = {
                "decision_source": "json",
                "attempts_used": attempts_used,
                "raw_response_text": raw_text,
                "parse_error": "",
            }
            return should_abstain, reason, confidence, debug_info
        except Exception as exc:
            last_error = str(exc)
            if attempt < retries:
                if retry_delay > 0:
                    time.sleep(retry_delay * (attempt + 1))
                # On the last retry, nudge temperature to break out of
                # deterministic failure modes (e.g. repetition loops).
                if attempt == retries - 1:
                    temperature = max(temperature, 0.3)

    # JSON parse failed on all retries — try natural language heuristic before
    # falling back to abstain so fine-tuned models that answer in prose still
    # contribute a signal rather than always defaulting to abstain.
    nl_pred = _extract_abstain_from_natural_language(raw_text)
    if nl_pred is not None:
        debug_info = {
            "decision_source": "nl_heuristic_fallback",
            "attempts_used": attempts_used,
            "raw_response_text": raw_text,
            "parse_error": last_error,
        }
        return nl_pred, f"nl_heuristic_fallback: {raw_text[:120]}", None, debug_info

    fallback_pred = True
    fallback_reason = f"strict_classifier_parse_error_default_abstain: {last_error[:180]}"
    debug_info = {
        "decision_source": "default_abstain_parse_error",
        "attempts_used": attempts_used,
        "raw_response_text": raw_text,
        "parse_error": last_error,
    }
    return fallback_pred, fallback_reason, None, debug_info


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
                **kw,
            )

        # OpenAI-compatible shim for _predict_abstain_strict.
        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, model, messages, temperature=0.0, max_tokens=1500, **kw):
            # Use Qwen chat template tokens so fine-tuned models see their
            # training format and are more likely to follow JSON instructions.
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            prompt_parts.append("<|im_start|>assistant\n")
            prompt = "\n".join(prompt_parts)
            out = self.generate(prompt, temperature=temperature, max_tokens=max_tokens, **kw)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=str(out)))]
            )

    return _Wrapper(client)


def _get_vllm_endpoint_client(*, base_url: str, model_name: str) -> Any:
    """Create a minimal OpenAI-style client for a vLLM HTTP endpoint.

    Exposes the interface used by `_predict_abstain_strict`:
    - `llm_client.chat.completions.create(...)` returning `choices[0].message.content`.
    - Also provides `.generate(...)` (OpenAI `/v1/completions`) for completeness.

    Auth: Bearer token from HF_TOKEN / HUGGINGFACE_TOKEN.
    """

    token = os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_TOKEN", "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set. Export HF_TOKEN='hf_...' before running vLLM endpoint mode."
        )

    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "requests is required for --model-api-mode=hf_vllm. Install with: pip install requests"
        ) from exc

    if not base_url.strip():
        raise RuntimeError("--model-endpoint-url is required when --model-api-mode=hf_vllm")

    base_url = base_url.strip().rstrip("/")
    model_name = (model_name or "").strip()
    if not model_name:
        raise RuntimeError("model_name is required for vLLM endpoint client")

    # Accept either:
    # - https://host               (we will call https://host/v1/...)
    # - https://host/v1            (we will call https://host/v1/...)
    api_base_url = base_url
    if not api_base_url.endswith("/v1"):
        api_base_url = f"{api_base_url}/v1"

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
    )

    def _raise_for_http(resp, *, endpoint_desc: str) -> None:
        if resp.status_code >= 400:
            text = ""
            try:
                text = resp.text
            except Exception:
                text = "<unreadable response body>"
            raise RuntimeError(
                f"vLLM HTTP error calling {endpoint_desc}: {resp.status_code} {text.strip()[:2000]}"
            )

    def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{api_base_url}{path}"
        resp = session.post(url, json=payload, timeout=120)
        _raise_for_http(resp, endpoint_desc=url)
        try:
            return resp.json()
        except Exception as exc:
            raise RuntimeError(
                f"vLLM response was not valid JSON from {url}: {str(exc)} | body={resp.text[:2000]}"
            ) from exc

    def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "user") or "user")
            content = str(msg.get("content", "") or "")
            parts.append(f"{role.upper()}:\n{content}")
        parts.append("ASSISTANT:\n")
        return "\n\n".join(parts)

    class _ChatCompletions:
        def create(self, model: str, messages, temperature=0.0, max_tokens=1500, **kw):
            # Preferred: OpenAI-style chat completions.
            chat_payload = {
                "model": model_name,
                "messages": list(messages),
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
            }
            try:
                data = _post_json("/chat/completions", chat_payload)
                content = (((data.get("choices") or [{}])[0]).get("message") or {}).get(
                    "content", ""
                )
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=str(content)))],
                )
            except RuntimeError as exc:
                # Fallback only if the *endpoint* is absent.
                # Do NOT fall back for model-not-found (param=model) or other request errors.
                msg = str(exc)
                status_match = re.search(r":\s*(\d{3})\b", msg)
                status = int(status_match.group(1)) if status_match else None
                body_lower = msg.lower()
                model_missing = (
                    '"param":"model"' in body_lower
                    or "param\":\"model\"" in body_lower
                    or (("model `" in body_lower or "model \"" in body_lower) and "does not exist" in body_lower)
                    or "unknown model" in body_lower
                )
                endpoint_missing = bool(
                    status in {404, 405, 501}
                    and ("/chat/completions" in body_lower or "chat/completions" in body_lower)
                    and ("not found" in body_lower or "404" in body_lower or "405" in body_lower)
                )
                if model_missing or not endpoint_missing:
                    raise

            prompt = _messages_to_prompt(list(messages))
            completion_payload = {
                "model": model_name,
                "prompt": str(prompt),
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
            }
            data = _post_json("/completions", completion_payload)
            choice0 = (data.get("choices") or [{}])[0]
            text = str(choice0.get("text", ""))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=str(text)))],
            )

    class _Chat:
        def __init__(self):
            self.completions = _ChatCompletions()

    class _VLLMWrapper:
        def __init__(self):
            self.chat = _Chat()

        def generate(self, prompt, temperature=0.0, max_tokens=1500, **kw):
            payload = {
                "model": model_name,
                "prompt": str(prompt),
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
            }
            data = _post_json("/completions", payload)
            choice0 = (data.get("choices") or [{}])[0]
            return str(choice0.get("text", ""))

    return _VLLMWrapper()


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
        return {
            "n_examples": 0,
            "n_should_abstain": 0,
            "n_should_answer": 0,
            "n_pred_abstain": 0,
            "n_pred_answer": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "abstention_accuracy": 0.0,
            "true_abstain_rate": 0.0,
            "false_answer_rate": 0.0,
            "false_abstain_rate": 0.0,
            "precision_abstain": 0.0,
            "recall_abstain": 0.0,
            "f1_abstain": 0.0,
        }

    should_abstain = [bool(r.get("should_abstain", False)) for r in rows]
    pred_abstain = [bool(r.get("pred_should_abstain", False)) for r in rows]

    tp = sum(1 for y, p in zip(should_abstain, pred_abstain) if y and p)
    tn = sum(1 for y, p in zip(should_abstain, pred_abstain) if (not y) and (not p))
    fp = sum(1 for y, p in zip(should_abstain, pred_abstain) if (not y) and p)
    fn = sum(1 for y, p in zip(should_abstain, pred_abstain) if y and (not p))

    n_should_abstain = sum(should_abstain)
    n_should_answer = n - n_should_abstain
    n_pred_abstain = sum(pred_abstain)
    n_pred_answer = n - n_pred_abstain

    accuracy = (tp + tn) / n
    true_abstain_rate = tp / n_should_abstain if n_should_abstain else 0.0
    false_answer_rate = fn / n_should_abstain if n_should_abstain else 0.0
    false_abstain_rate = fp / n_should_answer if n_should_answer else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_should_abstain if n_should_abstain else 0.0
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


def _compute_diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decision_source_counts = {
        "json": 0,
        "nl_heuristic_fallback": 0,
        "default_abstain_parse_error": 0,
        "unknown": 0,
    }
    parse_error_examples: list[dict[str, Any]] = []

    for row in rows:
        source = str(row.get("classifier_decision_source", "") or "unknown")
        if source not in decision_source_counts:
            source = "unknown"
        decision_source_counts[source] += 1

        if source != "json":
            parse_error_examples.append(
                {
                    "query_id": row.get("query_id", ""),
                    "slice_name": row.get("slice_name", ""),
                    "question": row.get("question", ""),
                    "should_abstain": bool(row.get("should_abstain", False)),
                    "pred_should_abstain": bool(row.get("pred_should_abstain", False)),
                    "classifier_decision_source": source,
                    "classifier_reason": row.get("classifier_reason", ""),
                    "classifier_confidence": row.get("classifier_confidence"),
                    "classifier_attempts_used": row.get("classifier_attempts_used"),
                    "classifier_parse_error": row.get("classifier_parse_error", ""),
                    "classifier_raw_response_text": row.get("classifier_raw_response_text", ""),
                }
            )

    total = len(rows)
    parse_success_count = decision_source_counts["json"]

    return {
        "decision_source_counts": decision_source_counts,
        "parse_success_count": parse_success_count,
        "parse_failure_count": total - parse_success_count,
        "parse_success_rate": (parse_success_count / total) if total else 0.0,
        "parse_failure_examples_preview": parse_error_examples[:50],
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
        print(f"Rewrote CSV header for updated schema: {csv_path}")


def main() -> None:
    args = parse_args()

    if args.max_queries_per_slice < 0:
        raise ValueError("--max-queries-per-slice must be >= 0")
    if args.retrieval_depth <= 0:
        raise ValueError("--retrieval-depth must be > 0")

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
            model_for_call = args.model_endpoint_model.strip() or args.model
        elif args.model_api_mode == "hf_vllm":
            model_for_call = args.model_endpoint_model.strip() or args.model
            llm_client = _get_vllm_endpoint_client(
                base_url=args.model_endpoint_url,
                model_name=model_for_call,
            )
        else:
            llm_client = _get_hf_client(args.model)
        if not llm_client:
            raise RuntimeError("Could not initialize HuggingFace client for abstention evaluation")

    start_ts = datetime.now().isoformat(timespec="seconds")
    output_path = _build_unique_output_path(args.output, args.run_name, start_ts)
    print(f"Running abstention eval | mode={args.prediction_mode} | n={len(all_rows)}")

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
            context = retrieve_context(question, client, n_results=args.retrieval_depth)
            pred, classifier_reason, classifier_confidence, classifier_debug = _predict_abstain_strict(
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
            row["classifier_decision_source"] = classifier_debug.get("decision_source", "")
            row["classifier_attempts_used"] = classifier_debug.get("attempts_used")
            row["classifier_parse_error"] = classifier_debug.get("parse_error", "")
            row["classifier_raw_response_text"] = classifier_debug.get("raw_response_text", "")
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

        row["pred_should_abstain"] = pred
        row["answer_text"] = answer_text

        if idx % 50 == 0 or idx == len(all_rows):
            print(f"processed {idx}/{len(all_rows)}")

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

    generation_types = sorted({str(r.get("generation_type", "")) for r in all_rows})
    per_generation_type_metrics = {
        value: _compute_metrics([r for r in all_rows if str(r.get("generation_type", "")) == value])
        for value in generation_types
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
    diagnostic_summary = _compute_diagnostic_summary(all_rows)

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
            "classifier_reason": r.get("classifier_reason", ""),
            "classifier_confidence": r.get("classifier_confidence"),
            "classifier_decision_source": r.get("classifier_decision_source", ""),
            "classifier_attempts_used": r.get("classifier_attempts_used"),
            "classifier_parse_error": r.get("classifier_parse_error", ""),
            "classifier_raw_response_text": r.get("classifier_raw_response_text", ""),
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
            "retrieval_depth": args.retrieval_depth,
        },
        "dataset_version": dataset_version,
        "n_examples_total": len(all_rows),
        "overall": overall_metrics,
        "per_slice": per_slice_metrics,
        "per_source_scope": per_source_scope_metrics,
        "per_reason_type": per_reason_type_metrics,
        "per_generation_type": per_generation_type_metrics,
        "adversarial_no_answer": adversarial_metrics,
        "diagnostics": diagnostic_summary,
        "n_incorrect": len(wrong_examples),
        "incorrect_examples_preview": wrong_examples[:25],
        "all_predictions": [
            {
                "query_id": r.get("query_id", ""),
                "slice_name": r.get("slice_name", ""),
                "question": r.get("question", ""),
                "should_abstain": bool(r.get("should_abstain", False)),
                "pred_should_abstain": bool(r.get("pred_should_abstain", False)),
                "reason": r.get("reason", ""),
                "reason_type": r.get("reason_type", ""),
                "source_scope": r.get("source_scope", ""),
                "classifier_reason": r.get("classifier_reason", ""),
                "classifier_confidence": r.get("classifier_confidence"),
                "classifier_decision_source": r.get("classifier_decision_source", ""),
                "classifier_attempts_used": r.get("classifier_attempts_used"),
                "classifier_parse_error": r.get("classifier_parse_error", ""),
                "classifier_raw_response_text": r.get("classifier_raw_response_text", ""),
            }
            for r in all_rows
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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
            output_file=output_path,
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
                output_file=output_path,
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
                output_file=output_path,
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
                output_file=output_path,
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
            output_file=output_path,
        )
    )

    _append_comparison_csv(args.comparison_csv, csv_rows)

    print("\n============================================")
    print(f"Run name:             {args.run_name}")
    print(f"Prediction mode:      {args.prediction_mode}")
    print(f"Model:                {args.model}")
    print(f"Model API mode:       {args.model_api_mode}")
    print(f"Retrieval depth:      {args.retrieval_depth}")
    if args.model_api_mode in {"hf_endpoint", "hf_vllm"}:
        print(f"Endpoint URL:         {args.model_endpoint_url}")
        print(f"Endpoint model arg:   {model_for_call}")
    print(f"Total examples:       {len(all_rows)}")
    print(f"Abstention accuracy:  {overall_metrics['abstention_accuracy']:.4f}")
    print(f"True abstain rate:    {overall_metrics['true_abstain_rate']:.4f}")
    print(f"False answer rate:    {overall_metrics['false_answer_rate']:.4f}")
    print(f"False abstain rate:   {overall_metrics['false_abstain_rate']:.4f}")
    print(f"Abstain F1:           {overall_metrics['f1_abstain']:.4f}")
    print(f"Parse success rate:   {diagnostic_summary['parse_success_rate']:.4f}")
    print(f"JSON output:          {output_path}")
    print(f"Comparison CSV:       {args.comparison_csv}")


if __name__ == "__main__":
    main()
