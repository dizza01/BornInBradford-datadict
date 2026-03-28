"""Run abstention benchmark generators and write a unified summary report.

This script can orchestrate:
1) Variable/table abstention benchmark generation.
2) Paper-specific abstention benchmark generation.
3) Both together.

It then writes a combined report with per-benchmark metadata plus aggregated
counts, so both slices can be tracked together per run.

Usage examples
--------------

    ../../.venv/bin/python eval/dataset_generator/generate_combined_abstention_benchmarks.py \
        --target both \
        --overwrite

    ../../.venv/bin/python eval/dataset_generator/generate_combined_abstention_benchmarks.py \
        --target variables_tables \
        --n-examples-vt 500 \
        --overwrite

    ../../.venv/bin/python eval/dataset_generator/generate_combined_abstention_benchmarks.py \
        --target papers \
        --n-examples-paper 400 \
        --overwrite

    ../../.venv/bin/python eval/dataset_generator/generate_combined_abstention_benchmarks.py \
        --n-examples-vt 500 \
        --n-examples-paper 400 \
        --positive-ratio 0.5 \
        --dataset-version combined_abstention_benchmark_v1 \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATOR_DIR = SCRIPT_DIR
EVAL_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "dataset_generator" else SCRIPT_DIR
DATASETS_DIR = EVAL_DIR / "evaluation_datasets"

VT_SCRIPT = GENERATOR_DIR / "generate_variable_abstention_benchmark.py"
PAPER_SCRIPT = GENERATOR_DIR / "generate_paper_abstention_benchmark.py"

VT_OUTPUT = DATASETS_DIR / "variable_abstention" / "abstention_benchmark.jsonl"
VT_REPORT = DATASETS_DIR / "variable_abstention" / "abstention_benchmark_report.json"
VT_QA = DATASETS_DIR / "variable_abstention" / "abstention_benchmark_qa_sample.jsonl"

PAPER_OUTPUT = DATASETS_DIR / "paper_abstention" / "paper_abstention_benchmark.jsonl"
PAPER_REPORT = DATASETS_DIR / "paper_abstention" / "paper_abstention_benchmark_report.json"
PAPER_QA = DATASETS_DIR / "paper_abstention" / "paper_abstention_benchmark_qa_sample.jsonl"

DEFAULT_COMBINED_REPORT = DATASETS_DIR / "combined_abstention" / "combined_abstention_benchmark_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one or both abstention benchmark slices and write unified report.",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["both", "variables_tables", "papers"],
        default="both",
        help="Which benchmark slice(s) to generate.",
    )
    parser.add_argument("--n-examples-vt", type=int, default=500)
    parser.add_argument("--n-examples-paper", type=int, default=400)
    parser.add_argument("--positive-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qa-sample-fraction", type=float, default=0.15)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-attempt-multiplier", type=int, default=25)

    parser.add_argument(
        "--dataset-version",
        type=str,
        default="combined_abstention_benchmark_v1",
        help="Base version tag used for both child benchmark versions.",
    )

    parser.add_argument("--vt-output", type=Path, default=VT_OUTPUT)
    parser.add_argument("--vt-report", type=Path, default=VT_REPORT)
    parser.add_argument("--vt-qa", type=Path, default=VT_QA)

    parser.add_argument("--paper-output", type=Path, default=PAPER_OUTPUT)
    parser.add_argument("--paper-report", type=Path, default=PAPER_REPORT)
    parser.add_argument("--paper-qa", type=Path, default=PAPER_QA)

    parser.add_argument("--combined-report", type=Path, default=DEFAULT_COMBINED_REPORT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--disable-index-snapshot-filter",
        action="store_true",
        help="Pass through to variable/table generator only.",
    )
    return parser.parse_args()


def _run_step(command: list[str], step_name: str) -> None:
    print(f"\n▶ {step_name}")
    print("  " + " ".join(command))
    subprocess.run(command, check=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Expected report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()

    run_vt = args.target in {"both", "variables_tables"}
    run_paper = args.target in {"both", "papers"}

    if run_vt and args.n_examples_vt <= 0:
        raise ValueError("--n-examples-vt must be > 0 when target includes variables_tables")
    if run_paper and args.n_examples_paper <= 0:
        raise ValueError("--n-examples-paper must be > 0 when target includes papers")
    if not (0.0 < args.positive_ratio < 1.0):
        raise ValueError("--positive-ratio must be between 0 and 1")
    if min(args.train_ratio, args.dev_ratio, args.test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")
    if abs((args.train_ratio + args.dev_ratio + args.test_ratio) - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to 1")

    py = sys.executable

    vt_version = f"{args.dataset_version}__variables_tables"
    paper_version = f"{args.dataset_version}__papers"

    vt_cmd = [
        py,
        str(VT_SCRIPT),
        "--n-examples",
        str(args.n_examples_vt),
        "--positive-ratio",
        str(args.positive_ratio),
        "--seed",
        str(args.seed),
        "--qa-sample-fraction",
        str(args.qa_sample_fraction),
        "--train-ratio",
        str(args.train_ratio),
        "--dev-ratio",
        str(args.dev_ratio),
        "--test-ratio",
        str(args.test_ratio),
        "--max-attempt-multiplier",
        str(args.max_attempt_multiplier),
        "--dataset-version",
        vt_version,
        "--output",
        str(args.vt_output),
        "--report-output",
        str(args.vt_report),
        "--qa-sample-output",
        str(args.vt_qa),
    ]
    if args.overwrite:
        vt_cmd.append("--overwrite")
    if args.disable_index_snapshot_filter:
        vt_cmd.append("--disable-index-snapshot-filter")

    paper_cmd = [
        py,
        str(PAPER_SCRIPT),
        "--n-examples",
        str(args.n_examples_paper),
        "--positive-ratio",
        str(args.positive_ratio),
        "--seed",
        str(args.seed),
        "--qa-sample-fraction",
        str(args.qa_sample_fraction),
        "--train-ratio",
        str(args.train_ratio),
        "--dev-ratio",
        str(args.dev_ratio),
        "--test-ratio",
        str(args.test_ratio),
        "--max-attempt-multiplier",
        str(args.max_attempt_multiplier),
        "--dataset-version",
        paper_version,
        "--output",
        str(args.paper_output),
        "--report-output",
        str(args.paper_report),
        "--qa-sample-output",
        str(args.paper_qa),
    ]
    if args.overwrite:
        paper_cmd.append("--overwrite")

    vt_report: dict[str, Any] | None = None
    paper_report: dict[str, Any] | None = None

    if run_vt:
        _run_step(vt_cmd, "Generate variables/tables abstention benchmark")
        vt_report = _load_json(args.vt_report)

    if run_paper:
        _run_step(paper_cmd, "Generate paper abstention benchmark")
        paper_report = _load_json(args.paper_report)

    total_examples = int((vt_report or {}).get("n_examples", 0)) + int((paper_report or {}).get("n_examples", 0))
    total_positive = int((vt_report or {}).get("n_positive", 0)) + int((paper_report or {}).get("n_positive", 0))
    total_negative = int((vt_report or {}).get("n_negative", 0)) + int((paper_report or {}).get("n_negative", 0))

    split_totals = {
        "train": int((vt_report or {}).get("split_distribution", {}).get("train", 0))
        + int((paper_report or {}).get("split_distribution", {}).get("train", 0)),
        "dev": int((vt_report or {}).get("split_distribution", {}).get("dev", 0))
        + int((paper_report or {}).get("split_distribution", {}).get("dev", 0)),
        "test": int((vt_report or {}).get("split_distribution", {}).get("test", 0))
        + int((paper_report or {}).get("split_distribution", {}).get("test", 0)),
    }

    combined = {
        "dataset_version": args.dataset_version,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "target": args.target,
            "n_examples_vt": args.n_examples_vt,
            "n_examples_paper": args.n_examples_paper,
            "positive_ratio": args.positive_ratio,
            "seed": args.seed,
            "qa_sample_fraction": args.qa_sample_fraction,
            "split_ratios": {
                "train": args.train_ratio,
                "dev": args.dev_ratio,
                "test": args.test_ratio,
            },
            "max_attempt_multiplier": args.max_attempt_multiplier,
            "disable_index_snapshot_filter": args.disable_index_snapshot_filter,
        },
        "outputs": {
            "variables_tables": (
                {
                    "dataset": str(args.vt_output),
                    "report": str(args.vt_report),
                    "qa_sample": str(args.vt_qa),
                }
                if run_vt
                else None
            ),
            "papers": (
                {
                    "dataset": str(args.paper_output),
                    "report": str(args.paper_report),
                    "qa_sample": str(args.paper_qa),
                }
                if run_paper
                else None
            ),
        },
        "summary": {
            "n_examples_total": total_examples,
            "n_positive_total": total_positive,
            "n_negative_total": total_negative,
            "split_distribution_total": split_totals,
            "positive_ratio_total": (total_positive / total_examples) if total_examples else 0.0,
        },
        "benchmarks": {
            "variables_tables": vt_report if run_vt else None,
            "papers": paper_report if run_paper else None,
        },
    }

    args.combined_report.parent.mkdir(parents=True, exist_ok=True)
    args.combined_report.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n════════════════════════════════════════════════════")
    print(f"Combined dataset version: {args.dataset_version}")
    print(f"Target:                 {args.target}")
    print(f"Unified report:           {args.combined_report}")
    print(f"Total examples:           {total_examples}")
    print(f"Total positive:           {total_positive}")
    print(f"Total negative:           {total_negative}")
    print(
        "Total split distribution: "
        f"train={split_totals['train']}, dev={split_totals['dev']}, test={split_totals['test']}"
    )


if __name__ == "__main__":
    main()
