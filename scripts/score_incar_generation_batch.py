#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from incar_generation_utils import (
    DEFAULT_OUTPUT_ROOT,
    dump_json,
    ensure_output_root,
    missing_generation_grade,
    score_generated_incar,
    summarize_generation_grades,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one model across all INCAR-generation benchmark cases."
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Benchmark root directory",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model label to score, or `all` to score every discovered model output.",
    )
    return parser.parse_args()


def discovered_model_names(benchmark_root: Path) -> list[str]:
    model_names: set[str] = set()
    case_dirs = sorted(path for path in (benchmark_root / "cases").iterdir() if path.is_dir())
    for case_dir in case_dirs:
        model_outputs_dir = case_dir / "model_outputs"
        if not model_outputs_dir.is_dir():
            continue
        for model_dir in model_outputs_dir.iterdir():
            if model_dir.is_dir():
                model_names.add(model_dir.name)
    return sorted(model_names)


def score_one_model(*, benchmark_root: Path, model_name: str) -> None:
    case_dirs = sorted(path for path in (benchmark_root / "cases").iterdir() if path.is_dir())
    grades: list[dict] = []

    for case_dir in case_dirs:
        candidate_path = case_dir / "model_outputs" / model_name / "INCAR_final"
        grade_path = case_dir / "model_outputs" / model_name / "grade.json"
        grade_path.parent.mkdir(parents=True, exist_ok=True)

        if candidate_path.exists():
            payload = score_generated_incar(
                case_dir=case_dir,
                candidate_path=candidate_path,
                model_name=model_name,
            )
        else:
            payload = missing_generation_grade(
                case_dir=case_dir,
                model_name=model_name,
                candidate_path=candidate_path,
            )

        dump_json(grade_path, payload)
        grades.append(payload)

    summary = summarize_generation_grades(
        benchmark_root=benchmark_root,
        model_name=model_name,
        grades=grades,
    )
    dump_json(benchmark_root / "leaderboards" / f"{model_name}_summary.json", summary)
    print(f"Scored {model_name}", flush=True)


def main() -> None:
    args = parse_args()
    benchmark_root = ensure_output_root(args.benchmark_root)

    if args.model_name == "all":
        model_names = discovered_model_names(benchmark_root)
        if not model_names:
            raise SystemExit(f"No model outputs found under {benchmark_root / 'cases'}")
        print(f"Scoring all discovered models: {', '.join(model_names)}", flush=True)
        for model_name in model_names:
            score_one_model(benchmark_root=benchmark_root, model_name=model_name)
        return

    score_one_model(benchmark_root=benchmark_root, model_name=args.model_name)


if __name__ == "__main__":
    main()
