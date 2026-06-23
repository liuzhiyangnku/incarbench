#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_utils import utc_now
from generate_leaderboard_markdown_report import (
    DEFAULT_REPORT_NAME,
    load_summaries,
    report_markdown,
)
from incar_generation_utils import (
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
    apply_reference_adjustments,
    build_case_metadata,
    build_scoring_payload,
    dump_json,
    ensure_output_root,
    fetch_seed_data,
    incar_text_from_dict,
    load_problem_csv,
    mentioned_scoring_keys_union,
    missing_generation_grade,
    normalize_incar_dict,
    row_source_kind,
    score_generated_incar,
    summarize_generation_grades,
    to_poscar_string,
)
from llm_config_utils import load_llm_benchmark_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an INCAR-generation benchmark from a CSV file and Materials Project data."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "problems" / "problem_set_v1.0.csv",
        help="Problem CSV path",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "llm_benchmark_config.json",
        help="LLM benchmark config path",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output benchmark root directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild cases even if the target case directory already looks complete",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first failed case instead of recording the failure and continuing",
    )
    parser.add_argument(
        "--sync-existing-definitions",
        action="store_true",
        help=(
            "Rebuild INCAR_reference plus metadata/scoring/prompt files for existing cases "
            "from cached inputs/INCAR_mp_raw.json, without refetching seed data or rerunning inference."
        ),
    )
    parser.add_argument(
        "--rescore-existing-outputs",
        action="store_true",
        help=(
            "After building or syncing case definitions, rescore all discovered "
            "existing model outputs and regenerate leaderboard reports."
        ),
    )
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def case_is_complete(case_dir: Path) -> bool:
    required_paths = (
        case_dir / "inputs" / "POSCAR",
        case_dir / "inputs" / "INCAR_reference",
        case_dir / "inputs" / "INCAR_mp_raw.json",
        case_dir / "metadata.json",
        case_dir / "scoring.json",
        case_dir / "prompt_context.txt",
        case_dir / "model_outputs",
    )
    return all(path.exists() for path in required_paths)


def build_index(rows: list[dict[str, str]]) -> dict:
    return {
        "version": "0.1",
        "benchmark_type": "incar_generation",
        "total_cases": len(rows),
        "cases": [
            {
                "case_id": row["case_id"],
                "source_kind": row_source_kind(row),
                "mp_id": row["mp_id"],
                "formula": row["formula"],
                "task_type": row["task_type"],
                "task_family": row.get("task_family") or row["task_type"],
                "material_family": row.get("material_family", ""),
                "challenge_type": row.get("challenge_type", ""),
                "difficulty": row.get("difficulty", ""),
                "calc_type_pattern": row["calc_type_pattern"],
                "normalization_profile": row["normalization_profile"],
            }
            for row in rows
        ],
    }


def build_readme() -> str:
    return """# INCAR Generation Benchmark

This benchmark is generated from a CSV problem file plus either:

- Materials Project source data
- local structure and INCAR seed files

## Workflow

1. Edit a CSV like `problems/problem_set_v1.0.csv`
2. Choose one source per row:
   - `source_kind=mp`
   - `source_kind=local`
3. Build the benchmark:

```bash
python3 ./scripts/build_incar_generation_benchmark.py
```

The builder is resumable by default. Re-running the same command will skip any case
whose output files already exist. Use `--overwrite` only when you want to rebuild
everything from scratch.

4. Run enabled models:

```bash
python3 ./scripts/run_llm_incar_batch.py
```

5. Score all generated INCAR files:

```bash
python3 ./scripts/score_incar_generation_batch.py --model-name gpt4o
```
"""


def build_case(
    *,
    row: dict[str, str],
    csv_path: Path,
    api_key: str,
    output_root: Path,
    global_keep_keys: set[str],
) -> None:
    seed = fetch_seed_data(api_key=api_key, row=row, csv_path=csv_path)
    normalized = normalize_incar_dict(
        seed["raw_incar_params"],
        row["normalization_profile"],
        extra_keep_keys=global_keep_keys,
    )
    normalized = apply_reference_adjustments(
        normalized,
        remove_keys_raw=row.get("reference_remove_keys"),
        overrides_raw=row.get("reference_overrides"),
    )
    metadata = build_case_metadata(
        row=row,
        source_kind=seed["source_kind"],
        source_reference=seed["reference_source"],
        selected_task_id=seed["selected_task_id"],
        selected_calc_type=seed["selected_calc_type"],
        formula_pretty=seed["formula_pretty"],
    )
    scoring = build_scoring_payload(row, normalized)

    case_dir = output_root / "cases" / row["case_id"]
    write_text(case_dir / "inputs" / "POSCAR", to_poscar_string(seed["structure"]))
    write_text(case_dir / "inputs" / "INCAR_reference", incar_text_from_dict(normalized))
    dump_json(case_dir / "inputs" / "INCAR_mp_raw.json", seed["raw_incar_params"])
    dump_json(case_dir / "metadata.json", metadata)
    dump_json(case_dir / "scoring.json", scoring)
    write_text(case_dir / "prompt_context.txt", metadata.get("prompt_context", ""))
    (case_dir / "model_outputs").mkdir(parents=True, exist_ok=True)


def sync_existing_case_definition(
    *,
    row: dict[str, str],
    output_root: Path,
    global_keep_keys: set[str],
) -> None:
    case_dir = output_root / "cases" / row["case_id"]
    metadata_path = case_dir / "metadata.json"
    raw_seed_path = case_dir / "inputs" / "INCAR_mp_raw.json"
    if not metadata_path.exists() or not raw_seed_path.exists():
        raise FileNotFoundError(
            f"{row['case_id']}: existing case is missing metadata.json or inputs/INCAR_mp_raw.json"
        )

    existing_metadata = load_json(metadata_path)
    raw_incar_params = load_json(raw_seed_path)
    normalized = normalize_incar_dict(
        raw_incar_params,
        row["normalization_profile"],
        extra_keep_keys=global_keep_keys,
    )
    normalized = apply_reference_adjustments(
        normalized,
        remove_keys_raw=row.get("reference_remove_keys"),
        overrides_raw=row.get("reference_overrides"),
    )
    metadata = build_case_metadata(
        row=row,
        source_kind=str(existing_metadata.get("source_kind") or row_source_kind(row)),
        source_reference=existing_metadata.get("reference_source") or {},
        selected_task_id=str(existing_metadata.get("selected_task_id") or ""),
        selected_calc_type=str(existing_metadata.get("selected_calc_type") or ""),
        formula_pretty=str(existing_metadata.get("formula") or row.get("formula") or ""),
    )
    scoring = build_scoring_payload(row, normalized)
    write_text(case_dir / "inputs" / "INCAR_reference", incar_text_from_dict(normalized))
    dump_json(case_dir / "metadata.json", metadata)
    dump_json(case_dir / "scoring.json", scoring)
    write_text(case_dir / "prompt_context.txt", metadata.get("prompt_context", ""))


def discovered_model_names(benchmark_root: Path) -> list[str]:
    model_names: set[str] = set()
    case_root = benchmark_root / "cases"
    if not case_root.is_dir():
        return []
    for case_dir in sorted(path for path in case_root.iterdir() if path.is_dir()):
        model_outputs_dir = case_dir / "model_outputs"
        if not model_outputs_dir.is_dir():
            continue
        for model_dir in model_outputs_dir.iterdir():
            if model_dir.is_dir():
                model_names.add(model_dir.name)
    return sorted(model_names)


def rescore_existing_outputs(benchmark_root: Path) -> None:
    case_dirs = sorted(path for path in (benchmark_root / "cases").iterdir() if path.is_dir())
    model_names = discovered_model_names(benchmark_root)
    if not model_names:
        print("No discovered generation model outputs to rescore.", flush=True)
        return

    for model_name in model_names:
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
        print(f"Rescored generation model {model_name}", flush=True)

    summaries = load_summaries(benchmark_root / "leaderboards")
    markdown = report_markdown(summaries, benchmark_root)
    write_text(benchmark_root / "leaderboards" / DEFAULT_REPORT_NAME, markdown)
    print(
        f"Regenerated generation report {(benchmark_root / 'leaderboards' / DEFAULT_REPORT_NAME)}",
        flush=True,
    )


def build_report(
    *,
    rows: list[dict[str, str]],
    built_case_ids: list[str],
    synced_case_ids: list[str],
    skipped_case_ids: list[str],
    failures: list[dict[str, str]],
    output_root: Path,
    overwrite: bool,
    fail_fast: bool,
    sync_existing_definitions: bool,
    rescored_existing_outputs: bool,
) -> dict:
    return {
        "generated_at_utc": utc_now(),
        "benchmark_root": str(output_root),
        "total_cases": len(rows),
        "built_cases": built_case_ids,
        "synced_cases": synced_case_ids,
        "skipped_cases": skipped_case_ids,
        "failed_cases": failures,
        "counts": {
            "built": len(built_case_ids),
            "synced": len(synced_case_ids),
            "skipped": len(skipped_case_ids),
            "failed": len(failures),
        },
        "resume_enabled": not overwrite,
        "overwrite": overwrite,
        "fail_fast": fail_fast,
        "sync_existing_definitions": sync_existing_definitions,
        "rescored_existing_outputs": rescored_existing_outputs,
    }


def main() -> None:
    args = parse_args()

    rows = load_problem_csv(args.csv)
    global_keep_keys = mentioned_scoring_keys_union(rows)
    output_root = ensure_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cases").mkdir(exist_ok=True)
    (output_root / "leaderboards").mkdir(exist_ok=True)

    write_text(output_root / "README.md", build_readme())
    write_text(output_root / "problem_set.csv", args.csv.read_text(encoding="utf-8"))
    dump_json(output_root / "metadata_index.json", build_index(rows))

    built_case_ids: list[str] = []
    synced_case_ids: list[str] = []
    skipped_case_ids: list[str] = []
    failures: list[dict[str, str]] = []
    api_key: str | None = None

    for row in rows:
        case_dir = output_root / "cases" / row["case_id"]
        if args.sync_existing_definitions:
            try:
                if not case_dir.is_dir():
                    raise FileNotFoundError(
                        f"{row['case_id']}: case directory not found under {output_root / 'cases'}"
                    )
                sync_existing_case_definition(
                    row=row,
                    output_root=output_root,
                    global_keep_keys=global_keep_keys,
                )
            except Exception as exc:
                failure = {
                    "case_id": row["case_id"],
                    "source_kind": row_source_kind(row),
                    "mp_id": row.get("mp_id", ""),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                print(f"Failed to sync case {row['case_id']}: {exc}", flush=True)
                if args.fail_fast:
                    break
                continue

            synced_case_ids.append(row["case_id"])
            print(f"Synced existing case {row['case_id']}", flush=True)
            continue

        if case_is_complete(case_dir) and not args.overwrite:
            skipped_case_ids.append(row["case_id"])
            print(f"Skip existing case {row['case_id']}", flush=True)
            continue

        if api_key is None:
            config = load_llm_benchmark_config(args.config)
            api_key = config["materials_project"]["api_key"]

        try:
            build_case(
                row=row,
                csv_path=args.csv,
                api_key=api_key,
                output_root=output_root,
                global_keep_keys=global_keep_keys,
            )
        except Exception as exc:
            failure = {
                "case_id": row["case_id"],
                "source_kind": row_source_kind(row),
                "mp_id": row.get("mp_id", ""),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"Failed case {row['case_id']}: {exc}", flush=True)
            if args.fail_fast:
                break
            continue

        built_case_ids.append(row["case_id"])
        print(f"Built case {row['case_id']}", flush=True)

    report = build_report(
        rows=rows,
        built_case_ids=built_case_ids,
        synced_case_ids=synced_case_ids,
        skipped_case_ids=skipped_case_ids,
        failures=failures,
        output_root=output_root,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
        sync_existing_definitions=args.sync_existing_definitions,
        rescored_existing_outputs=args.rescore_existing_outputs,
    )
    report_path = output_root / "build_report.json"
    dump_json(report_path, report)

    if args.rescore_existing_outputs:
        rescore_existing_outputs(output_root)

    if failures:
        raise SystemExit(
            f"Build finished with {len(failures)} failed case(s); "
            f"{len(built_case_ids)} built, {len(synced_case_ids)} synced, {len(skipped_case_ids)} skipped. "
            f"See {report_path}"
        )

    print(
        f"Build finished: {len(built_case_ids)} built, {len(synced_case_ids)} synced, "
        f"{len(skipped_case_ids)} skipped. "
        f"Report: {report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
