#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from benchmark_utils import utc_now
from incar_repair_utils import (
    DEFAULT_REPAIR_OUTPUT_ROOT,
    DEFAULT_REPAIR_REPORT_NAME,
    DEFAULT_SOURCE_BENCHMARK_ROOT,
    build_repair_variants,
    build_repair_case_metadata,
    build_repair_index,
    build_repair_readme,
    dump_json,
    ensure_output_root,
    incar_text_from_dict,
    load_repair_summaries,
    load_source_case,
    missing_repair_grade,
    repair_case_dirs,
    repair_report_markdown,
    source_case_ids_from_root,
    repair_case_id_for,
    score_repaired_incar,
    summarize_repair_grades,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a standalone MVP INCAR-repair benchmark from the existing generation benchmark."
    )
    parser.add_argument(
        "--source-benchmark-root",
        type=Path,
        default=DEFAULT_SOURCE_BENCHMARK_ROOT,
        help="Source generation benchmark root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_REPAIR_OUTPUT_ROOT,
        help="Output repair benchmark root",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Only include the named source case. Repeat for multiple cases.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild cases even if the repair case directory already exists",
    )
    parser.add_argument(
        "--sync-existing-definitions",
        action="store_true",
        help=(
            "Update existing repair-case metadata/scoring/manifests from the current "
            "source generation benchmark without rerunning inference."
        ),
    )
    parser.add_argument(
        "--rescore-existing-outputs",
        action="store_true",
        help=(
            "After building or syncing repair cases, rescore all discovered "
            "existing repair outputs and regenerate leaderboard reports."
        ),
    )
    return parser.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def repair_case_is_complete(case_dir: Path) -> bool:
    required_paths = (
        case_dir / "inputs" / "POSCAR",
        case_dir / "inputs" / "INCAR_reference",
        case_dir / "inputs" / "INCAR_bad",
        case_dir / "inputs" / "error_manifest.json",
        case_dir / "metadata.json",
        case_dir / "scoring.json",
        case_dir / "model_outputs",
    )
    return all(path.exists() for path in required_paths)


def write_repair_case(
    *,
    source_case_id: str,
    source_case: dict,
    variant_type: str,
    bad_params: dict[str, str],
    error_manifest: dict,
    output_root: Path,
) -> tuple[str, dict]:
    repair_case_id = repair_case_id_for(
        source_case_id,
        variant_type,
        error_manifest["corruption_family"],
    )
    repair_case_dir = output_root / "cases" / repair_case_id
    repair_metadata = build_repair_case_metadata(
        source_case["metadata"],
        error_manifest,
        variant_type=variant_type,
    )
    error_manifest["repair_case_id"] = repair_case_id
    error_manifest["source_reference_snapshot"] = {
        key: source_case["reference_params"].get(key)
        for key in error_manifest["target_keys"]
    }
    error_manifest["corrupted_snapshot"] = {
        key: bad_params.get(key) for key in error_manifest["target_keys"]
    }

    write_text(repair_case_dir / "inputs" / "POSCAR", source_case["poscar_text"])
    write_text(repair_case_dir / "inputs" / "INCAR_reference", source_case["reference_text"])
    write_text(repair_case_dir / "inputs" / "INCAR_bad", incar_text_from_dict(bad_params))
    dump_json(repair_case_dir / "inputs" / "error_manifest.json", error_manifest)
    dump_json(repair_case_dir / "metadata.json", repair_metadata)
    dump_json(repair_case_dir / "scoring.json", source_case["scoring"])
    write_text(repair_case_dir / "prompt_context.txt", repair_metadata.get("prompt_context", ""))
    (repair_case_dir / "model_outputs").mkdir(parents=True, exist_ok=True)
    return repair_case_id, repair_metadata


def discovered_model_names(benchmark_root: Path) -> list[str]:
    model_names: set[str] = set()
    for case_dir in repair_case_dirs(benchmark_root):
        model_outputs_dir = case_dir / "model_outputs"
        if not model_outputs_dir.is_dir():
            continue
        for model_dir in model_outputs_dir.iterdir():
            if model_dir.is_dir():
                model_names.add(model_dir.name)
    return sorted(model_names)


def rescore_existing_outputs(benchmark_root: Path) -> None:
    case_dirs = repair_case_dirs(benchmark_root)
    model_names = discovered_model_names(benchmark_root)
    if not model_names:
        print("No discovered repair model outputs to rescore.", flush=True)
        return

    for model_name in model_names:
        grades: list[dict] = []
        for case_dir in case_dirs:
            candidate_path = case_dir / "model_outputs" / model_name / "INCAR_fixed"
            grade_path = case_dir / "model_outputs" / model_name / "grade.json"
            grade_path.parent.mkdir(parents=True, exist_ok=True)

            if candidate_path.exists():
                payload = score_repaired_incar(
                    case_dir=case_dir,
                    candidate_path=candidate_path,
                    model_name=model_name,
                )
            else:
                payload = missing_repair_grade(
                    case_dir=case_dir,
                    model_name=model_name,
                    candidate_path=candidate_path,
                )

            dump_json(grade_path, payload)
            grades.append(payload)

        summary = summarize_repair_grades(
            benchmark_root=benchmark_root,
            model_name=model_name,
            grades=grades,
        )
        dump_json(benchmark_root / "leaderboards" / f"{model_name}_summary.json", summary)
        print(f"Rescored repair model {model_name}", flush=True)

    summaries = load_repair_summaries(benchmark_root / "leaderboards")
    markdown = repair_report_markdown(summaries, benchmark_root)
    write_text(benchmark_root / "leaderboards" / DEFAULT_REPAIR_REPORT_NAME, markdown)
    print(
        f"Regenerated repair report {(benchmark_root / 'leaderboards' / DEFAULT_REPAIR_REPORT_NAME)}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    source_root = ensure_output_root(args.source_benchmark_root)
    output_root = ensure_output_root(args.output_root)
    source_case_ids = source_case_ids_from_root(source_root, args.case_ids)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cases").mkdir(exist_ok=True)
    (output_root / "leaderboards").mkdir(exist_ok=True)

    write_text(output_root / "README.md", build_repair_readme())

    built_cases: list[dict] = []
    synced_cases: list[str] = []
    skipped_cases: list[str] = []
    failed_cases: list[dict[str, str]] = []

    for source_case_id in source_case_ids:
        source_case_dir = source_root / "cases" / source_case_id
        if not source_case_dir.is_dir():
            failed_cases.append({"source_case_id": source_case_id, "error": "source case not found"})
            print(f"Missing source case {source_case_id}", flush=True)
            continue

        try:
            source_case = load_source_case(source_case_dir)
            variants = build_repair_variants(
                metadata=source_case["metadata"],
                reference_params=source_case["reference_params"],
            )
            for variant_type, bad_params, error_manifest in variants:
                repair_case_id = repair_case_id_for(
                    source_case_id,
                    variant_type,
                    error_manifest["corruption_family"],
                )
                repair_case_dir = output_root / "cases" / repair_case_id

                if args.sync_existing_definitions:
                    if not repair_case_dir.is_dir():
                        raise FileNotFoundError(
                            f"{repair_case_id}: repair case directory not found under {output_root / 'cases'}"
                        )
                    repair_case_id, repair_metadata = write_repair_case(
                        source_case_id=source_case_id,
                        source_case=source_case,
                        variant_type=variant_type,
                        bad_params=bad_params,
                        error_manifest=error_manifest,
                        output_root=output_root,
                    )
                    synced_cases.append(repair_case_id)
                    built_cases.append(
                        {
                            "case_id": repair_case_id,
                            "source_case_id": source_case_id,
                            "repair_variant_type": variant_type,
                            "corruption_family": error_manifest["corruption_family"],
                            "difficulty": repair_metadata.get("difficulty"),
                            "task_type": repair_metadata.get("task_type"),
                            "task_family": repair_metadata.get("task_family"),
                            "material_family": repair_metadata.get("material_family"),
                            "challenge_type": repair_metadata.get("challenge_type"),
                        }
                    )
                    print(f"Synced existing repair case {repair_case_id}", flush=True)
                    continue

                if repair_case_is_complete(repair_case_dir) and not args.overwrite:
                    skipped_cases.append(repair_case_id)
                    built_cases.append(
                        {
                            "case_id": repair_case_id,
                            "source_case_id": source_case_id,
                            "repair_variant_type": variant_type,
                            "corruption_family": error_manifest["corruption_family"],
                            "difficulty": source_case["metadata"].get("difficulty"),
                            "task_type": source_case["metadata"].get("task_type"),
                            "task_family": source_case["metadata"].get("task_family"),
                            "material_family": source_case["metadata"].get("material_family"),
                            "challenge_type": source_case["metadata"].get("challenge_type"),
                        }
                    )
                    print(f"Skip existing repair case {repair_case_id}", flush=True)
                    continue

                repair_case_id, repair_metadata = write_repair_case(
                    source_case_id=source_case_id,
                    source_case=source_case,
                    variant_type=variant_type,
                    bad_params=bad_params,
                    error_manifest=error_manifest,
                    output_root=output_root,
                )

                built_cases.append(
                    {
                        "case_id": repair_case_id,
                        "source_case_id": source_case_id,
                        "repair_variant_type": variant_type,
                        "corruption_family": error_manifest["corruption_family"],
                        "difficulty": repair_metadata.get("difficulty"),
                        "task_type": repair_metadata.get("task_type"),
                        "task_family": repair_metadata.get("task_family"),
                        "material_family": repair_metadata.get("material_family"),
                        "challenge_type": repair_metadata.get("challenge_type"),
                    }
                )
                print(f"Built repair case {repair_case_id}", flush=True)
        except Exception as exc:
            failed_cases.append({"source_case_id": source_case_id, "error": str(exc)})
            print(f"Failed repair case {source_case_id}: {exc}", flush=True)

    dump_json(output_root / "metadata_index.json", build_repair_index(built_cases))
    dump_json(
        output_root / "build_report.json",
        {
            "generated_at_utc": utc_now(),
            "source_benchmark_root": str(source_root),
            "output_root": str(output_root),
            "requested_source_case_ids": source_case_ids,
            "built_cases": built_cases,
            "synced_cases": synced_cases,
            "skipped_cases": skipped_cases,
            "failed_cases": failed_cases,
            "sync_existing_definitions": args.sync_existing_definitions,
            "rescored_existing_outputs": args.rescore_existing_outputs,
        },
    )

    if args.rescore_existing_outputs:
        rescore_existing_outputs(output_root)

    print(
        "Repair benchmark ready: "
        f"built={len(built_cases)} "
        f"synced={len(synced_cases)} "
        f"skipped={len(skipped_cases)} "
        f"failed={len(failed_cases)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
