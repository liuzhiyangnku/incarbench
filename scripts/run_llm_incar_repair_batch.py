#!/usr/bin/env python3
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from benchmark_utils import dump_json, utc_now
from incar_generation_utils import invoke_model_with_retries
from incar_repair_utils import (
    DEFAULT_REPAIR_OUTPUT_ROOT,
    enabled_models,
    ensure_output_root,
    extract_incar_from_response,
    repair_case_dirs,
    repair_prompt_messages_for_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run enabled LLMs over all INCAR-repair benchmark cases."
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_REPAIR_OUTPUT_ROOT,
        help="Repair benchmark root directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/llm_benchmark_config.json"),
        help="LLM benchmark config path",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        dest="model_names",
        help="Only run the named model. Repeat for multiple models.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Only run the named repair case. Repeat for multiple cases.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing INCAR_fixed outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without calling models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_root = ensure_output_root(args.benchmark_root)
    models = enabled_models(args.config, args.model_names)
    if not models:
        raise SystemExit("No enabled models selected in llm_benchmark_config.json")

    case_dirs = repair_case_dirs(benchmark_root, args.case_ids)

    print(
        f"Starting repair batch run: {len(models)} model(s), {len(case_dirs)} case(s), "
        f"benchmark_root={benchmark_root}",
        flush=True,
    )

    for model_cfg in models:
        model_name = model_cfg["name"]
        successes = 0
        failures = 0
        print(f"Starting model {model_name}", flush=True)
        for case_dir in case_dirs:
            output_dir = case_dir / "model_outputs" / model_name
            fixed_path = output_dir / "INCAR_fixed"
            error_path = output_dir / "error.json"

            if fixed_path.exists() and not args.overwrite:
                print(f"Skip existing {model_name} {case_dir.name}", flush=True)
                continue

            if args.dry_run:
                print(f"[dry-run] {model_name} -> {case_dir.name}", flush=True)
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            response_path = output_dir / "response.json"
            request_path = output_dir / "request.json"

            messages = repair_prompt_messages_for_case(case_dir)
            dump_json(
                request_path,
                {
                    "generated_at_utc": utc_now(),
                    "model_name": model_name,
                    "messages": messages,
                },
            )

            print(f"Running {model_name} {case_dir.name}", flush=True)
            try:
                text, raw = invoke_model_with_retries(model_cfg=model_cfg, messages=messages)
                fixed_text = extract_incar_from_response(text)
                fixed_path.write_text(fixed_text, encoding="utf-8")
                dump_json(
                    response_path,
                    {
                        "generated_at_utc": utc_now(),
                        "model_name": model_name,
                        "raw_response": raw,
                        "extracted_text": fixed_text,
                    },
                )
                if error_path.exists():
                    error_path.unlink()
                successes += 1
                print(f"Wrote {fixed_path}", flush=True)
            except Exception as exc:
                failures += 1
                dump_json(
                    error_path,
                    {
                        "generated_at_utc": utc_now(),
                        "model_name": model_name,
                        "case_id": case_dir.name,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(f"Failed {model_name} {case_dir.name}: {exc}", flush=True)

        print(
            f"Completed model {model_name}: {successes} succeeded, {failures} failed",
            flush=True,
        )

    print("Repair batch run completed.", flush=True)


if __name__ == "__main__":
    main()
