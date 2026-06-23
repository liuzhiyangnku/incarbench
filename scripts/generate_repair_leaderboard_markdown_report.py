#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from incar_repair_utils import (
    DEFAULT_REPAIR_OUTPUT_ROOT,
    DEFAULT_REPAIR_REPORT_NAME,
    load_repair_summaries,
    repair_report_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown report from INCAR-repair leaderboard summary JSON files."
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_REPAIR_OUTPUT_ROOT,
        help="Repair benchmark root directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output Markdown file. Default: <benchmark-root>/leaderboards/repair_evaluation_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    leaderboards_dir = benchmark_root / "leaderboards"
    output_path = args.output or (leaderboards_dir / DEFAULT_REPAIR_REPORT_NAME)
    summaries = load_repair_summaries(leaderboards_dir)
    markdown = repair_report_markdown(summaries, benchmark_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Repair report written to: {output_path}")


if __name__ == "__main__":
    main()
