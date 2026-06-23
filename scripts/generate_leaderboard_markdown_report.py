#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from benchmark_utils import load_json, utc_now

DEFAULT_BENCHMARK_ROOT = Path("incar_generation_benchmark")
DEFAULT_REPORT_NAME = "detailed_evaluation_report.md"
DIMENSION_SPECS = (
    ("by_difficulty", "By Difficulty"),
    ("by_task_type", "By Task Type"),
    ("by_task_family", "By Task Family"),
    ("by_material_family", "By Material Family"),
    ("by_challenge_type", "By Challenge Type"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown report from leaderboard summary JSON files."
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Benchmark root directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output Markdown file. Default: <benchmark-root>/leaderboards/detailed_evaluation_report.md",
    )
    return parser.parse_args()


def format_score(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def format_rate(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):.2f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def slugify(text: str) -> str:
    lowered = text.strip().lower()
    chars = []
    previous_dash = False
    for char in lowered:
        if char.isalnum():
            chars.append(char)
            previous_dash = False
        else:
            if not previous_dash:
                chars.append("-")
                previous_dash = True
    slug = "".join(chars).strip("-")
    return slug or "section"


def load_summaries(leaderboards_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(leaderboards_dir.glob("*_summary.json")):
        payload = load_json(path)
        payload["_path"] = str(path)
        summaries.append(payload)
    if not summaries:
        raise SystemExit(f"No summary JSON files found under {leaderboards_dir}")
    return summaries


def model_sort_key(summary: dict[str, Any]) -> tuple[float, float, float, str]:
    average_scores = summary.get("average_scores", {})
    must_match = average_scores.get("must_match")
    semantic = average_scores.get("must_match_semantic")
    policy = average_scores.get("must_match_policy")
    must_match_score = float(must_match) if must_match is not None else float("-inf")
    semantic_score = float(semantic) if semantic is not None else float("-inf")
    policy_score = float(policy) if policy is not None else float("-inf")
    must_match_score = float(must_match) if must_match is not None else float("-inf")
    return (-must_match_score, -semantic_score, -policy_score, str(summary.get("model_name", "")))


def format_group_scores(group_payload: dict[str, Any]) -> str:
    average_scores = group_payload.get("average_scores", {}) if group_payload else {}
    semantic = format_score(average_scores.get("must_match_semantic"))
    policy = format_score(average_scores.get("must_match_policy"))
    must_match = format_score(average_scores.get("must_match"))
    if must_match == semantic == policy == "-":
        return "-"
    return f"{semantic} / {policy} / {must_match}"


def overall_table(summaries: list[dict[str, Any]]) -> str:
    headers = [
        "Model",
        "Avg Must Match",
        "Avg Semantic",
        "Avg Policy",
        "Minimum Task-Runnable Rate",
        "Perfect Case Rate",
        "Graded",
        "Missing",
    ]
    rows: list[list[str]] = []
    for summary in sorted(summaries, key=model_sort_key):
        average_scores = summary.get("average_scores", {})
        rows.append(
            [
                str(summary["model_name"]),
                format_score(average_scores.get("must_match")),
                format_score(average_scores.get("must_match_semantic")),
                format_score(average_scores.get("must_match_policy")),
                format_rate(summary.get("minimum_task_runnable_rate")),
                format_rate(summary.get("perfect_case_rate")),
                f"{summary.get('graded_cases', 0)}/{summary.get('total_cases', 0)}",
                str(summary.get("missing_cases", 0)),
            ]
        )
    return markdown_table(headers, rows)


def best_model_for_group(
    summaries: list[dict[str, Any]],
    *,
    dimension_key: str,
    group_name: str,
) -> str:
    candidates: list[tuple[float, str]] = []
    for summary in summaries:
        group_payload = summary.get(dimension_key, {}).get(group_name)
        if not group_payload:
            continue
        value = group_payload.get("average_scores", {}).get("must_match")
        if value is None:
            continue
        candidates.append((float(value), str(summary["model_name"])))
    if not candidates:
        return "-"
    best_score, best_name = max(candidates, key=lambda item: (item[0], item[1]))
    return f"{best_name} ({best_score:.2f})"


def group_total_cases(
    summaries: list[dict[str, Any]],
    *,
    dimension_key: str,
    group_name: str,
) -> int:
    total_cases = 0
    for summary in summaries:
        group_payload = summary.get(dimension_key, {}).get(group_name)
        if not group_payload:
            continue
        total_cases = max(total_cases, int(group_payload.get("total_cases", 0)))
    return total_cases


def dimension_rows(
    summaries: list[dict[str, Any]],
    *,
    dimension_key: str,
) -> list[list[str]]:
    group_names: set[str] = set()
    for summary in summaries:
        group_names.update(summary.get(dimension_key, {}).keys())

    rows: list[list[str]] = []
    for group_name in sorted(group_names):
        row = [
            group_name,
            str(group_total_cases(summaries, dimension_key=dimension_key, group_name=group_name)),
        ]
        for summary in sorted(summaries, key=model_sort_key):
            group_payload = summary.get(dimension_key, {}).get(group_name, {})
            row.append(format_group_scores(group_payload))
        row.append(best_model_for_group(summaries, dimension_key=dimension_key, group_name=group_name))
        rows.append(row)
    return rows


def dimension_section(
    summaries: list[dict[str, Any]],
    *,
    dimension_key: str,
    title: str,
) -> str:
    model_names = [str(summary["model_name"]) for summary in sorted(summaries, key=model_sort_key)]
    headers = ["Group", "Cases", *model_names, "Best Model"]
    body = markdown_table(headers, dimension_rows(summaries, dimension_key=dimension_key))
    return f"## {title}\n\n{body}"


def report_markdown(summaries: list[dict[str, Any]], benchmark_root: Path) -> str:
    overview = overall_table(summaries)
    toc_lines = ["## Contents", ""]
    toc_lines.append("- [Overall Ranking](#overall-ranking)")
    for _, title in DIMENSION_SPECS:
        toc_lines.append(f"- [{title}](#{slugify(title)})")
    toc_lines.append("- [Case Details](#case-details)")

    sections = ["# Detailed Evaluation Report", ""]
    sections.append(f"- Generated at: `{utc_now()}`")
    sections.append(f"- Benchmark root: `{benchmark_root.resolve()}`")
    sections.append(f"- Models covered: `{len(summaries)}`")
    sections.append(
        "- Models are ranked primarily by `average_scores.must_match` from the summary JSON."
    )
    sections.append(
        "- `Minimum Task-Runnable Rate` is reported as a supplementary case-level feasibility metric and does not change the ranking order."
    )
    sections.append(
        "- Dimension table cells are shown as `must_match_semantic / must_match_policy / must_match`."
    )
    sections.append("")
    sections.extend(toc_lines)
    sections.append("")
    sections.append("## Overall Ranking")
    sections.append("")
    sections.append(overview)
    sections.append("")

    for dimension_key, title in DIMENSION_SPECS:
        sections.append(dimension_section(summaries, dimension_key=dimension_key, title=title))
        sections.append("")

    sections.append("## Case Details")
    sections.append("")
    sections.append(
        "Each row shows which semantic and policy keys were missed, which optional keys were not matched, and whether unsupported extra keys were added."
    )
    sections.append("")
    for summary in sorted(summaries, key=model_sort_key):
        sections.append(f"### {summary['model_name']}")
        sections.append("")
        headers = [
            "Case",
            "Task",
            "Material",
            "Challenge",
            "Minimum Runnable",
            "Runnable Failure Reasons",
            "Imputed Defaults",
            "Missing Semantic",
            "Missing Policy",
            "Optional Miss",
            "Extra Keys",
            "Semantic",
            "Policy",
            "Must",
            "Perfect",
        ]
        rows: list[list[str]] = []
        for case in summary.get("cases", []):
            case_context = case.get("case_context", {}) or {}
            missing_required = case.get("missing_required_keys", {}) or {}
            rows.append(
                [
                    str(case.get("case_id", "-")),
                    str(case_context.get("task_type") or "-"),
                    str(case_context.get("material_family") or "-"),
                    str(case_context.get("challenge_type") or "-"),
                    str(case.get("minimum_task_runnable", "-")),
                    ", ".join(case.get("minimum_task_runnable_reasons", []) or []) or "-",
                    ", ".join(
                        f"{key}={value}"
                        for key, value in sorted((case.get("default_imputed_keys") or {}).items())
                    )
                    or "-",
                    ", ".join(missing_required.get("semantic", []) or []) or "-",
                    ", ".join(missing_required.get("policy", []) or []) or "-",
                    ", ".join(case.get("optional_missed_keys", []) or []) or "-",
                    ", ".join(case.get("extra_keys", []) or []) or "-",
                    format_score((case.get("scores") or {}).get("must_match_semantic")),
                    format_score((case.get("scores") or {}).get("must_match_policy")),
                    format_score((case.get("scores") or {}).get("must_match")),
                    str(case.get("perfect_case", "-")),
                ]
            )
        sections.append(markdown_table(headers, rows))
        sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    leaderboards_dir = benchmark_root / "leaderboards"
    output_path = args.output or (leaderboards_dir / DEFAULT_REPORT_NAME)

    summaries = load_summaries(leaderboards_dir)
    markdown = report_markdown(summaries, benchmark_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
