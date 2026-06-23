from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"

TOTEN_RE = re.compile(rf"free\s+energy\s+TOTEN\s*=\s*({FLOAT_PATTERN})")
EFERMI_RE = re.compile(rf"E-fermi\s*:\s*({FLOAT_PATTERN})")
NELECT_MAG_RE = re.compile(
    rf"number of electron\s+({FLOAT_PATTERN})\s+magnetization\s+({FLOAT_PATTERN})"
)
ISPIN_RE = re.compile(r"\bISPIN\s*=\s*(\d+)")
OSZICAR_IONIC_RE = re.compile(
    rf"^\s*(\d+)\s+F=\s*({FLOAT_PATTERN})\s+E0=\s*({FLOAT_PATTERN})\s+d E =\s*({FLOAT_PATTERN})"
)
ELECTRONIC_STEP_RE = re.compile(r"^\s*(?:DAV|RMM|DIA|CGA|EDD):\s+(\d+)")

CONVERGENCE_MARKERS = (
    "aborting loop because EDIFF is reached",
    "reached required accuracy - stopping structural energy minimisation",
    "reached required accuracy - stopping structural energy minimization",
)
COMPLETION_MARKERS = (
    "General timing and accounting informations for this job",
    "Voluntary context switches:",
)
RUNTIME_SENSITIVE_FAILURES = {"cross_step", "misleading_error"}


@dataclass
class ParsedOutcar:
    total_energy: float | None
    efermi: float | None
    total_magnetization: float | None
    completed: bool
    electronic_converged: bool
    ispin: int | None


@dataclass
class ParsedOszicar:
    ionic_steps: int
    electronic_steps: int
    final_free_energy: float | None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def last_float_match(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_outcar(path: Path | None) -> ParsedOutcar:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return ParsedOutcar(
            total_energy=None,
            efermi=None,
            total_magnetization=None,
            completed=False,
            electronic_converged=False,
            ispin=None,
        )

    text = load_text(path)
    total_energy = last_float_match(TOTEN_RE, text)
    efermi = last_float_match(EFERMI_RE, text)

    magnetization = None
    magnetization_matches = NELECT_MAG_RE.findall(text)
    if magnetization_matches:
        magnetization = float(magnetization_matches[-1][1])

    ispin = None
    ispin_matches = ISPIN_RE.findall(text)
    if ispin_matches:
        ispin = int(ispin_matches[-1])

    if magnetization is None and ispin == 1:
        magnetization = 0.0

    completed = any(marker in text for marker in COMPLETION_MARKERS)
    electronic_converged = any(marker in text for marker in CONVERGENCE_MARKERS)

    return ParsedOutcar(
        total_energy=total_energy,
        efermi=efermi,
        total_magnetization=magnetization,
        completed=completed,
        electronic_converged=electronic_converged,
        ispin=ispin,
    )


def parse_oszicar(path: Path | None) -> ParsedOszicar:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return ParsedOszicar(ionic_steps=0, electronic_steps=0, final_free_energy=None)

    ionic_steps = 0
    electronic_steps = 0
    final_free_energy = None

    for raw_line in load_text(path).splitlines():
        ionic_match = OSZICAR_IONIC_RE.match(raw_line)
        if ionic_match:
            ionic_steps += 1
            final_free_energy = float(ionic_match.group(2))
            continue

        electronic_match = ELECTRONIC_STEP_RE.match(raw_line)
        if electronic_match:
            electronic_steps = max(electronic_steps, int(electronic_match.group(1)))

    return ParsedOszicar(
        ionic_steps=ionic_steps,
        electronic_steps=electronic_steps,
        final_free_energy=final_free_energy,
    )


def infer_converged(
    parsed_outcar: ParsedOutcar,
    parsed_oszicar: ParsedOszicar,
    exit_code: int | None,
) -> bool | None:
    if exit_code is not None and exit_code != 0:
        return False

    if not parsed_outcar.completed:
        return False

    if parsed_outcar.electronic_converged:
        return True

    if parsed_outcar.total_energy is not None and parsed_oszicar.ionic_steps > 0:
        return True

    if parsed_outcar.total_energy is not None and parsed_outcar.efermi is not None:
        return True

    return None


def extract_vasp_result(
    *,
    metadata: dict[str, Any] | None,
    natoms: int | None,
    outcar_path: Path | None,
    oszicar_path: Path | None,
    exit_code: int | None,
) -> dict[str, Any]:
    parsed_outcar = parse_outcar(outcar_path)
    parsed_oszicar = parse_oszicar(oszicar_path)

    if natoms is None and metadata is not None:
        natoms = metadata.get("system", {}).get("natoms")

    energy_per_atom = None
    if parsed_outcar.total_energy is not None and natoms:
        energy_per_atom = parsed_outcar.total_energy / natoms

    return {
        "total_magnetization": parsed_outcar.total_magnetization,
        "energy_per_atom": energy_per_atom,
        "converged": infer_converged(parsed_outcar, parsed_oszicar, exit_code),
        "vasp_exit_code": exit_code,
        "total_energy": parsed_outcar.total_energy,
        "natoms": natoms,
        "efermi": parsed_outcar.efermi,
        "completed": parsed_outcar.completed,
        "electronic_converged": parsed_outcar.electronic_converged,
        "ionic_steps": parsed_oszicar.ionic_steps,
        "electronic_steps": parsed_oszicar.electronic_steps,
        "final_free_energy": parsed_oszicar.final_free_energy,
    }


def result_loaded(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False

    interesting_keys = (
        "total_magnetization",
        "energy_per_atom",
        "converged",
        "vasp_exit_code",
        "total_energy",
    )
    return any(payload.get(key) is not None for key in interesting_keys)


def metric_difference(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    metric_name: str | None,
) -> float | None:
    if not metric_name or not left or not right:
        return None

    left_value = left.get(metric_name)
    right_value = right.get(metric_name)
    if left_value is None or right_value is None:
        return None
    return abs(left_value - right_value)


def bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def evaluate_case_payload(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = case_dir / "metadata.json"
    metadata = load_json(metadata_path)

    trap_result_path = case_dir / "results" / "trap" / "result.json"
    reference_result_path = case_dir / "results" / "reference" / "result.json"

    trap_result = load_json(trap_result_path) if trap_result_path.exists() else None
    reference_result = (
        load_json(reference_result_path) if reference_result_path.exists() else None
    )

    criterion = metadata.get("failure_criterion", {})
    primary_name = criterion.get("primary_metric")
    secondary_name = criterion.get("secondary_metric")
    primary_threshold = criterion.get("threshold")
    secondary_threshold = criterion.get("secondary_threshold")

    primary_diff = metric_difference(trap_result, reference_result, primary_name)
    secondary_diff = metric_difference(trap_result, reference_result, secondary_name)

    primary_triggered = (
        primary_diff is not None
        and primary_threshold is not None
        and primary_diff >= primary_threshold
    )
    secondary_triggered = (
        secondary_diff is not None
        and secondary_threshold is not None
        and secondary_diff >= secondary_threshold
    )

    trap_loaded = result_loaded(trap_result)
    reference_loaded = result_loaded(reference_result)

    trap_exit_code = trap_result.get("vasp_exit_code") if trap_result else None
    reference_exit_code = reference_result.get("vasp_exit_code") if reference_result else None
    runtime_failure = (
        trap_exit_code not in (None, 0) and reference_exit_code in (None, 0)
    )

    notes: list[str] = []
    if not trap_loaded:
        notes.append("trap result missing or empty")
    if not reference_loaded:
        notes.append("reference result missing or empty")

    failure_type = metadata.get("failure_type")
    enough_metric_info = primary_diff is not None and (
        secondary_name is None or secondary_diff is not None
    )

    failure_confirmed: bool | None
    if failure_type in RUNTIME_SENSITIVE_FAILURES and runtime_failure:
        failure_confirmed = True
        notes.append("trap runtime failure observed while reference remained runnable")
    elif primary_triggered or secondary_triggered:
        failure_confirmed = True
    elif enough_metric_info and trap_loaded and reference_loaded:
        failure_confirmed = False
    else:
        failure_confirmed = None

    if trap_result:
        metadata["trap_result"] = {
            "total_magnetization": trap_result.get("total_magnetization"),
            "energy_per_atom": trap_result.get("energy_per_atom"),
            "converged": trap_result.get("converged"),
            "vasp_exit_code": trap_result.get("vasp_exit_code"),
        }

    if reference_result:
        metadata["ground_truth"] = {
            "total_magnetization": reference_result.get("total_magnetization"),
            "energy_per_atom": reference_result.get("energy_per_atom"),
            "converged": reference_result.get("converged"),
        }

    report = {
        "case_id": metadata.get("case_id"),
        "failure_type": failure_type,
        "trap_result_loaded": trap_loaded,
        "reference_result_loaded": reference_loaded,
        "failure_confirmed": failure_confirmed,
        "observed_failure_mode": (
            "trap_runtime_error"
            if failure_type in RUNTIME_SENSITIVE_FAILURES and runtime_failure
            else "metric_deviation"
            if primary_triggered or secondary_triggered
            else "insufficient_data"
            if failure_confirmed is None
            else "no_failure_signal"
        ),
        "primary_metric": {
            "name": primary_name,
            "trap_value": trap_result.get(primary_name) if trap_result and primary_name else None,
            "reference_value": (
                reference_result.get(primary_name)
                if reference_result and primary_name
                else None
            ),
            "absolute_difference": primary_diff,
            "threshold": primary_threshold,
            "triggered": primary_triggered if primary_diff is not None else None,
        },
        "secondary_metric": {
            "name": secondary_name,
            "trap_value": (
                trap_result.get(secondary_name) if trap_result and secondary_name else None
            ),
            "reference_value": (
                reference_result.get(secondary_name)
                if reference_result and secondary_name
                else None
            ),
            "absolute_difference": secondary_diff,
            "threshold": secondary_threshold,
            "triggered": secondary_triggered if secondary_diff is not None else None,
        },
        "convergence": {
            "trap": trap_result.get("converged") if trap_result else None,
            "reference": reference_result.get("converged") if reference_result else None,
        },
        "exit_codes": {
            "trap": trap_exit_code,
            "reference": reference_exit_code,
        },
        "notes": "; ".join(notes),
        "generated_at_utc": utc_now(),
    }

    return metadata, report


def summarize_dataset(dataset_root: Path) -> dict[str, Any]:
    case_dirs = sorted(
        path for path in (dataset_root / "cases").iterdir() if path.is_dir()
    )
    cases: list[dict[str, Any]] = []

    failure_type_summary: dict[str, dict[str, int]] = {}
    confirmed_failures = 0
    evaluated_cases = 0
    pending_cases = 0

    for case_dir in case_dirs:
        metadata_path = case_dir / "metadata.json"
        report_path = case_dir / "analysis" / "failure_report.json"
        if not metadata_path.exists():
            continue

        metadata = load_json(metadata_path)
        failure_type = metadata.get("failure_type", "unknown")
        summary_bucket = failure_type_summary.setdefault(
            failure_type,
            {"total": 0, "evaluated": 0, "confirmed": 0, "pending": 0},
        )
        summary_bucket["total"] += 1

        report = load_json(report_path) if report_path.exists() else None
        failure_confirmed = report.get("failure_confirmed") if report else None
        trap_loaded = report.get("trap_result_loaded") if report else False
        reference_loaded = report.get("reference_result_loaded") if report else False

        if failure_confirmed is None:
            pending_cases += 1
            summary_bucket["pending"] += 1
        else:
            evaluated_cases += 1
            summary_bucket["evaluated"] += 1
            if failure_confirmed:
                confirmed_failures += 1
                summary_bucket["confirmed"] += 1

        cases.append(
            {
                "case_id": metadata.get("case_id", case_dir.name),
                "failure_type": failure_type,
                "trap_result_loaded": trap_loaded,
                "reference_result_loaded": reference_loaded,
                "failure_confirmed": failure_confirmed,
                "report_path": str(report_path.relative_to(dataset_root.parent)),
            }
        )

    return {
        "generated_at_utc": utc_now(),
        "dataset_root": str(dataset_root),
        "total_cases": len(cases),
        "evaluated_cases": evaluated_cases,
        "confirmed_failures": confirmed_failures,
        "pending_cases": pending_cases,
        "failure_type_summary": failure_type_summary,
        "cases": cases,
    }
