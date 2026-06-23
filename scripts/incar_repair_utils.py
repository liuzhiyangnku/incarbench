from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from benchmark_utils import utc_now
from incar_generation_utils import (
    canonicalize_key,
    dump_json,
    enabled_models,
    ensure_output_root,
    extract_incar_from_response,
    is_ismear_policy_consistent,
    incar_text_from_dict,
    invoke_model,
    load_json,
    missing_generation_grade,
    parse_incar_file,
    runnable_policy_matches,
    score_generated_incar,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_REPAIR_OUTPUT_ROOT = Path("incar_repair_benchmark")
DEFAULT_SOURCE_BENCHMARK_ROOT = Path("incar_generation_benchmark")
DEFAULT_REPAIR_REPORT_NAME = "repair_evaluation_report.md"
DEFAULT_REPAIR_PROMPT_TEMPLATE_PATH = REPO_ROOT / "config" / "incar_repair_prompt_template.json"
REPAIR_DIMENSION_SPECS = (
    ("by_difficulty", "By Difficulty"),
    ("by_task_type", "By Task Type"),
    ("by_task_family", "By Task Family"),
    ("by_material_family", "By Material Family"),
    ("by_challenge_type", "By Challenge Type"),
    ("by_corruption_family", "By Corruption Family"),
)
REPAIR_VARIANT_ORDER = (
    "task_driven",
    "material_driven",
    "control",
)
SPECIAL_BLOCK_KEYS = (
    "LDAU",
    "LDAUTYPE",
    "LDAUL",
    "LDAUU",
    "LDAUJ",
    "LMAXMIX",
    "LASPH",
    "LSORBIT",
    "SAXIS",
    "IVDW",
    "ISPIN",
    "MAGMOM",
)

RUNNABLE_POLICY_KEYS = {
    "LDAU",
    "LDAUTYPE",
    "LDAUL",
    "LDAUU",
    "LDAUJ",
    "IVDW",
    "LSORBIT",
}


def _is_runnable_policy_key(key: str, reference_params: dict[str, str]) -> bool:
    if key == "ISMEAR":
        ref_ismear = _integer_value(reference_params.get("ISMEAR"))
        return ref_ismear is not None
    return key in RUNNABLE_POLICY_KEYS

NONMAGNETIC_GAPPED_FAMILIES = {
    "semiconductor",
    "ionic_insulator",
    "binary_compound",
    "oxide",
}
CORRELATED_FAMILIES = {
    "correlated_oxide",
    "transition_metal_oxide",
    "battery_material",
    "perovskite_multinary_oxide",
}
LAYERED_FAMILIES = {
    "layered_material",
    "layered_anisotropic",
    "chalcogenide",
}


def _slugify(raw: str) -> str:
    text = raw.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _update_key(
    params: dict[str, str],
    manifest: dict[str, Any],
    key: str,
    new_value: str | None,
    *,
    reason: str,
) -> None:
    canonical_key = canonicalize_key(key)
    original_present = canonical_key in params
    original_value = params.get(canonical_key)
    if new_value is None:
        params.pop(canonical_key, None)
    else:
        params[canonical_key] = str(new_value)

    manifest["mutations"].append(
        {
            "key": canonical_key,
            "operation": "remove" if new_value is None else ("replace" if original_present else "insert"),
            "before": original_value,
            "after": None if new_value is None else str(new_value),
            "reason": reason,
        }
    )
    if canonical_key not in manifest["target_keys"]:
        manifest["target_keys"].append(canonical_key)


def _smearing_confusion(params: dict[str, str], metadata: dict[str, Any], manifest: dict[str, Any]) -> None:
    current = params.get("ISMEAR")
    if current in {"1", "2"}:
        target_ismear = "0"
        target_sigma = "0.05"
    else:
        target_ismear = "1"
        target_sigma = "0.2"
    _update_key(params, manifest, "ISMEAR", target_ismear, reason="Inject metal/semiconductor smearing confusion")
    if "SIGMA" in params:
        _update_key(params, manifest, "SIGMA", target_sigma, reason="Keep the wrong smearing pair consistent")


def _relax_as_static(params: dict[str, str], metadata: dict[str, Any], manifest: dict[str, Any]) -> None:
    _update_key(params, manifest, "IBRION", "-1", reason="Force a static-style ionic update mode")
    _update_key(params, manifest, "NSW", "0", reason="Remove ionic steps from a relaxation case")
    if "EDIFFG" in params:
        _update_key(params, manifest, "EDIFFG", None, reason="Drop the force convergence target")


def _nscf_as_scf(params: dict[str, str], metadata: dict[str, Any], manifest: dict[str, Any]) -> None:
    _update_key(params, manifest, "ICHARG", "2", reason="Make the NSCF step self-consistent")
    if params.get("ISYM") == "0":
        _update_key(params, manifest, "ISYM", "2", reason="Re-enable symmetry for an NSCF workflow")


def _dftu_drop(params: dict[str, str], metadata: dict[str, Any], manifest: dict[str, Any]) -> None:
    for key in ("LDAU", "LDAUTYPE", "LDAUL", "LDAUU", "LDAUJ", "LMAXMIX", "LASPH"):
        if key in params:
            _update_key(params, manifest, key, None, reason="Remove the DFT+U block")


def _spin_drop(params: dict[str, str], metadata: dict[str, Any], manifest: dict[str, Any]) -> None:
    if "ISPIN" in params:
        _update_key(params, manifest, "ISPIN", "1", reason="Collapse a spin-polarized setup to non-spin")
    if "MAGMOM" in params:
        _update_key(params, manifest, "MAGMOM", None, reason="Remove magnetic initialization")


def _spin_dftu_drop(params: dict[str, str], metadata: dict[str, Any], manifest: dict[str, Any]) -> None:
    _spin_drop(params, metadata, manifest)
    _dftu_drop(params, metadata, manifest)


def _soc_drop(params: dict[str, str], metadata: dict[str, Any], manifest: dict[str, Any]) -> None:
    if "LSORBIT" in params:
        _update_key(params, manifest, "LSORBIT", None, reason="Drop SOC activation")
    if "SAXIS" in params:
        _update_key(params, manifest, "SAXIS", None, reason="Drop SOC spin axis metadata")
    _update_key(params, manifest, "ISYM", "2", reason="Re-enable symmetry after removing SOC")


def _vdw_drop(params: dict[str, str], metadata: dict[str, Any], manifest: dict[str, Any]) -> None:
    if "IVDW" in params:
        _update_key(params, manifest, "IVDW", None, reason="Remove the vdW correction")


def _symmetry_enable(params: dict[str, str], metadata: dict[str, Any], manifest: dict[str, Any]) -> None:
    _update_key(params, manifest, "ISYM", "2", reason="Force symmetry back on")


def _generic_required_drop(params: dict[str, str], metadata: dict[str, Any], manifest: dict[str, Any]) -> None:
    candidate_keys = []
    for key in (
        metadata.get("repair_priority_keys", [])
        if isinstance(metadata.get("repair_priority_keys"), list)
        else []
    ):
        if key in params:
            candidate_keys.append(key)

    for key in ("ICHARG", "ISPIN", "EDIFF", "ENCUT", "ISMEAR"):
        if key in params and key not in candidate_keys:
            candidate_keys.append(key)

    if not candidate_keys:
        raise ValueError(f"{metadata['case_id']}: no suitable key found for generic corruption")
    _update_key(
        params,
        manifest,
        candidate_keys[0],
        None,
        reason="Remove one required key as a generic repair target",
    )


def _static_as_relax(params: dict[str, str], metadata: dict[str, Any], manifest: dict[str, Any]) -> None:
    _update_key(params, manifest, "IBRION", "2", reason="Turn a static step into an ionic relaxation")
    _update_key(params, manifest, "NSW", "99", reason="Allow ionic steps in a static task")
    _update_key(params, manifest, "ISIF", "3", reason="Promote the static calculation to a full relaxation")
    if "EDIFFG" not in params:
        _update_key(params, manifest, "EDIFFG", "-0.02", reason="Add a force-based stopping criterion")


def _metal_as_gapped(params: dict[str, str], metadata: dict[str, Any], manifest: dict[str, Any]) -> None:
    _update_key(params, manifest, "ISMEAR", "0", reason="Force a conservative gapped-system smearing on a metallic task")
    if "SIGMA" in params:
        _update_key(params, manifest, "SIGMA", "0.05", reason="Use a small semiconductor-style Gaussian width")


def choose_task_corruption_family(metadata: dict[str, Any], reference_params: dict[str, str]) -> str:
    task_type = str(metadata.get("task_type") or "")
    if task_type == "static_scf":
        return "static_as_relax"
    if task_type == "geometry_relax":
        return "relax_as_static"
    if task_type == "line_mode_bands":
        return "line_mode_as_scf"
    if task_type == "dos_nscf":
        return "dos_as_scf"
    return "generic_required_drop"


def choose_material_corruption_family(metadata: dict[str, Any], reference_params: dict[str, str]) -> str:
    material_family = str(metadata.get("material_family") or "")

    if material_family in NONMAGNETIC_GAPPED_FAMILIES:
        return "smearing_confusion"
    if material_family in {"metal", "elemental"}:
        return "metal_as_gapped"
    if material_family == "magnetic_metal":
        return "spin_drop"
    if material_family in CORRELATED_FAMILIES:
        if any(key in reference_params for key in ("ISPIN", "MAGMOM")):
            return "spin_dftu_drop"
        return "dftu_drop"
    if material_family in LAYERED_FAMILIES:
        return "vdw_drop"
    if material_family == "heavy_element_semiconductor":
        return "soc_drop"
    if material_family == "complex_low_symmetry":
        return "symmetry_enable"
    return "generic_required_drop"


def control_manifest_for_case(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": utc_now(),
        "source_case_id": metadata["case_id"],
        "corruption_family": "control_clean_draft",
        "variant_type": "control",
        "target_keys": [],
        "mutations": [],
        "summary": {
            "num_target_keys": 0,
            "num_mutations": 0,
        },
    }


def apply_corruption_family(
    corruption_family: str,
    *,
    metadata: dict[str, Any],
    reference_params: dict[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    params = dict(reference_params)
    manifest: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "source_case_id": metadata["case_id"],
        "corruption_family": corruption_family,
        "target_keys": [],
        "mutations": [],
    }

    if corruption_family == "smearing_confusion":
        _smearing_confusion(params, metadata, manifest)
    elif corruption_family == "metal_as_gapped":
        _metal_as_gapped(params, metadata, manifest)
    elif corruption_family == "static_as_relax":
        _static_as_relax(params, metadata, manifest)
    elif corruption_family == "relax_as_static":
        _relax_as_static(params, metadata, manifest)
    elif corruption_family in {"line_mode_as_scf", "dos_as_scf", "nscf_as_scf"}:
        _nscf_as_scf(params, metadata, manifest)
    elif corruption_family == "dftu_drop":
        _dftu_drop(params, metadata, manifest)
    elif corruption_family == "spin_drop":
        _spin_drop(params, metadata, manifest)
    elif corruption_family == "spin_dftu_drop":
        _spin_dftu_drop(params, metadata, manifest)
    elif corruption_family == "soc_drop":
        _soc_drop(params, metadata, manifest)
    elif corruption_family == "vdw_drop":
        _vdw_drop(params, metadata, manifest)
    elif corruption_family == "symmetry_enable":
        _symmetry_enable(params, metadata, manifest)
    else:
        _generic_required_drop(params, metadata, manifest)

    if not manifest["target_keys"] and corruption_family != "generic_required_drop":
        return apply_corruption_family(
            "generic_required_drop",
            metadata=metadata,
            reference_params=reference_params,
        )

    manifest["summary"] = {
        "num_target_keys": len(manifest["target_keys"]),
        "num_mutations": len(manifest["mutations"]),
    }
    return params, manifest


def build_repair_variants(
    *,
    metadata: dict[str, Any],
    reference_params: dict[str, str],
) -> list[tuple[str, dict[str, str], dict[str, Any]]]:
    variants: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    task_family = choose_task_corruption_family(metadata, reference_params)
    task_params, task_manifest = apply_corruption_family(
        task_family,
        metadata=metadata,
        reference_params=reference_params,
    )
    task_manifest["variant_type"] = "task_driven"
    variants.append(("task_driven", task_params, task_manifest))

    material_family = choose_material_corruption_family(metadata, reference_params)
    material_params, material_manifest = apply_corruption_family(
        material_family,
        metadata=metadata,
        reference_params=reference_params,
    )
    material_manifest["variant_type"] = "material_driven"
    variants.append(("material_driven", material_params, material_manifest))

    control_manifest = control_manifest_for_case(metadata)
    variants.append(("control", dict(reference_params), control_manifest))
    return variants


def source_case_ids_from_root(source_root: Path, requested_case_ids: list[str] | None = None) -> list[str]:
    if requested_case_ids:
        return requested_case_ids
    case_root = source_root / "cases"
    return sorted(path.name for path in case_root.iterdir() if path.is_dir())


def repair_case_id_for(source_case_id: str, variant_type: str, corruption_family: str) -> str:
    return f"{source_case_id}__repair__{variant_type}__{_slugify(corruption_family)}"


def repair_case_dirs(benchmark_root: Path, requested_case_ids: list[str] | None = None) -> list[Path]:
    requested = set(requested_case_ids or [])
    index_path = benchmark_root / "metadata_index.json"
    if index_path.exists():
        payload = load_json(index_path)
        case_ids = [case["case_id"] for case in payload.get("cases", [])]
        if requested:
            case_ids = [case_id for case_id in case_ids if case_id in requested]
        return [benchmark_root / "cases" / case_id for case_id in case_ids if (benchmark_root / "cases" / case_id).is_dir()]

    case_dirs = sorted(path for path in (benchmark_root / "cases").iterdir() if path.is_dir())
    if requested:
        case_dirs = [case_dir for case_dir in case_dirs if case_dir.name in requested]
    return case_dirs


def load_source_case(source_case_dir: Path) -> dict[str, Any]:
    metadata = load_json(source_case_dir / "metadata.json")
    scoring = load_json(source_case_dir / "scoring.json")
    reference_path = source_case_dir / "inputs" / "INCAR_reference"
    poscar_path = source_case_dir / "inputs" / "POSCAR"
    return {
        "case_dir": source_case_dir,
        "metadata": metadata,
        "scoring": scoring,
        "reference_path": reference_path,
        "poscar_path": poscar_path,
        "reference_params": parse_incar_file(reference_path),
        "reference_text": reference_path.read_text(encoding="utf-8"),
        "poscar_text": poscar_path.read_text(encoding="utf-8"),
    }


def build_repair_case_metadata(
    source_metadata: dict[str, Any],
    manifest: dict[str, Any],
    *,
    variant_type: str,
) -> dict[str, Any]:
    metadata = dict(source_metadata)
    metadata["benchmark_type"] = "incar_repair"
    metadata["source_case_id"] = source_metadata["case_id"]
    metadata["case_id"] = repair_case_id_for(
        source_metadata["case_id"],
        variant_type,
        manifest["corruption_family"],
    )
    metadata["repair_goal"] = "repair_broken_incar"
    metadata["repair_variant_type"] = variant_type
    metadata["corruption_family"] = manifest["corruption_family"]
    metadata["repair_manifest_summary"] = manifest["summary"]
    return metadata


def build_repair_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "0.1",
        "benchmark_type": "incar_repair",
        "total_cases": len(rows),
        "cases": rows,
    }


def build_repair_readme() -> str:
    return """# INCAR Repair Benchmark

This standalone benchmark evaluates whether an LLM can repair a deliberately corrupted VASP INCAR.

Each case contains:

- `POSCAR`
- `INCAR_reference`
- `INCAR_bad`
- `error_manifest.json`
- the same metadata/scoring tags used by the generation benchmark

The model sees the broken INCAR and must output a corrected final INCAR.
"""


def load_repair_prompt_template(template_path: Path | None = None) -> dict[str, Any]:
    path = template_path or DEFAULT_REPAIR_PROMPT_TEMPLATE_PATH
    return load_json(path)


def render_repair_user_prompt(template_text: str, values: dict[str, str]) -> str:
    rendered = template_text
    for key, value in values.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def repair_prompt_messages_for_case(case_dir: Path) -> list[dict[str, str]]:
    metadata = load_json(case_dir / "metadata.json")
    poscar_text = (case_dir / "inputs" / "POSCAR").read_text(encoding="utf-8").strip()
    bad_incar_text = (case_dir / "inputs" / "INCAR_bad").read_text(encoding="utf-8").strip()
    prompt_template = load_repair_prompt_template()
    system_prompt = prompt_template["system_prompt"]
    user_prompt = render_repair_user_prompt(
        prompt_template["user_template"],
        {
            "formula": str(metadata.get("formula") or ""),
            "task_type": str(metadata.get("task_type") or ""),
            "prompt_goal": str(metadata.get("prompt_goal") or ""),
            "material_description": str(metadata.get("material_description") or ""),
            "prompt_constraints": str(metadata.get("prompt_constraints") or ""),
            "prompt_context": str(metadata.get("prompt_context") or ""),
            "poscar": poscar_text,
            "incar_draft": bad_incar_text,
        },
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def repair_case_context(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "difficulty": metadata.get("difficulty"),
        "task_type": metadata.get("task_type"),
        "task_family": metadata.get("task_family"),
        "material_family": metadata.get("material_family"),
        "challenge_type": metadata.get("challenge_type"),
        "corruption_family": metadata.get("corruption_family"),
        "repair_variant_type": metadata.get("repair_variant_type"),
    }


def _integer_value(text: str | None) -> int | None:
    if text is None:
        return None
    stripped = str(text).strip()
    if not stripped or " " in stripped:
        return None
    try:
        value = float(stripped)
    except ValueError:
        return None
    rounded = round(value)
    if abs(value - rounded) > 1e-9:
        return None
    return int(rounded)


def _task_workflow_pass(task_type: str, candidate_params: dict[str, str]) -> bool:
    return not _task_workflow_failure_reasons(task_type, candidate_params)


def _task_workflow_failure_reasons(task_type: str, candidate_params: dict[str, str]) -> list[str]:
    ibrion = _integer_value(candidate_params.get("IBRION"))
    nsw = _integer_value(candidate_params.get("NSW"))
    icharg = _integer_value(candidate_params.get("ICHARG"))

    reasons: list[str] = []

    if task_type == "static_scf":
        if ibrion not in (None, -1):
            reasons.append("workflow:IBRION")
        if nsw not in (None, 0):
            reasons.append("workflow:NSW")
        return reasons
    if task_type == "geometry_relax":
        if ibrion in (None, -1):
            reasons.append("workflow:IBRION")
        if nsw is None or nsw <= 0:
            reasons.append("workflow:NSW")
        return reasons
    if task_type in {"line_mode_bands", "dos_nscf"}:
        if icharg != 11:
            reasons.append("workflow:ICHARG")
        if ibrion not in (None, -1):
            reasons.append("workflow:IBRION")
        if nsw not in (None, 0):
            reasons.append("workflow:NSW")
        return reasons
    return [f"workflow:UNKNOWN_TASK:{task_type or 'missing'}"]


def minimum_task_runnable_assessment(
    *,
    metadata: dict[str, Any],
    scoring: dict[str, Any],
    reference_params: dict[str, str],
    candidate_params: dict[str, str],
) -> dict[str, Any]:
    task_type = str(metadata.get("task_type") or "")
    reasons = _task_workflow_failure_reasons(task_type, candidate_params)

    semantic_keys = [canonicalize_key(key) for key in scoring.get("semantic_must_match_keys", [])]
    policy_keys = [canonicalize_key(key) for key in scoring.get("policy_match_keys", [])]

    for key in semantic_keys:
        if reference_params.get(key) != candidate_params.get(key):
            reasons.append(f"semantic:{key}")

    for key in policy_keys:
        if _is_runnable_policy_key(key, reference_params) and not runnable_policy_matches(key, reference_params, candidate_params):
            reasons.append(f"policy:{key}")

    deduped = list(dict.fromkeys(reasons))
    return {"passed": not deduped, "reasons": deduped}


def minimum_task_runnable_case(
    *,
    metadata: dict[str, Any],
    scoring: dict[str, Any],
    reference_params: dict[str, str],
    candidate_params: dict[str, str],
) -> bool:
    return minimum_task_runnable_assessment(
        metadata=metadata,
        scoring=scoring,
        reference_params=reference_params,
        candidate_params=candidate_params,
    )["passed"]


def score_repaired_incar(
    *,
    case_dir: Path,
    candidate_path: Path,
    model_name: str,
) -> dict[str, Any]:
    metadata = load_json(case_dir / "metadata.json")
    scoring = load_json(case_dir / "scoring.json")
    error_manifest = load_json(case_dir / "inputs" / "error_manifest.json")
    reference_params = parse_incar_file(case_dir / "inputs" / "INCAR_reference")
    bad_params = parse_incar_file(case_dir / "inputs" / "INCAR_bad")
    candidate_params = parse_incar_file(candidate_path)

    base_grade = score_generated_incar(
        case_dir=case_dir,
        candidate_path=candidate_path,
        model_name=model_name,
    )
    effective_candidate_params = dict(candidate_params)
    effective_candidate_params.update(base_grade.get("default_imputed_keys", {}))

    target_keys = [canonicalize_key(key) for key in error_manifest.get("target_keys", [])]
    ignore_keys = {canonicalize_key(key) for key in scoring.get("ignore_keys", [])}
    reference_keys = {key for key in reference_params if key not in ignore_keys}
    preserve_keys = sorted(key for key in reference_keys if key not in set(target_keys))

    target_fixed = sum(1 for key in target_keys if effective_candidate_params.get(key) == reference_params.get(key))
    unchanged_bad = sum(1 for key in target_keys if effective_candidate_params.get(key) == bad_params.get(key))
    preserved = sum(1 for key in preserve_keys if effective_candidate_params.get(key) == reference_params.get(key))
    over_edited_keys = sorted(
        key for key in preserve_keys if effective_candidate_params.get(key) != reference_params.get(key)
    )
    target_parameter_outcomes = []
    fixed_keys: list[str] = []
    still_wrong_keys: list[str] = []
    unchanged_bad_keys: list[str] = []
    for key in target_keys:
        broken_value = bad_params.get(key)
        expected_value = reference_params.get(key)
        repaired_value = effective_candidate_params.get(key)
        if repaired_value == expected_value:
            status = "fixed"
            fixed_keys.append(key)
        elif repaired_value == broken_value:
            status = "unchanged_bad"
            unchanged_bad_keys.append(key)
            still_wrong_keys.append(key)
        else:
            status = "still_wrong"
            still_wrong_keys.append(key)
        target_parameter_outcomes.append(
            {
                "parameter": key,
                "broken_value": broken_value,
                "expected_value": expected_value,
                "repaired_value": repaired_value,
                "status": status,
            }
        )

    preservation_violations = [
        {
            "parameter": key,
            "expected_value": reference_params.get(key),
            "repaired_value": effective_candidate_params.get(key),
        }
        for key in over_edited_keys
    ]

    target_fix_rate = 100.0 if not target_keys else 100.0 * target_fixed / len(target_keys)
    preservation_rate = 100.0 if not preserve_keys else 100.0 * preserved / len(preserve_keys)
    format_cleanliness = float(base_grade["score_breakdown"]["extra_keys"])
    repair_total = round((0.50 * target_fix_rate + 0.50 * preservation_rate) * 0.95 + 0.05 * format_cleanliness, 2)
    semantic_regression_keys = sorted(base_grade.get("missing_required_keys", {}).get("semantic", []) or [])
    policy_regression_keys = sorted(
        item.get("parameter")
        for item in (base_grade.get("policy_match", {}) or {}).get("items", [])
        if not item.get("matched") and item.get("parameter")
    )
    repair_regression_summary = {
        "semantic_regression_keys": semantic_regression_keys,
        "policy_regression_keys": policy_regression_keys,
        "regressed": bool(
            semantic_regression_keys
            or policy_regression_keys
        ),
    }
    runnable_assessment = minimum_task_runnable_assessment(
        metadata=metadata,
        scoring=scoring,
        reference_params=reference_params,
        candidate_params=effective_candidate_params,
    )
    minimum_task_runnable = bool(runnable_assessment["passed"])

    base_grade["case_context"] = repair_case_context(metadata)
    base_grade["source_case_id"] = metadata.get("source_case_id")
    base_grade["repair_score_breakdown"] = {
        "target_fix_rate": round(target_fix_rate, 2),
        "preservation_rate": round(preservation_rate, 2),
        "format_cleanliness": round(format_cleanliness, 2),
        "repair_total": repair_total,
    }
    base_grade["repair_score_weights"] = {
        "target_fix_rate": 0.475,
        "preservation_rate": 0.475,
        "format_cleanliness": 0.05,
    }
    base_grade["repair_diagnostics"] = {
        "corruption_family": metadata.get("corruption_family"),
        "target_keys": target_keys,
        "target_total": len(target_keys),
        "target_fixed": target_fixed,
        "unchanged_bad_key_count": unchanged_bad,
        "preserve_total": len(preserve_keys),
        "preserved": preserved,
        "over_edited_keys": over_edited_keys,
    }
    base_grade["repair_parameter_outcomes"] = target_parameter_outcomes
    base_grade["repair_result_summary"] = {
        "corruption_family": metadata.get("corruption_family"),
        "target_keys": target_keys,
        "fixed_keys": fixed_keys,
        "still_wrong_keys": still_wrong_keys,
        "unchanged_bad_keys": unchanged_bad_keys,
        "over_edited_keys": over_edited_keys,
    }
    base_grade["preservation_violations"] = preservation_violations
    base_grade["repair_regression_summary"] = repair_regression_summary
    base_grade["minimum_task_runnable"] = minimum_task_runnable
    base_grade["minimum_task_runnable_reasons"] = runnable_assessment["reasons"]
    base_grade["perfect_repair"] = (
        target_fix_rate == 100.0
        and preservation_rate == 100.0
        and unchanged_bad == 0
        and not base_grade.get("extra_keys")
    )
    return base_grade


def missing_repair_grade(case_dir: Path, model_name: str, candidate_path: Path) -> dict[str, Any]:
    metadata = load_json(case_dir / "metadata.json")
    error_manifest = load_json(case_dir / "inputs" / "error_manifest.json")
    target_keys = [canonicalize_key(key) for key in error_manifest.get("target_keys", [])]
    target_parameter_outcomes = [
        {
            "parameter": key,
            "broken_value": (error_manifest.get("corrupted_snapshot") or {}).get(key),
            "expected_value": (error_manifest.get("source_reference_snapshot") or {}).get(key),
            "repaired_value": None,
            "status": "missing_candidate",
        }
        for key in target_keys
    ]
    payload = missing_generation_grade(case_dir=case_dir, model_name=model_name, candidate_path=candidate_path)
    payload["case_context"] = repair_case_context(metadata)
    payload["source_case_id"] = metadata.get("source_case_id")
    payload["repair_result_summary"] = {
        "corruption_family": metadata.get("corruption_family"),
        "target_keys": target_keys,
        "fixed_keys": [],
        "still_wrong_keys": target_keys,
        "unchanged_bad_keys": target_keys,
        "over_edited_keys": [],
    }
    payload["repair_parameter_outcomes"] = target_parameter_outcomes
    payload["preservation_violations"] = []
    payload["repair_regression_summary"] = {
        "semantic_regression_keys": [],
        "policy_regression_keys": [],
        "regressed": None,
    }
    payload["minimum_task_runnable"] = False
    payload["minimum_task_runnable_reasons"] = ["missing_candidate"]
    return payload


def summarize_repair_grade_subset(grades: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [grade for grade in grades if grade.get("status") == "graded"]

    def average(key: str) -> float | None:
        if not graded:
            return None
        return round(sum(float(grade["repair_score_breakdown"][key]) for grade in graded) / len(graded), 2)

    perfect_repairs = sum(1 for grade in graded if grade.get("perfect_repair"))
    return {
        "total_cases": len(grades),
        "graded_cases": len(graded),
        "missing_cases": len(grades) - len(graded),
        "average_scores": {
            "target_fix_rate": average("target_fix_rate"),
            "preservation_rate": average("preservation_rate"),
            "format_cleanliness": average("format_cleanliness"),
            "repair_total": average("repair_total"),
        },
        "perfect_repairs": perfect_repairs,
        "perfect_repair_rate": round(perfect_repairs / len(graded), 4) if graded else None,
        "case_ids": [grade["case_id"] for grade in grades],
    }


def _group_sort_key(value: str) -> tuple[int, str]:
    if value.startswith("L") and value[1:].isdigit():
        return (0, f"{int(value[1:]):04d}")
    preferred_order = {
        "static_scf": 0,
        "geometry_relax": 1,
        "line_mode_bands": 2,
        "dos_nscf": 3,
    }
    if value in preferred_order:
        return (1, f"{preferred_order[value]:04d}")
    return (2, value)


def grouped_repair_summaries(grades: list[dict[str, Any]], *, group_key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for grade in grades:
        case_context = grade.get("case_context") or {}
        group_value = case_context.get(group_key) or "unknown"
        grouped.setdefault(str(group_value), []).append(grade)
    return {
        group_value: summarize_repair_grade_subset(grouped[group_value])
        for group_value in sorted(grouped, key=_group_sort_key)
    }


def summarize_repair_grades(
    *,
    benchmark_root: Path,
    model_name: str,
    grades: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_core = summarize_repair_grade_subset(grades)
    return {
        "generated_at_utc": utc_now(),
        "benchmark_root": str(benchmark_root),
        "model_name": model_name,
        **summary_core,
        "by_difficulty": grouped_repair_summaries(grades, group_key="difficulty"),
        "by_task_type": grouped_repair_summaries(grades, group_key="task_type"),
        "by_task_family": grouped_repair_summaries(grades, group_key="task_family"),
        "by_material_family": grouped_repair_summaries(grades, group_key="material_family"),
        "by_challenge_type": grouped_repair_summaries(grades, group_key="challenge_type"),
        "by_corruption_family": grouped_repair_summaries(grades, group_key="corruption_family"),
        "cases": [
            {
                "case_id": grade["case_id"],
                "source_case_id": grade.get("source_case_id"),
                "status": grade["status"],
                "case_context": grade.get("case_context", {}),
                "repair_scores": grade.get("repair_score_breakdown", {}),
                "alignment_scores": grade.get("score_breakdown", {}),
                "repair_result_summary": grade.get("repair_result_summary", {}),
                "repair_regression_summary": grade.get("repair_regression_summary", {}),
                "minimum_task_runnable": grade.get("minimum_task_runnable"),
                "minimum_task_runnable_reasons": grade.get("minimum_task_runnable_reasons", []),
                "perfect_repair": grade.get("perfect_repair"),
            }
            for grade in grades
        ],
    }


def load_repair_summaries(leaderboards_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(leaderboards_dir.glob("*_summary.json")):
        payload = load_json(path)
        payload["_path"] = str(path)
        summaries.append(payload)
    if not summaries:
        raise SystemExit(f"No summary JSON files found under {leaderboards_dir}")
    return summaries


def model_sort_key(summary: dict[str, Any]) -> tuple[float, float, str]:
    average_scores = summary.get("average_scores", {})
    repair_total = average_scores.get("repair_total")
    target_fix_rate = average_scores.get("target_fix_rate")
    repair_score = float(repair_total) if repair_total is not None else float("-inf")
    fix_score = float(target_fix_rate) if target_fix_rate is not None else float("-inf")
    return (-repair_score, -fix_score, str(summary.get("model_name", "")))


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
    return "".join(chars).strip("-") or "section"


def repair_report_markdown(summaries: list[dict[str, Any]], benchmark_root: Path) -> str:
    REPORT_DIMENSIONS = (
        ("corruption_family", "By Corruption Family"),
        ("task_type", "By Task Type"),
        ("material_family", "By Material Family"),
    )

    def case_rows_with_variant(summary: dict[str, Any], variant_type: str) -> list[dict[str, Any]]:
        return [
            case
            for case in summary.get("cases", [])
            if case.get("case_context", {}).get("repair_variant_type") == variant_type
        ]

    def error_case_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            case
            for case in summary.get("cases", [])
            if case.get("case_context", {}).get("repair_variant_type") != "control"
        ]

    def group_case_rows(summary: dict[str, Any], group_key: str, group_name: str) -> list[dict[str, Any]]:
        return [
            case
            for case in error_case_rows(summary)
            if str(case.get("case_context", {}).get(group_key) or "unknown") == group_name
        ]

    def average(values: list[float]) -> float | None:
        return (sum(values) / len(values)) if values else None

    def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        graded = [row for row in rows if row.get("status") == "graded"]

        def avg_score(metric: str) -> float | None:
            values = [row.get("repair_scores", {}).get(metric) for row in graded]
            values = [float(value) for value in values if value is not None]
            return average(values)

        target_counts = [len(row.get("repair_result_summary", {}).get("target_keys", []) or []) for row in graded]
        still_wrong_counts = [len(row.get("repair_result_summary", {}).get("still_wrong_keys", []) or []) for row in graded]
        over_edited_counts = [len(row.get("repair_result_summary", {}).get("over_edited_keys", []) or []) for row in graded]
        over_edit_cases = sum(1 for count in over_edited_counts if count > 0)
        perfect_cases = sum(1 for row in graded if row.get("perfect_repair"))
        runnable_cases = sum(1 for row in graded if row.get("minimum_task_runnable") is True)

        total_targets = sum(target_counts)
        total_still_wrong = sum(still_wrong_counts)

        return {
            "total_cases": len(rows),
            "graded_cases": len(graded),
            "target_fix_rate": avg_score("target_fix_rate"),
            "preservation_rate": avg_score("preservation_rate"),
            "format_cleanliness": avg_score("format_cleanliness"),
            "repair_total": avg_score("repair_total"),
            "perfect_rate": (perfect_cases / len(graded)) if graded else None,
            "runnable_rate": (runnable_cases / len(graded)) if graded else None,
            "over_edit_rate": (over_edit_cases / len(graded)) if graded else None,
            "avg_edit_count": average([float(count) for count in over_edited_counts]),
            "avg_over_edited_keys": average([float(count) for count in over_edited_counts]),
            "still_wrong_rate": ((total_still_wrong / total_targets) if total_targets else None),
        }

    def control_sort_key(summary: dict[str, Any]) -> tuple[float, float, float, str]:
        agg = aggregate_rows(case_rows_with_variant(summary, "control"))
        preservation = agg["preservation_rate"] if agg["preservation_rate"] is not None else float("-inf")
        runnable = agg["runnable_rate"] if agg["runnable_rate"] is not None else float("-inf")
        over_edit = agg["over_edit_rate"] if agg["over_edit_rate"] is not None else float("inf")
        perfect = agg["perfect_rate"] if agg["perfect_rate"] is not None else float("-inf")
        return (-float(preservation), -float(runnable), float(over_edit), -float(perfect), str(summary.get("model_name", "")))

    def error_sort_key(summary: dict[str, Any]) -> tuple[float, float, float, str]:
        agg = aggregate_rows(error_case_rows(summary))
        fix_rate = agg["target_fix_rate"] if agg["target_fix_rate"] is not None else float("-inf")
        runnable = agg["runnable_rate"] if agg["runnable_rate"] is not None else float("-inf")
        preservation = agg["preservation_rate"] if agg["preservation_rate"] is not None else float("-inf")
        return (-float(fix_rate), -float(runnable), -float(preservation), str(summary.get("model_name", "")))

    def group_sort_key(rows: list[dict[str, Any]]) -> tuple[float, float, str]:
        agg = aggregate_rows(rows)
        fix_rate = agg["target_fix_rate"] if agg["target_fix_rate"] is not None else float("-inf")
        preservation = agg["preservation_rate"] if agg["preservation_rate"] is not None else float("-inf")
        return (-float(fix_rate), -float(preservation), "")

    sections = ["# INCAR Repair Benchmark Report", ""]
    sections.append(f"- Generated at: `{utc_now()}`")
    sections.append(f"- Benchmark root: `{benchmark_root.resolve()}`")
    sections.append(f"- Models covered: `{len(summaries)}`")
    sections.append(
        "- This report keeps two primary leaderboards: one for error repair and one for control preservation."
    )
    sections.append("")
    sections.append("## Contents")
    sections.append("")
    sections.append("- [Control Group](#control-group)")
    sections.append("- [Error Group](#error-group)")
    sections.append("- [By Corruption Family](#by-corruption-family)")
    sections.append("- [By Task Type](#by-task-type)")
    sections.append("- [By Material Family](#by-material-family)")
    sections.append("- [Case Repair Details](#case-repair-details)")
    sections.append("")

    control_headers = [
        "Model",
        "Control Cases",
        "Graded",
        "Control Preservation Rate",
        "Control Over-Edit Rate",
        "Perfect Control Rate",
        "Minimum Task-Runnable Rate",
        "Avg Edit Count on Control",
        "Avg Cleanliness",
    ]
    control_rows = []
    for summary in sorted(summaries, key=control_sort_key):
        agg = aggregate_rows(case_rows_with_variant(summary, "control"))
        control_rows.append(
            [
                str(summary["model_name"]),
                str(agg["total_cases"]),
                str(agg["graded_cases"]),
                format_score(agg["preservation_rate"]),
                format_rate(agg["over_edit_rate"]),
                format_rate(agg["perfect_rate"]),
                format_rate(agg["runnable_rate"]),
                format_score(agg["avg_edit_count"]),
                format_score(agg["format_cleanliness"]),
            ]
        )
    sections.append("## Control Group")
    sections.append("")
    sections.append("This section measures whether models leave already-correct drafts alone while preserving minimum task-runnability.")
    sections.append("")
    sections.append(markdown_table(control_headers, control_rows))
    sections.append("")

    error_headers = [
        "Model",
        "Error Cases",
        "Graded",
        "Error Fix Rate",
        "Error Preservation Rate",
        "Perfect Error Repair Rate",
        "Minimum Task-Runnable Rate",
        "Avg Over-Edited Keys",
        "Avg Cleanliness",
    ]
    error_rows = []
    for summary in sorted(summaries, key=error_sort_key):
        agg = aggregate_rows(error_case_rows(summary))
        error_rows.append(
            [
                str(summary["model_name"]),
                str(agg["total_cases"]),
                str(agg["graded_cases"]),
                format_score(agg["target_fix_rate"]),
                format_score(agg["preservation_rate"]),
                format_rate(agg["perfect_rate"]),
                format_rate(agg["runnable_rate"]),
                format_score(agg["avg_over_edited_keys"]),
                format_score(agg["format_cleanliness"]),
            ]
        )
    sections.append("## Error Group")
    sections.append("")
    sections.append("This section measures whether models fix the intentionally injected errors without creating new ones.")
    sections.append("")
    sections.append(markdown_table(error_headers, error_rows))
    sections.append("")

    for group_key, title in REPORT_DIMENSIONS:
        model_names = [str(summary["model_name"]) for summary in sorted(summaries, key=error_sort_key)]
        group_names: set[str] = set()
        for summary in summaries:
            for case in error_case_rows(summary):
                group_names.add(str(case.get("case_context", {}).get(group_key) or "unknown"))

        rows: list[list[str]] = []
        for group_name in sorted(group_names, key=_group_sort_key):
            total_cases = 0
            best_fix = float("-inf")
            best_runnable = float("-inf")
            best_model = "-"
            row = [group_name]
            for summary in summaries:
                total_cases = max(total_cases, len(group_case_rows(summary, group_key, group_name)))
            row.append(str(total_cases))

            for summary in sorted(summaries, key=error_sort_key):
                agg = aggregate_rows(group_case_rows(summary, group_key, group_name))
                row.append(
                    " / ".join(
                        [
                            format_score(agg["target_fix_rate"]),
                            format_rate(agg["runnable_rate"]),
                        ]
                    )
                )
                fix_rate = agg["target_fix_rate"]
                runnable = agg["runnable_rate"]
                if fix_rate is not None:
                    candidate_key = (
                        float(fix_rate),
                        float(runnable if runnable is not None else float("-inf")),
                    )
                    best_key = (best_fix, best_runnable)
                    if candidate_key > best_key:
                        best_fix = float(fix_rate)
                        best_runnable = float(runnable if runnable is not None else float("-inf"))
                        best_model = str(summary["model_name"])
            row.append(f"{best_model} ({best_fix:.2f})" if best_model != "-" else "-")
            rows.append(row)

        sections.append(f"## {title}")
        sections.append("")
        sections.append("These grouped sections summarize error variants only.")
        sections.append("")
        sections.append("Cell format: `Error Fix Rate / Minimum Task-Runnable Rate`.")
        sections.append("")
        sections.append(markdown_table(["Group", "Cases", *model_names, "Best Model"], rows))
        sections.append("")

    sections.append("## Case Repair Details")
    sections.append("")
    sections.append(
        "Each row shows which parameters were intentionally broken, which were fixed, which remain wrong, and which correct parameters were over-edited."
    )
    sections.append("")
    for summary in sorted(summaries, key=error_sort_key):
        sections.append(f"### {summary['model_name']}")
        sections.append("")
        headers = [
            "Case",
            "Variant",
            "Corruption",
            "Minimum Runnable",
            "Runnable Failure Reasons",
            "Broken",
            "Fixed",
            "Still Wrong",
            "Over-Edited",
            "Regressed",
            "Semantic Regr",
            "Policy Regr",
            "Target Fix",
            "Preservation",
            "Cleanliness",
            "Repair Total",
        ]
        rows = []
        for case in summary.get("cases", []):
            repair_summary = case.get("repair_result_summary", {})
            regression_summary = case.get("repair_regression_summary", {})
            rows.append(
                [
                    str(case["case_id"]),
                    str(case.get("case_context", {}).get("repair_variant_type") or "-"),
                    str(repair_summary.get("corruption_family") or case.get("case_context", {}).get("corruption_family") or "-"),
                    str(case.get("minimum_task_runnable", "-")),
                    ", ".join(case.get("minimum_task_runnable_reasons", []) or []) or "-",
                    ", ".join(repair_summary.get("target_keys", [])) or "-",
                    ", ".join(repair_summary.get("fixed_keys", [])) or "-",
                    ", ".join(repair_summary.get("still_wrong_keys", [])) or "-",
                    ", ".join(repair_summary.get("over_edited_keys", [])) or "-",
                    str(regression_summary.get("regressed", "-")),
                    ", ".join(regression_summary.get("semantic_regression_keys", [])) or "-",
                    ", ".join(regression_summary.get("policy_regression_keys", [])) or "-",
                    format_score(case.get("repair_scores", {}).get("target_fix_rate")),
                    format_score(case.get("repair_scores", {}).get("preservation_rate")),
                    format_score(case.get("repair_scores", {}).get("format_cleanliness")),
                    format_score(case.get("repair_scores", {}).get("repair_total")),
                ]
            )
        sections.append(markdown_table(headers, rows))
        sections.append("")

    return "\n".join(sections).rstrip() + "\n"


__all__ = [
    "DEFAULT_REPAIR_OUTPUT_ROOT",
    "DEFAULT_SOURCE_BENCHMARK_ROOT",
    "DEFAULT_REPAIR_REPORT_NAME",
    "REPAIR_DIMENSION_SPECS",
    "MVP_SOURCE_CASE_IDS",
    "build_corrupted_incar",
    "build_repair_case_metadata",
    "build_repair_index",
    "build_repair_readme",
    "dump_json",
    "enabled_models",
    "ensure_output_root",
    "extract_incar_from_response",
    "incar_text_from_dict",
    "invoke_model",
    "load_json",
    "load_repair_summaries",
    "load_source_case",
    "missing_repair_grade",
    "mvp_source_case_ids",
    "repair_case_context",
    "repair_case_dirs",
    "repair_case_id_for",
    "repair_prompt_messages_for_case",
    "repair_report_markdown",
    "score_repaired_incar",
    "summarize_repair_grades",
]
