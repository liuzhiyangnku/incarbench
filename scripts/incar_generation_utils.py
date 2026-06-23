from __future__ import annotations

import csv
import html
import json
import os
import re
import subprocess
import sys
import time
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib import error, request

from benchmark_utils import dump_json, load_json, utc_now
from llm_config_utils import load_llm_benchmark_config

CSV_LIST_SEPARATOR = "|"
IGNORED_SCORE_KEYS = {"SYSTEM"}
DEFAULT_OUTPUT_ROOT = Path("incar_generation_benchmark")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
UNSET = "__UNSET__"
COMMENT_MARKERS = ("!",)
NUMERIC_TOKEN_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?$")
MP_MAX_RETRIES = 4
MP_RETRY_DELAY_SECONDS = 3
MP_MAX_RETRY_DELAY_SECONDS = 30
DEFAULT_LOCAL_PROXY_URL = "http://127.0.0.1:7890"
GCLOUD_TOKEN_TTL_SECONDS = 3000
ENUM_NORMALIZATION = {
    "PREC": {
        "LOW": "Low",
        "MEDIUM": "Medium",
        "NORMAL": "Normal",
        "ACCURATE": "Accurate",
        "SINGLE": "Single",
        "HIGH": "High",
        "ACCURA": "Accurate",
    },
    "ALGO": {
        "NORMAL": "Normal",
        "FAST": "Fast",
        "VERYFAST": "VeryFast",
        "ALL": "All",
        "DAMPED": "Damped",
    },
    "LREAL": {
        ".TRUE.": ".TRUE.",
        ".FALSE.": ".FALSE.",
        "TRUE": ".TRUE.",
        "FALSE": ".FALSE.",
        "AUTO": "Auto",
        "A": "Auto",
        "ON": ".TRUE.",
        "OFF": ".FALSE.",
    },
}

INCAR_ORDER = [
    "PREC",
    "ENCUT",
    "EDIFF",
    "EDIFFG",
    "ISMEAR",
    "SIGMA",
    "ISPIN",
    "MAGMOM",
    "IBRION",
    "NSW",
    "ISIF",
    "NELM",
    "LREAL",
    "ALGO",
    "LASPH",
    "LMAXMIX",
    "ISYM",
    "ISTART",
    "ICHARG",
    "LDAU",
    "LDAUTYPE",
    "LDAUL",
    "LDAUU",
    "LDAUJ",
    "METAGGA",
    "GGA",
    "ADDGRID",
    "LWAVE",
    "LCHARG",
]

STATIC_DEFAULT_KEYS = {
    "PREC",
    "ENCUT",
    "EDIFF",
    "ISMEAR",
    "SIGMA",
    "IBRION",
    "NSW",
    "ISIF",
    "NELM",
    "LREAL",
    "ALGO",
    "ISYM",
}

STATIC_MAGNETIC_KEYS = STATIC_DEFAULT_KEYS | {
    "ISPIN",
    "MAGMOM",
}

RELAX_DEFAULT_KEYS = {
    "PREC",
    "ENCUT",
    "EDIFF",
    "EDIFFG",
    "ISMEAR",
    "SIGMA",
    "IBRION",
    "NSW",
    "ISIF",
    "NELM",
    "LREAL",
    "ALGO",
}

BAND_STRUCTURE_KEYS = {
    "PREC",
    "ENCUT",
    "EDIFF",
    "ISMEAR",
    "SIGMA",
    "IBRION",
    "NSW",
    "NELM",
    "LREAL",
    "ALGO",
    "ICHARG",
    "ISYM",
    "NBANDS",
}

DOS_NSCF_KEYS = {
    "PREC",
    "ENCUT",
    "EDIFF",
    "ISMEAR",
    "SIGMA",
    "IBRION",
    "NSW",
    "NELM",
    "LREAL",
    "ALGO",
    "ICHARG",
    "ISYM",
    "NEDOS",
}

DEFAULT_POLICY_RULES: dict[str, dict[str, Any]] = {
    "ENCUT": {
        "type": "directional_numeric",
        "allow_higher_rel": 0.2,
        "allow_lower_rel": 0.05,
    },
    "EDIFF": {
        "type": "strictness_numeric",
        "lower_is_stricter": True,
        "max_looser_ratio": 2.0,
    },
    "EDIFFG": {
        "type": "signed_ratio",
        "sign_must_match": True,
        "max_ratio": 1.5,
    },
    "NSW": {
        "type": "abs_tolerance",
        "max_abs_diff": 5.0,
    },
    "NELM": {
        "type": "abs_tolerance",
        "max_abs_diff": 20.0,
    },
    "SIGMA": {
        "type": "abs_tolerance",
        "max_abs_diff": 0.05,
    },
    "NBANDS": {
        "type": "directional_numeric",
        "allow_higher_rel": 1.0,
        "allow_lower_rel": 0.0,
    },
}

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

_GCLOUD_TOKEN_CACHE: dict[str, Any] = {"token": None, "acquired_at": 0.0}

RUNNABLE_POLICY_KEYS = {
    "LDAU",
    "LDAUTYPE",
    "LDAUL",
    "LDAUU",
    "LDAUJ",
    "IVDW",
    "LSORBIT",
}


def canonicalize_key(raw: str) -> str:
    return raw.strip().upper()


def _integer_value(raw: str | None) -> int | None:
    if raw is None:
        return None
    token = str(raw).strip()
    if not token or " " in token:
        return None
    try:
        return int(float(token))
    except ValueError:
        return None


def is_ismear_policy_consistent(expected: str | None, observed: str | None) -> bool:
    expected_value = _integer_value(expected)
    observed_value = _integer_value(observed)
    if expected_value is None or observed_value is None:
        return False
    if expected_value >= 0:
        return observed_value >= 0
    return observed_value <= 0


def runnable_policy_matches(key: str, reference_params: dict[str, str], candidate_params: dict[str, str]) -> bool:
    expected = reference_params.get(key)
    observed = candidate_params.get(key)
    if key == "ISMEAR":
        return is_ismear_policy_consistent(expected, observed)
    return expected == observed


def _is_runnable_policy_key(key: str, reference_params: dict[str, str]) -> bool:
    if key == "ISMEAR":
        return _integer_value(reference_params.get("ISMEAR")) is not None
    return key in RUNNABLE_POLICY_KEYS


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


def canonicalize_value(raw: Any) -> str:
    if raw is None:
        return UNSET

    value = str(raw).strip()
    if not value:
        return UNSET

    value = re.sub(r"\s+", " ", value)

    def normalize_token(token: str) -> str:
        upper = token.upper()
        if upper in {".TRUE.", "TRUE", "T", "YES", "ON"}:
            return ".TRUE."
        if upper in {".FALSE.", "FALSE", "F", "NO", "OFF"}:
            return ".FALSE."
        if NUMERIC_TOKEN_RE.match(token):
            try:
                number = Decimal(token)
                if number == number.to_integral():
                    return str(number.quantize(Decimal("1")))
                return format(number.normalize(), "g")
            except InvalidOperation:
                pass
        return upper

    return " ".join(normalize_token(token) for token in value.split(" "))


def strip_inline_comment(line: str) -> str:
    stripped = line
    for marker in COMMENT_MARKERS:
        if marker in stripped:
            stripped = stripped.split(marker, 1)[0]
    return stripped.strip()


def parse_incar_text(text: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = strip_inline_comment(raw_line)
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        params[canonicalize_key(key)] = canonicalize_value(value)
    return params


def parse_incar_file(path: Path) -> dict[str, str]:
    return parse_incar_text(path.read_text(encoding="utf-8"))

NORMALIZATION_PROFILES: dict[str, dict[str, Any]] = {
    "static_default": {
        "keep": STATIC_DEFAULT_KEYS,
        "defaults": {"IBRION": "-1", "NSW": "0", "ISIF": "2"},
    },
    "static_magnetic": {
        "keep": STATIC_MAGNETIC_KEYS,
        "defaults": {"IBRION": "-1", "NSW": "0", "ISIF": "2", "ISPIN": "2"},
    },
    "relax_default": {
        "keep": RELAX_DEFAULT_KEYS,
        "defaults": {"IBRION": "2", "NSW": "99", "EDIFFG": "-0.02"},
    },
    "band_structure": {
        "keep": BAND_STRUCTURE_KEYS,
        "defaults": {"IBRION": "-1", "NSW": "0", "ICHARG": "11", "ISYM": "0"},
    },
    "dos_nscf": {
        "keep": DOS_NSCF_KEYS,
        "defaults": {"IBRION": "-1", "NSW": "0", "ICHARG": "11", "ISYM": "0"},
    },
}


def split_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(CSV_LIST_SEPARATOR) if item.strip()]


def parse_json_object(raw: str | None) -> dict[str, Any]:
    if raw is None or not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")
    return payload


def parse_key_value_overrides(raw: str | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in split_csv_list(raw):
        if "=" not in item:
            raise ValueError(f"Invalid override item: {item!r}")
        key, value = item.split("=", 1)
        overrides[canonicalize_key(key)] = value.strip()
    return overrides


def row_mentioned_scoring_keys(row: dict[str, str]) -> set[str]:
    mentioned: set[str] = set()
    for field in (
        "semantic_must_match_keys",
        "policy_match_keys",
        "optional_match_keys",
        "must_match_keys",
    ):
        mentioned |= {canonicalize_key(value) for value in split_csv_list(row.get(field))}
    return mentioned


def mentioned_scoring_keys_union(rows: list[dict[str, str]]) -> set[str]:
    mentioned: set[str] = set()
    for row in rows:
        mentioned.update(row_mentioned_scoring_keys(row))
    return mentioned


DEFAULTS_DOC_PATH = REPO_ROOT / "docs" / "experimental" / "vasp_incar_defaults_from_wiki.md"


@lru_cache(maxsize=1)
def load_incar_default_texts() -> dict[str, str]:
    if not DEFAULTS_DOC_PATH.exists():
        return {}

    defaults: dict[str, str] = {}
    current_key: str | None = None
    for line in DEFAULTS_DOC_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            current_key = canonicalize_key(stripped[4:])
            continue
        if current_key and stripped.startswith("- 默认值："):
            defaults[current_key] = stripped.split("：", 1)[1].strip()
            current_key = None
    return defaults


def _clean_default_text(text: str) -> str:
    cleaned = html.unescape(text)
    cleaned = cleaned.replace("[math]", "")
    cleaned = cleaned.replace("[/math]", "")
    cleaned = cleaned.replace("\\displaystyle", "")
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_simple_default_value(text: str) -> str | None:
    cleaned = _clean_default_text(text)
    if " if " in cleaned.lower() or " else" in cleaned.lower() or " for " in cleaned.lower():
        return None
    if "=" not in cleaned:
        return None
    value = cleaned.split("=", 1)[1].strip()
    value = re.sub(r"\([^)]*\)", "", value).strip()
    if value == "10^-4":
        return "0.0001"
    if NUMERIC_TOKEN_RE.match(value) or value in {".TRUE.", ".FALSE."}:
        return value
    return value if re.fullmatch(r"[A-Za-z][A-Za-z0-9.]*", value) else None


def resolve_candidate_default_value(
    *,
    key: str,
    candidate_params: dict[str, str],
    raw_seed_params: dict[str, Any],
) -> str | None:
    key = canonicalize_key(key)
    default_text = load_incar_default_texts().get(key)
    if default_text is None:
        return None

    simple = _extract_simple_default_value(default_text)
    if simple is not None:
        return canonicalize_value(simple)

    if key == "ENCUT":
        for fallback_key in ("ENMAX", "ENINI", "ENCUTGW"):
            if fallback_key in raw_seed_params and raw_seed_params[fallback_key] is not None:
                return canonicalize_value(format_incar_value(raw_seed_params[fallback_key]))
        return None

    if key == "EDIFFG":
        ediff = candidate_params.get("EDIFF") or resolve_candidate_default_value(
            key="EDIFF",
            candidate_params=candidate_params,
            raw_seed_params=raw_seed_params,
        )
        if ediff is None:
            return None
        try:
            return canonicalize_value(str(float(ediff) * 10.0))
        except ValueError:
            return None

    if key == "IBRION":
        nsw = candidate_params.get("NSW")
        if nsw is None:
            return None
        try:
            nsw_value = int(float(nsw))
        except ValueError:
            return None
        return canonicalize_value("-1" if nsw_value in {-1, 0} else "0")

    if key == "ICHARG":
        istart = candidate_params.get("ISTART")
        if istart is None:
            return canonicalize_value("2")
        try:
            istart_value = int(float(istart))
        except ValueError:
            return None
        return canonicalize_value("2" if istart_value == 0 else "0")

    if key == "ISIF":
        ibrion = candidate_params.get("IBRION")
        if ibrion is None:
            return canonicalize_value("2")
        try:
            ibrion_value = int(float(ibrion))
        except ValueError:
            return None
        return canonicalize_value("0" if ibrion_value == 0 else "2")

    if key == "ISYM":
        return canonicalize_value("2")

    return None


def apply_candidate_defaults_for_scoring(
    *,
    reference_params: dict[str, str],
    candidate_params: dict[str, str],
    raw_seed_params: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    effective = dict(candidate_params)
    imputed: dict[str, str] = {}

    changed = True
    while changed:
        changed = False
        for key in reference_params:
            if key in effective:
                continue
            default_value = resolve_candidate_default_value(
                key=key,
                candidate_params=effective,
                raw_seed_params=raw_seed_params,
            )
            if default_value is None:
                continue
            effective[key] = default_value
            imputed[key] = default_value
            changed = True

    return effective, imputed


def load_problem_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            cleaned: dict[str, str] = {}
            for key, value in row.items():
                if key is None:
                    continue
                if isinstance(value, list):
                    value = " ".join(str(item) for item in value if item is not None)
                cleaned[key] = (value or "").strip()
            if not cleaned.get("case_id"):
                continue
            rows.append(cleaned)
        return rows


def row_source_kind(row: dict[str, str]) -> str:
    return (row.get("source_kind") or "mp").strip().lower()


def resolve_input_path(path_str: str, *, csv_path: Path) -> Path:
    raw = Path(path_str)
    if raw.is_absolute():
        return raw
    return (csv_path.parent / raw).resolve()


def to_poscar_string(structure: Any) -> str:
    try:
        from pymatgen.io.vasp import Poscar
    except ImportError as exc:
        raise RuntimeError("pymatgen is required to write POSCAR files") from exc

    return str(Poscar(structure))


def format_incar_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, (list, tuple)):
        return " ".join(format_incar_value(item) for item in value)
    return str(value)


def normalize_value_for_key(key: str, value: str) -> str:
    key = canonicalize_key(key)
    raw_value = value.strip()
    upper_value = raw_value.upper()

    if key in ENUM_NORMALIZATION and upper_value in ENUM_NORMALIZATION[key]:
        return ENUM_NORMALIZATION[key][upper_value]

    if upper_value in {".TRUE.", ".FALSE."}:
        return upper_value

    return raw_value


def normalize_incar_dict(
    raw_params: dict[str, Any],
    profile_name: str,
    *,
    extra_keep_keys: set[str] | None = None,
) -> dict[str, str]:
    if profile_name not in NORMALIZATION_PROFILES:
        raise ValueError(f"Unknown normalization profile: {profile_name}")

    profile = NORMALIZATION_PROFILES[profile_name]
    keep_keys = {canonicalize_key(key) for key in profile.get("keep", set())}
    if extra_keep_keys:
        keep_keys |= {canonicalize_key(key) for key in extra_keep_keys}

    canonical_raw: dict[str, str] = {}
    for key, value in raw_params.items():
        canonical_key = canonicalize_key(key)
        if value is None:
            continue
        canonical_raw[canonical_key] = format_incar_value(value)

    normalized: dict[str, str] = {}
    for canonical_key, formatted_value in canonical_raw.items():
        if keep_keys and canonical_key not in keep_keys:
            continue
        normalized[canonical_key] = normalize_value_for_key(canonical_key, formatted_value)

    if "ENCUT" in keep_keys and "ENCUT" not in normalized:
        for fallback_key in ("ENMAX", "ENINI", "ENCUTGW"):
            if fallback_key in canonical_raw:
                normalized["ENCUT"] = normalize_value_for_key("ENCUT", canonical_raw[fallback_key])
                break

    if "EDIFF" in keep_keys and "EDIFF" not in normalized:
        if "EDIFF" in canonical_raw:
            normalized["EDIFF"] = normalize_value_for_key("EDIFF", canonical_raw["EDIFF"])

    if "NELM" in keep_keys and "NELM" not in normalized and "NELM" in canonical_raw:
        normalized["NELM"] = normalize_value_for_key("NELM", canonical_raw["NELM"])

    for key, value in profile.get("defaults", {}).items():
        canonical_key = canonicalize_key(key)
        normalized[canonical_key] = normalize_value_for_key(canonical_key, str(value))

    return normalized


def apply_reference_adjustments(
    params: dict[str, str],
    *,
    remove_keys_raw: str | None,
    overrides_raw: str | None,
) -> dict[str, str]:
    updated = dict(params)

    for key in split_csv_list(remove_keys_raw):
        updated.pop(canonicalize_key(key), None)

    for key, value in parse_key_value_overrides(overrides_raw).items():
        updated[key] = value

    return updated


def incar_text_from_dict(params: dict[str, str]) -> str:
    def sort_key(item_key: str) -> tuple[int, str]:
        if item_key in INCAR_ORDER:
            return (INCAR_ORDER.index(item_key), item_key)
        return (len(INCAR_ORDER), item_key)

    lines = [f"{key:<8}= {params[key]}" for key in sorted(params, key=sort_key)]
    return "\n".join(lines) + "\n"


def choose_task_id(
    *,
    calc_types: dict[Any, Any],
    preferred_task_id: str | None,
    calc_type_pattern: str,
) -> str:
    if preferred_task_id:
        return preferred_task_id

    patterns = [token.lower() for token in split_csv_list(calc_type_pattern)]
    if not patterns:
        raise ValueError("calc_type_pattern must not be empty when preferred_task_id is absent")

    for task_id, calc_type in calc_types.items():
        calc_type_text = str(calc_type).lower()
        if any(pattern in calc_type_text for pattern in patterns):
            return str(task_id)

    raise ValueError(f"No task matched calc_type_pattern={calc_type_pattern!r}")


def build_case_metadata(
    *,
    row: dict[str, str],
    source_kind: str,
    source_reference: dict[str, Any],
    selected_task_id: str,
    selected_calc_type: str,
    formula_pretty: str,
) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "benchmark_type": "incar_generation",
        "source_kind": source_kind,
        "mp_id": row.get("mp_id", ""),
        "formula": row.get("formula") or formula_pretty,
        "task_type": row["task_type"],
        "task_family": row.get("task_family") or row["task_type"],
        "material_family": row.get("material_family", ""),
        "challenge_type": row.get("challenge_type", ""),
        "difficulty": row.get("difficulty", ""),
        "prompt_goal": row["prompt_goal"],
        "selected_task_id": selected_task_id,
        "selected_calc_type": selected_calc_type,
        "normalization_profile": row["normalization_profile"],
        "material_description": row.get("material_description", ""),
        "prompt_constraints": row.get("prompt_constraints", ""),
        "prompt_context": row.get("prompt_context", ""),
        "reference_source": source_reference,
        "reference_adjustments": {
            "reference_remove_keys": split_csv_list(row.get("reference_remove_keys")),
            "reference_overrides": parse_key_value_overrides(row.get("reference_overrides")),
        },
        "notes": row.get("notes", ""),
    }


def build_scoring_payload(row: dict[str, str], reference_params: dict[str, str]) -> dict[str, Any]:
    semantic_must_match = [
        canonicalize_key(value)
        for value in split_csv_list(row.get("semantic_must_match_keys"))
    ]
    policy_match = [
        canonicalize_key(value)
        for value in split_csv_list(row.get("policy_match_keys"))
    ]
    legacy_must_match = [
        canonicalize_key(value)
        for value in split_csv_list(row.get("must_match_keys"))
    ]
    optional_match = [canonicalize_key(value) for value in split_csv_list(row.get("optional_match_keys"))]
    ignore_keys = {
        canonicalize_key(value) for value in split_csv_list(row.get("ignore_keys"))
    } | IGNORED_SCORE_KEYS
    allowed_extra = {canonicalize_key(value) for value in split_csv_list(row.get("allowed_extra_keys"))}
    policy_rule_overrides = {
        canonicalize_key(key): value
        for key, value in parse_json_object(row.get("policy_match_rules_json")).items()
    }

    if legacy_must_match:
        for key in legacy_must_match:
            if key in DEFAULT_POLICY_RULES:
                policy_match.append(key)
            else:
                semantic_must_match.append(key)

    policy_match_rules = {
        key: policy_rule_overrides.get(key, DEFAULT_POLICY_RULES.get(key, {"type": "exact"}))
        for key in policy_match
    }

    return {
        "semantic_must_match_keys": semantic_must_match,
        "policy_match_keys": policy_match,
        "policy_match_rules": policy_match_rules,
        "optional_match_keys": optional_match,
        "ignore_keys": sorted(ignore_keys),
        "allowed_extra_keys": sorted(allowed_extra),
    }


def ensure_output_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_mp_client(api_key: str):
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise RuntimeError(
            "mp-api is required. Install it first, for example: pip install mp-api pymatgen"
        ) from exc

    return MPRester(api_key)


def should_retry_mp_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    transient_signals = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "failed to resolve",
        "max retries exceeded",
        "connection error",
        "connection aborted",
        "connection reset",
        "name resolution",
        "temporarily unavailable",
        "timed out",
        "read timeout",
        "heartbeat",
        "too many requests",
        "rate limit",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "remote end closed connection",
    )
    return any(signal in message for signal in transient_signals)


def mp_retry_delay_seconds(attempt: int) -> int:
    return min(MP_RETRY_DELAY_SECONDS * (2 ** max(attempt - 1, 0)), MP_MAX_RETRY_DELAY_SECONDS)


def fetch_mp_seed_data(
    *,
    api_key: str,
    row: dict[str, str],
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, MP_MAX_RETRIES + 1):
        try:
            with load_mp_client(api_key) as mpr:
                docs = mpr.materials.search(
                    material_ids=[row["mp_id"]],
                    fields=["material_id", "formula_pretty", "structure", "calc_types"],
                )
                if not docs:
                    raise ValueError(f"MP material not found: {row['mp_id']}")
                material_doc = docs[0]
                calc_types = {
                    str(task_id): str(calc_type)
                    for task_id, calc_type in material_doc.calc_types.items()
                }
                selected_task_id = choose_task_id(
                    calc_types=calc_types,
                    preferred_task_id=row.get("preferred_task_id") or None,
                    calc_type_pattern=row["calc_type_pattern"],
                )
                task_docs = mpr.materials.tasks.search(task_ids=[selected_task_id])
                if not task_docs:
                    raise ValueError(f"MP task not found: {selected_task_id}")
                task_doc = task_docs[0]
                raw_params = dict(task_doc.input.parameters)
                structure = getattr(task_doc.input, "structure", None) or material_doc.structure

            return {
                "source_kind": "mp",
                "material_id": str(material_doc.material_id),
                "formula_pretty": str(material_doc.formula_pretty),
                "selected_task_id": selected_task_id,
                "selected_calc_type": calc_types[selected_task_id],
                "structure": structure,
                "raw_incar_params": raw_params,
                "reference_source": {
                    "provider": "Materials Project",
                    "mp_id": row["mp_id"],
                    "task_id": selected_task_id,
                    "calc_type": calc_types[selected_task_id],
                },
            }
        except Exception as exc:
            last_error = exc
            if not should_retry_mp_exception(exc) or attempt == MP_MAX_RETRIES:
                raise
            delay_seconds = mp_retry_delay_seconds(attempt)
            print(
                f"Retrying MP fetch for {row['case_id']} after attempt {attempt}/{MP_MAX_RETRIES} "
                f"failed: {exc}. Sleeping {delay_seconds}s.",
                flush=True,
            )
            time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


def load_structure_from_local_file(path: Path):
    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise RuntimeError("pymatgen is required to read local structure files") from exc

    return Structure.from_file(path)


def fetch_local_seed_data(
    *,
    row: dict[str, str],
    csv_path: Path,
) -> dict[str, Any]:
    structure_path_raw = row.get("local_structure_path") or row.get("local_poscar_path")
    if not structure_path_raw:
        raise ValueError(f"{row['case_id']}: local source requires local_structure_path or local_poscar_path")

    structure_path = resolve_input_path(structure_path_raw, csv_path=csv_path)
    if not structure_path.exists():
        raise FileNotFoundError(f"{row['case_id']}: local structure file not found: {structure_path}")

    incar_json_raw = row.get("local_incar_json_path", "")
    incar_text_raw = row.get("local_incar_path", "")
    if not incar_json_raw and not incar_text_raw:
        raise ValueError(f"{row['case_id']}: local source requires local_incar_json_path or local_incar_path")

    if incar_json_raw:
        incar_json_path = resolve_input_path(incar_json_raw, csv_path=csv_path)
        raw_incar_params = load_json(incar_json_path)
    else:
        incar_text_path = resolve_input_path(incar_text_raw, csv_path=csv_path)
        raw_incar_params = parse_incar_file(incar_text_path)

    structure = load_structure_from_local_file(structure_path)
    formula_pretty = row.get("formula") or structure.composition.reduced_formula
    selected_task_id = row.get("preferred_task_id") or "local_seed"
    selected_calc_type = row.get("local_calc_type") or row.get("task_type") or "local_seed"

    return {
        "source_kind": "local",
        "material_id": row.get("mp_id") or row["case_id"],
        "formula_pretty": formula_pretty,
        "selected_task_id": selected_task_id,
        "selected_calc_type": selected_calc_type,
        "structure": structure,
        "raw_incar_params": raw_incar_params,
        "reference_source": {
            "provider": "local",
            "structure_path": str(structure_path),
            "incar_json_path": str(resolve_input_path(incar_json_raw, csv_path=csv_path)) if incar_json_raw else None,
            "incar_path": str(resolve_input_path(incar_text_raw, csv_path=csv_path)) if incar_text_raw else None,
            "task_id": selected_task_id,
            "calc_type": selected_calc_type,
        },
    }


def fetch_seed_data(
    *,
    api_key: str,
    row: dict[str, str],
    csv_path: Path,
) -> dict[str, Any]:
    source_kind = row_source_kind(row)
    if source_kind == "mp":
        if not row.get("mp_id"):
            raise ValueError(f"{row['case_id']}: mp source requires mp_id")
        if not api_key:
            raise ValueError("materials_project.api_key is required for source_kind=mp")
        return fetch_mp_seed_data(api_key=api_key, row=row)
    if source_kind == "local":
        return fetch_local_seed_data(row=row, csv_path=csv_path)
    raise ValueError(f"{row['case_id']}: unsupported source_kind={source_kind!r}")


def prompt_messages_for_case(case_dir: Path) -> list[dict[str, str]]:
    metadata = load_json(case_dir / "metadata.json")
    poscar_text = (case_dir / "inputs" / "POSCAR").read_text(encoding="utf-8")

    system_prompt = (
        "You are an expert VASP user. Generate a valid INCAR for the requested task. "
        "Output only the final INCAR text, without markdown or explanation."
    )

    user_sections = [
        f"Case ID: {metadata['case_id']}",
        f"Material: {metadata['formula']}",
        f"Task type: {metadata['task_type']}",
        f"Goal: {metadata['prompt_goal']}",
    ]

    if metadata.get("material_description"):
        user_sections.append(f"Material description: {metadata['material_description']}")
    if metadata.get("prompt_constraints"):
        user_sections.append(f"Constraints: {metadata['prompt_constraints']}")
    if metadata.get("prompt_context"):
        user_sections.append(f"Context: {metadata['prompt_context']}")

    user_sections.append("POSCAR:")
    user_sections.append(poscar_text.strip())
    user_sections.append("Write the INCAR now.")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(user_sections)},
    ]


def extract_incar_from_response(text: str) -> str:
    fence_match = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        extracted = fence_match.group(1).strip()
        if extracted:
            return extracted + "\n"

    cleaned = text.strip()
    return cleaned + ("\n" if cleaned else "")


def http_json_request(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int = 120,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc

    return json.loads(body)


def _extract_text_from_openai_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def _response_to_payload(response: Any) -> dict[str, Any]:
    def json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [json_safe(item) for item in value]
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            try:
                return isoformat()
            except Exception:
                pass
        return repr(value)

    if response is None:
        return {}
    if isinstance(response, dict):
        return json_safe(response)
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            return json_safe(model_dump(mode="json"))
        except Exception:
            try:
                return json_safe(model_dump())
            except Exception:
                pass
    try:
        return json.loads(str(response))
    except Exception:
        return {"raw_response_repr": repr(response)}


def _extract_text_from_openai_response(response: Any) -> str:
    if response is None:
        return ""

    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message is not None:
            return _extract_text_from_openai_content(getattr(message, "content", None))

    if isinstance(response, dict):
        choices = response.get("choices")
        if choices:
            message = choices[0].get("message", {})
            return _extract_text_from_openai_content(message.get("content"))

        output_text = response.get("output_text")
        if output_text:
            return str(output_text)

        output = response.get("output")
        if output:
            parts: list[str] = []
            for item in output:
                content = item.get("content", []) if isinstance(item, dict) else []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        parts.append(str(block.get("text", "")))
            if parts:
                return "".join(parts)

    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    output = getattr(response, "output", None)
    if output:
        parts: list[str] = []
        for item in output:
            content = getattr(item, "content", None) or []
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type == "output_text":
                    parts.append(str(getattr(block, "text", "")))
        if parts:
            return "".join(parts)

    return ""


def _normalize_text_messages_for_openai(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            normalized.append({"role": role, "content": content})
        else:
            normalized.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                }
            )
    return normalized


def _normalize_text_messages_for_responses_api(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, Any]]]:
    instructions_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            instructions_parts.append(content)
            continue

        input_items.append(
            {
                "role": role,
                "content": [
                    {
                        "type": "input_text",
                        "text": content,
                    }
                ],
            }
        )

    instructions = "\n\n".join(instructions_parts) if instructions_parts else None
    return instructions, input_items


def _openai_sdk_client(model_cfg: dict[str, Any]):
    try:
        from openai import OpenAI
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "openai package is required for SDK-based model invocation. Install it in .venv first."
        ) from exc

    if bool(model_cfg.get("vertexai", False)):
        try:
            from google.auth import default
            import google.auth.transport.requests
        except ImportError as exc:
            raise RuntimeError(
                "google-auth package is required for Vertex AI OpenAI-compatible invocation. Install it in .venv first."
            ) from exc

        project = str(model_cfg.get("project") or "").strip()
        location = str(model_cfg.get("location") or "").strip()
        if not project or not location:
            raise RuntimeError("Vertex AI model config must define non-empty project and location")

        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(google.auth.transport.requests.Request())
        token = getattr(credentials, "token", None)
        if not token:
            raise RuntimeError("Failed to obtain a Vertex AI access token via Application Default Credentials")

        base_url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/"
            f"{project}/locations/{location}/endpoints/openapi"
        )
        client_kwargs: dict[str, Any] = {
            "api_key": token,
            "base_url": base_url,
        }
    else:
        client_kwargs = {
            "api_key": model_cfg["api_key"],
            "base_url": model_cfg["base_url"].rstrip("/"),
        }

    if model_cfg.get("organization"):
        client_kwargs["organization"] = model_cfg["organization"]
    if model_cfg.get("headers"):
        client_kwargs["default_headers"] = model_cfg["headers"]

    proxy_url = _proxy_url_for_model(model_cfg)
    http_client_kwargs: dict[str, Any] = {"trust_env": False}
    if proxy_url is not None:
        http_client_kwargs["proxy"] = proxy_url
    client_kwargs["http_client"] = httpx.Client(**http_client_kwargs)

    return OpenAI(**client_kwargs)


def _gcloud_access_token() -> str:
    now = time.time()
    cached = _GCLOUD_TOKEN_CACHE.get("token")
    acquired_at = float(_GCLOUD_TOKEN_CACHE.get("acquired_at") or 0.0)
    if cached and now - acquired_at < GCLOUD_TOKEN_TTL_SECONDS:
        return str(cached)

    try:
        completed = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("gcloud CLI is required for access_method=third_party_via_vertex") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            "Failed to obtain Vertex bearer token via `gcloud auth print-access-token`. "
            "Run `gcloud auth login` first." + (f" stderr={stderr}" if stderr else "")
        ) from exc

    token = completed.stdout.strip()
    if not token:
        raise RuntimeError("`gcloud auth print-access-token` returned an empty token")

    _GCLOUD_TOKEN_CACHE["token"] = token
    _GCLOUD_TOKEN_CACHE["acquired_at"] = now
    return token


def _vertex_openapi_base_url(model_cfg: dict[str, Any]) -> str:
    endpoint = str(model_cfg.get("endpoint") or "").strip()
    region = str(model_cfg.get("region") or "").strip()
    project_id = str(model_cfg.get("project_id") or "").strip()
    api_version = str(model_cfg.get("api_version") or "v1beta1").strip()
    if not endpoint or not region or not project_id:
        raise RuntimeError("third_party_via_vertex requires endpoint, region, and project_id")
    return (
        f"https://{endpoint}/"
        f"{api_version}/projects/{project_id}/locations/{region}/endpoints/openapi"
    )


def _third_party_via_vertex_client(model_cfg: dict[str, Any]):
    try:
        from openai import OpenAI
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "openai package is required for access_method=third_party_via_vertex. Install it in .venv first."
        ) from exc

    token = _gcloud_access_token()
    client_kwargs: dict[str, Any] = {
        "api_key": token,
        "base_url": _vertex_openapi_base_url(model_cfg),
    }
    if model_cfg.get("headers"):
        client_kwargs["default_headers"] = model_cfg["headers"]

    proxy_url = _proxy_url_for_model(model_cfg)
    http_client_kwargs: dict[str, Any] = {"trust_env": False}
    if proxy_url is not None:
        http_client_kwargs["proxy"] = proxy_url
    client_kwargs["http_client"] = httpx.Client(**http_client_kwargs)
    return OpenAI(**client_kwargs)


def invoke_third_party_via_vertex_model(
    *,
    model_cfg: dict[str, Any],
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    client = _third_party_via_vertex_client(model_cfg)
    request_params = _drop_none_values(dict(model_cfg.get("default_params", {})))
    response = client.chat.completions.create(
        model=model_cfg["model_id"],
        messages=messages,
        **request_params,
    )
    text = _extract_text_from_openai_response(response)
    return text, _response_to_payload(response)


def _anthropic_sdk_client(model_cfg: dict[str, Any]):
    try:
        import anthropic
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "anthropic package is required for Anthropic SDK-based invocation. Install it in .venv first."
        ) from exc

    client_kwargs: dict[str, Any] = {
        "api_key": model_cfg["api_key"],
        "base_url": model_cfg["base_url"].rstrip("/"),
        "default_headers": model_cfg.get("headers", {}) or None,
    }
    proxy_url = _proxy_url_for_model(model_cfg)
    http_client_kwargs: dict[str, Any] = {"trust_env": False}
    if proxy_url is not None:
        http_client_kwargs["proxy"] = proxy_url
    client_kwargs["http_client"] = httpx.Client(**http_client_kwargs)
    return anthropic.Anthropic(**client_kwargs)


def _model_uses_proxy(model_cfg: dict[str, Any]) -> bool:
    return bool(model_cfg.get("use_proxy", False))


def _proxy_url_for_model(model_cfg: dict[str, Any]) -> str | None:
    if not _model_uses_proxy(model_cfg):
        return None
    raw = str(model_cfg.get("proxy_url") or DEFAULT_LOCAL_PROXY_URL).strip()
    return raw.rstrip("/") if raw else DEFAULT_LOCAL_PROXY_URL


def _google_genai_client(model_cfg: dict[str, Any]):
    try:
        from google import genai
        from google.genai import errors as genai_errors
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "google-genai package is required for Gemini SDK-based invocation. Install it in .venv first."
        ) from exc

    proxy_url = _proxy_url_for_model(model_cfg)
    client_args: dict[str, Any] = {"trust_env": False}
    async_client_args: dict[str, Any] = {"trust_env": False}
    if proxy_url is not None:
        client_args["proxy"] = proxy_url
        async_client_args["proxy"] = proxy_url

    if bool(model_cfg.get("vertexai", False)):
        project = str(model_cfg.get("project") or "").strip()
        location = str(model_cfg.get("location") or "").strip()

        http_options = types.HttpOptions(
            api_version=model_cfg.get("api_version", "v1"),
            headers=model_cfg.get("headers", {}) or None,
            timeout=model_cfg.get("timeout_ms"),
            client_args=client_args,
            async_client_args=async_client_args,
        )
        client_kwargs: dict[str, Any] = {
            "vertexai": True,
            "http_options": http_options,
        }
        if model_cfg.get("api_key"):
            client_kwargs["api_key"] = model_cfg["api_key"]
            env_overrides = {}
            if project:
                env_overrides["GOOGLE_CLOUD_PROJECT"] = project
            if location:
                env_overrides["GOOGLE_CLOUD_LOCATION"] = location
            previous = {key: os.environ.get(key) for key in env_overrides}
            try:
                os.environ.update(env_overrides)
                return (genai.Client(**client_kwargs), types, genai_errors)
            finally:
                for key, old in previous.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old
        if project:
            client_kwargs["project"] = project
        if location:
            client_kwargs["location"] = location

        return (genai.Client(**client_kwargs), types, genai_errors)

    http_options = types.HttpOptions(
        base_url=model_cfg["base_url"].rstrip("/") + "/",
        api_version=model_cfg.get("api_version", "v1beta"),
        headers=model_cfg.get("headers", {}) or None,
        timeout=model_cfg.get("timeout_ms"),
        client_args=client_args,
        async_client_args=async_client_args,
    )
    return genai.Client(api_key=model_cfg["api_key"], http_options=http_options), types, genai_errors


def _google_genai_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _should_retry_google_genai_exception(exc: Exception) -> bool:
    status_code = _google_genai_status_code(exc)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True

    text = str(exc).lower()
    transient_markers = (
        "no available gemini accounts",
        "no available accounts",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "connection reset",
        "service unavailable",
        "resource exhausted",
        "rate limit",
        "overloaded",
    )
    return any(marker in text for marker in transient_markers)


def _google_genai_retry_delay_seconds(model_cfg: dict[str, Any], attempt: int) -> int:
    base_delay = max(1, int(model_cfg.get("sdk_retry_base_delay_seconds", 15)))
    max_delay = max(base_delay, int(model_cfg.get("sdk_retry_max_delay_seconds", 180)))
    return min(base_delay * (2 ** max(0, attempt - 1)), max_delay)


def _should_retry_model_exception(exc: Exception) -> bool:
    text = str(exc).lower()
    transient_markers = (
        "request timed out",
        "read timeout",
        "timed out",
        "timeout",
        "rate limit",
        "too many requests",
        "temporarily unavailable",
        "service unavailable",
        "connection reset",
        "connection aborted",
        "connection refused",
        "connection error",
        "remote protocol error",
        "server disconnected",
        "resource exhausted",
        "overloaded",
        "bad gateway",
        "gateway timeout",
        "internal server error",
    )
    if any(marker in text for marker in transient_markers):
        return True

    class_name = type(exc).__name__
    return class_name in {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "ReadTimeout",
        "ConnectTimeout",
        "RemoteProtocolError",
    }


def _model_retry_delay_seconds(model_cfg: dict[str, Any], attempt: int) -> int:
    base_delay = max(
        1,
        int(
            model_cfg.get("request_retry_base_delay_seconds")
            or model_cfg.get("sdk_retry_base_delay_seconds")
            or 15
        ),
    )
    max_delay = max(
        base_delay,
        int(
            model_cfg.get("request_retry_max_delay_seconds")
            or model_cfg.get("sdk_retry_max_delay_seconds")
            or 180
        ),
    )
    return min(base_delay * (2 ** max(0, attempt - 1)), max_delay)


def invoke_model_with_retries(
    *,
    model_cfg: dict[str, Any],
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    max_attempts = max(
        1,
        int(
            model_cfg.get("request_max_attempts")
            or model_cfg.get("sdk_max_attempts")
            or 3
        ),
    )
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return invoke_model(model_cfg=model_cfg, messages=messages)
        except Exception as exc:
            last_exc = exc
            retryable = _should_retry_model_exception(exc)
            if not retryable or attempt >= max_attempts:
                raise

            delay_seconds = _model_retry_delay_seconds(model_cfg, attempt)
            print(
                f"Transient model error for model={model_cfg['name']} attempt {attempt}/{max_attempts}: {exc}; retrying in {delay_seconds}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay_seconds)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Model invocation failed for model={model_cfg['name']}")


def _drop_none_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _normalize_text_messages_for_google_genai(
    messages: list[dict[str, str]],
) -> tuple[str | None, str]:
    system_parts: list[str] = []
    content_parts: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            content_parts.append(content)
        else:
            content_parts.append(f"{role.upper()}:\n{content}")
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    contents = "\n\n".join(content_parts)
    return system_instruction, contents


def _extract_text_from_google_genai_response(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)

    candidates = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(str(part_text))
    return "".join(parts)


def _moonshot_request_params(model_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    raw_params = dict(model_cfg.get("default_params", {}))
    direct_params = _drop_none_values(
        {
            "max_tokens": raw_params.get("max_tokens"),
        }
    )

    extra_body = _drop_none_values(
        {
            "thinking": raw_params.get("thinking"),
        }
    )

    return direct_params, (extra_body or None)


def _qwen_request_params(model_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    raw_params = dict(model_cfg.get("default_params", {}))
    direct_params = _drop_none_values(
        {
            "max_tokens": raw_params.get("max_tokens"),
            "temperature": raw_params.get("temperature"),
            "top_p": raw_params.get("top_p"),
        }
    )
    extra_body = _drop_none_values(
        {
            "enable_thinking": raw_params.get("enable_thinking"),
            "thinking_budget": raw_params.get("thinking_budget"),
        }
    )
    return direct_params, (extra_body or None)


def invoke_openai_model(
    *,
    model_cfg: dict[str, Any],
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    client = _openai_sdk_client(model_cfg)
    raw_params = dict(model_cfg.get("default_params", {}))
    instructions, input_items = _normalize_text_messages_for_responses_api(messages)
    request_params = _drop_none_values(
        {
            "max_output_tokens": raw_params.get("max_tokens"),
            **{k: v for k, v in raw_params.items() if k != "max_tokens"},
        }
    )
    response = client.responses.create(
        model=model_cfg["model_id"],
        instructions=instructions,
        input=input_items,
        **request_params,
    )
    text = _extract_text_from_openai_response(response)
    return text, _response_to_payload(response)


def invoke_moonshot_model(
    *,
    model_cfg: dict[str, Any],
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    client = _openai_sdk_client(model_cfg)
    sdk_messages = _normalize_text_messages_for_openai(messages)
    request_params, extra_body = _moonshot_request_params(model_cfg)
    try:
        response = client.chat.completions.create(
            model=model_cfg["model_id"],
            messages=sdk_messages,
            **request_params,
            extra_body=extra_body,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Moonshot call failed for model={model_cfg['model_id']} base_url={model_cfg['base_url']}: {exc}"
        ) from exc

    text = _extract_text_from_openai_response(response)
    return text, _response_to_payload(response)


def invoke_deepseek_model(
    *,
    model_cfg: dict[str, Any],
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    client = _openai_sdk_client(model_cfg)
    request_params = _drop_none_values(dict(model_cfg.get("default_params", {})))
    response = client.chat.completions.create(
        model=model_cfg["model_id"],
        messages=messages,
        **request_params,
    )
    text = _extract_text_from_openai_response(response)
    return text, _response_to_payload(response)


def invoke_anthropic_model(
    *,
    model_cfg: dict[str, Any],
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    client = _anthropic_sdk_client(model_cfg)
    system = ""
    filtered_messages = []
    for message in messages:
        if message["role"] == "system":
            system = message["content"]
        else:
            filtered_messages.append(message)

    raw_params = dict(model_cfg.get("default_params", {}))
    request_params = _drop_none_values(raw_params)
    max_tokens = request_params.pop("max_tokens", None) or model_cfg.get("max_tokens") or 4096
    response = client.messages.create(
        model=model_cfg["model_id"],
        messages=filtered_messages,
        system=system or None,
        max_tokens=int(max_tokens),
        **request_params,
    )
    text = "".join(getattr(block, "text", "") for block in getattr(response, "content", []) or [])
    return text, _response_to_payload(response)


def invoke_google_genai_model(
    *,
    model_cfg: dict[str, Any],
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    client, genai_types, genai_errors = _google_genai_client(model_cfg)
    system_instruction, contents = _normalize_text_messages_for_google_genai(messages)
    raw_params = dict(model_cfg.get("default_params", {}))
    config_kwargs = _drop_none_values(
        {
            "system_instruction": system_instruction,
            "temperature": raw_params.get("temperature"),
            "top_p": raw_params.get("top_p"),
            "top_k": raw_params.get("top_k"),
            "max_output_tokens": raw_params.get("max_output_tokens") or raw_params.get("max_tokens"),
        }
    )
    max_attempts = max(1, int(model_cfg.get("sdk_max_attempts", 6)))
    response = None
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=model_cfg["model_id"],
                contents=contents,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
            break
        except Exception as exc:
            last_exc = exc
            retryable = isinstance(exc, genai_errors.APIError) and _should_retry_google_genai_exception(exc)
            if not retryable or attempt >= max_attempts:
                raise RuntimeError(
                    f"Gemini SDK call failed for model={model_cfg['model_id']} base_url={model_cfg['base_url']} "
                    f"after {attempt} attempt(s): {exc}"
                ) from exc

            delay_seconds = _google_genai_retry_delay_seconds(model_cfg, attempt)
            print(
                f"Gemini transient error for model={model_cfg['name']} attempt {attempt}/{max_attempts}: {exc}; retrying in {delay_seconds}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay_seconds)

    if response is None:
        raise RuntimeError(
            f"Gemini SDK call failed for model={model_cfg['model_id']} base_url={model_cfg['base_url']}: {last_exc}"
        )
    text = _extract_text_from_google_genai_response(response)
    return text, _response_to_payload(response)


def invoke_local_openai_compatible_model(
    *,
    model_cfg: dict[str, Any],
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    client = _openai_sdk_client(model_cfg)
    provider = (model_cfg.get("provider") or "").strip().lower()
    if provider == "qwen":
        request_params, extra_body = _qwen_request_params(model_cfg)
        response = client.chat.completions.create(
            model=model_cfg["model_id"],
            messages=messages,
            extra_body=extra_body,
            **request_params,
        )
    else:
        request_params = _drop_none_values(dict(model_cfg.get("default_params", {})))
        response = client.chat.completions.create(
            model=model_cfg["model_id"],
            messages=messages,
            **request_params,
        )
    text = _extract_text_from_openai_response(response)
    return text, _response_to_payload(response)


def invoke_openai_compatible_http_raw_model(
    *,
    model_cfg: dict[str, Any],
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    provider = (model_cfg.get("provider") or "").strip().lower()
    payload: dict[str, Any] = {
        "model": model_cfg["model_id"],
        "messages": messages,
        **_drop_none_values(dict(model_cfg.get("default_params", {}))),
    }
    if provider == "qwen":
        request_params, extra_body = _qwen_request_params(model_cfg)
        payload = {
            "model": model_cfg["model_id"],
            "messages": messages,
            **request_params,
            **(extra_body or {}),
        }
    elif provider == "moonshot":
        request_params, extra_body = _moonshot_request_params(model_cfg)
        payload = {
            "model": model_cfg["model_id"],
            "messages": messages,
            **request_params,
            **(extra_body or {}),
        }

    url = model_cfg["base_url"].rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_cfg['api_key']}",
        **(model_cfg.get("headers") or {}),
    }
    raw = http_json_request(
        url=url,
        headers=headers,
        payload=payload,
        timeout=int(model_cfg.get("timeout_ms") or 120000) // 1000,
    )
    text = _extract_text_from_openai_response(raw)
    return text, _response_to_payload(raw)


def invoke_model(
    *,
    model_cfg: dict[str, Any],
    messages: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    access_method = model_cfg["access_method"]
    provider = (model_cfg.get("provider") or "").strip().lower()
    base_url = str(model_cfg.get("base_url") or "").rstrip("/")
    default_params = dict(model_cfg.get("default_params", {}))
    headers = {"Content-Type": "application/json", **model_cfg.get("headers", {})}

    if provider == "moonshot":
        return invoke_moonshot_model(model_cfg=model_cfg, messages=messages)

    if access_method == "third_party_via_vertex":
        return invoke_third_party_via_vertex_model(model_cfg=model_cfg, messages=messages)

    if provider == "openai":
        return invoke_openai_model(model_cfg=model_cfg, messages=messages)

    if provider == "deepseek":
        return invoke_deepseek_model(model_cfg=model_cfg, messages=messages)

    if provider == "anthropic" and access_method in {"anthropic_sdk", "anthropic_http"}:
        return invoke_anthropic_model(model_cfg=model_cfg, messages=messages)

    if provider == "gemini" and access_method == "google_genai_sdk":
        return invoke_google_genai_model(model_cfg=model_cfg, messages=messages)

    if provider in {"local", "ollama", "qwen", "glm", "gemini"} and access_method in {
        "openai_sdk",
        "openai_compatible_http",
    }:
        return invoke_local_openai_compatible_model(model_cfg=model_cfg, messages=messages)

    if access_method == "openai_compatible_http_raw":
        return invoke_openai_compatible_http_raw_model(model_cfg=model_cfg, messages=messages)

    if access_method in {"openai_compatible_http", "openai_sdk"}:
        return invoke_openai_model(model_cfg=model_cfg, messages=messages)

    if access_method == "anthropic_http":
        headers["x-api-key"] = model_cfg["api_key"]
        headers["anthropic-version"] = model_cfg.get("anthropic_version", "2023-06-01")
        system = ""
        filtered_messages = []
        for message in messages:
            if message["role"] == "system":
                system = message["content"]
            else:
                filtered_messages.append(message)

        payload = {
            "model": model_cfg["model_id"],
            "messages": filtered_messages,
            "system": system,
            **default_params,
        }
        raw = http_json_request(
            url=f"{base_url}/v1/messages",
            headers=headers,
            payload=payload,
        )
        text = "".join(block.get("text", "") for block in raw.get("content", []))
        return text, raw

    if access_method == "mock_copy_reference":
        raise RuntimeError("mock_copy_reference should be handled by the batch runner directly")

    raise ValueError(f"Unsupported access_method: {access_method}")


def enabled_models(config_path: Path | None = None, model_names: list[str] | None = None) -> list[dict[str, Any]]:
    config = load_llm_benchmark_config(config_path)
    models = [model for model in config["models"] if model.get("enabled")]
    if model_names:
        requested = set(model_names)
        models = [model for model in models if model["name"] in requested]
    return models


def score_generated_incar(
    *,
    case_dir: Path,
    candidate_path: Path,
    model_name: str,
) -> dict[str, Any]:
    metadata = load_json(case_dir / "metadata.json")
    scoring = load_json(case_dir / "scoring.json")
    reference_params = parse_incar_file(case_dir / "inputs" / "INCAR_reference")
    candidate_params = parse_incar_file(candidate_path)
    raw_seed_path = case_dir / "inputs" / "INCAR_mp_raw.json"
    raw_seed_params = load_json(raw_seed_path) if raw_seed_path.exists() else {}
    effective_candidate_params, default_imputed_keys = apply_candidate_defaults_for_scoring(
        reference_params=reference_params,
        candidate_params=candidate_params,
        raw_seed_params=raw_seed_params,
    )

    semantic_must_match = [
        canonicalize_key(key) for key in scoring.get("semantic_must_match_keys", [])
    ]
    policy_match = [canonicalize_key(key) for key in scoring.get("policy_match_keys", [])]
    policy_match_rules = {
        canonicalize_key(key): value
        for key, value in scoring.get("policy_match_rules", {}).items()
    }
    optional_match = [canonicalize_key(key) for key in scoring.get("optional_match_keys", [])]
    ignore_keys = {canonicalize_key(key) for key in scoring.get("ignore_keys", [])}
    allowed_extra = {canonicalize_key(key) for key in scoring.get("allowed_extra_keys", [])}

    if "must_match_keys" in scoring:
        raise ValueError(
            f"{case_dir}/scoring.json is using legacy schema with must_match_keys. "
            "Please rebuild the benchmark with the new scoring schema."
        )

    def value_for(params: dict[str, str], key: str) -> str | None:
        return params.get(key)

    semantic_items = []
    semantic_hits = 0
    for key in semantic_must_match:
        expected = value_for(reference_params, key)
        observed = value_for(effective_candidate_params, key)
        matched = expected == observed
        semantic_hits += int(matched)
        semantic_items.append({"parameter": key, "expected": expected, "observed": observed, "matched": matched})

    def parse_single_numeric(text: str | None) -> float | None:
        if text is None:
            return None
        if " " in text.strip():
            return None
        if not NUMERIC_TOKEN_RE.match(text.strip()):
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def compare_policy(key: str, expected: str | None, observed: str | None, rule: dict[str, Any]) -> str:
        if expected is None or observed is None:
            return "mismatch"
        if expected == observed:
            return "exact"

        rule_type = rule.get("type", "exact")
        expected_num = parse_single_numeric(expected)
        observed_num = parse_single_numeric(observed)

        if key == "ISMEAR" and expected_num is not None and observed_num is not None:
            if is_ismear_policy_consistent(expected, observed):
                return "tolerated"
            return "mismatch"

        if rule_type == "exact" or expected_num is None or observed_num is None:
            return "mismatch"

        if rule_type == "abs_tolerance":
            max_abs_diff = float(rule.get("max_abs_diff", 0.0))
            return "tolerated" if abs(observed_num - expected_num) <= max_abs_diff else "mismatch"

        if rule_type == "directional_numeric":
            allow_higher_rel = float(rule.get("allow_higher_rel", 0.0))
            allow_lower_rel = float(rule.get("allow_lower_rel", 0.0))
            if observed_num > expected_num:
                return (
                    "tolerated"
                    if observed_num <= expected_num * (1 + allow_higher_rel)
                    else "mismatch"
                )
            return (
                "tolerated"
                if observed_num >= expected_num * (1 - allow_lower_rel)
                else "mismatch"
            )

        if rule_type == "strictness_numeric":
            lower_is_stricter = bool(rule.get("lower_is_stricter", True))
            max_looser_ratio = float(rule.get("max_looser_ratio", 1.0))
            if lower_is_stricter:
                if observed_num <= expected_num:
                    return "tolerated"
                return "tolerated" if observed_num <= expected_num * max_looser_ratio else "mismatch"
            if observed_num >= expected_num:
                return "tolerated"
            return "tolerated" if observed_num >= expected_num / max_looser_ratio else "mismatch"

        if rule_type == "signed_ratio":
            if bool(rule.get("sign_must_match", False)):
                if (expected_num < 0) != (observed_num < 0):
                    return "mismatch"
            if expected_num == 0 or observed_num == 0:
                return "mismatch"
            ratio = max(abs(expected_num), abs(observed_num)) / min(abs(expected_num), abs(observed_num))
            return "tolerated" if ratio <= float(rule.get("max_ratio", 1.0)) else "mismatch"

        return "mismatch"

    policy_items = []
    policy_hits = 0
    for key in policy_match:
        expected = value_for(reference_params, key)
        observed = value_for(effective_candidate_params, key)
        rule = policy_match_rules.get(key, {"type": "exact"})
        status = compare_policy(key, expected, observed, rule)
        matched = status in {"exact", "tolerated"}
        policy_hits += int(matched)
        policy_items.append(
            {
                "parameter": key,
                "expected": expected,
                "observed": observed,
                "status": status,
                "matched": matched,
                "rule": rule,
            }
        )

    optional_items = []
    optional_hits = 0
    for key in optional_match:
        expected = value_for(reference_params, key)
        observed = value_for(effective_candidate_params, key)
        matched = expected == observed
        optional_hits += int(matched)
        optional_items.append({"parameter": key, "expected": expected, "observed": observed, "matched": matched})

    reference_keys = {key for key in reference_params if key not in ignore_keys}
    candidate_keys = {key for key in effective_candidate_params if key not in ignore_keys}
    extra_keys = sorted(key for key in candidate_keys - reference_keys if key not in allowed_extra)
    missing_semantic = sorted(item["parameter"] for item in semantic_items if not item["matched"])
    missing_policy = sorted(item["parameter"] for item in policy_items if not item["matched"])

    semantic_percent = (
        100.0
        if not semantic_must_match
        else 100.0 * semantic_hits / len(semantic_must_match)
    )
    policy_percent = (
        100.0
        if not policy_match
        else 100.0 * policy_hits / len(policy_match)
    )
    must_percent = round((semantic_percent + policy_percent) / 2.0, 2)
    optional_percent = (
        100.0
        if not optional_match
        else 100.0 * optional_hits / len(optional_match)
    )
    extra_keys_percent = max(0.0, 100.0 - 20.0 * len(extra_keys))

    case_context = {
        "difficulty": metadata.get("difficulty"),
        "task_type": metadata.get("task_type"),
        "task_family": metadata.get("task_family"),
        "material_family": metadata.get("material_family"),
        "challenge_type": metadata.get("challenge_type"),
    }

    runnable_assessment = minimum_task_runnable_assessment(
        metadata=metadata,
        scoring=scoring,
        reference_params=reference_params,
        candidate_params=effective_candidate_params,
    )
    minimum_task_runnable = bool(runnable_assessment["passed"])

    return {
        "generated_at_utc": utc_now(),
        "case_id": metadata["case_id"],
        "model_name": model_name,
        "case_context": case_context,
        "status": "graded",
        "score_breakdown": {
            "must_match": must_percent,
            "must_match_semantic": round(semantic_percent, 2),
            "must_match_policy": round(policy_percent, 2),
            "optional_match": round(optional_percent, 2),
            "extra_keys": round(extra_keys_percent, 2),
        },
        "score_weights": {
            "must_match_semantic": 0.50,
            "must_match_policy": 0.50,
        },
        "semantic_must_match": {
            "total": len(semantic_must_match),
            "matched": semantic_hits,
            "items": semantic_items,
        },
        "policy_match": {
            "total": len(policy_match),
            "matched": policy_hits,
            "items": policy_items,
        },
        "optional_match": {
            "total": len(optional_match),
            "matched": optional_hits,
            "items": optional_items,
        },
        "extra_keys": extra_keys,
        "default_imputed_keys": default_imputed_keys,
        "minimum_task_runnable": minimum_task_runnable,
        "minimum_task_runnable_reasons": runnable_assessment["reasons"],
        "perfect_case": not missing_semantic and not missing_policy and not extra_keys,
        "missing_required_keys": {
            "semantic": missing_semantic,
            "policy": missing_policy,
        },
    }


def missing_generation_grade(case_dir: Path, model_name: str, candidate_path: Path) -> dict[str, Any]:
    metadata = load_json(case_dir / "metadata.json")
    return {
        "generated_at_utc": utc_now(),
        "case_id": metadata["case_id"],
        "model_name": model_name,
        "case_context": {
            "difficulty": metadata.get("difficulty"),
            "task_type": metadata.get("task_type"),
            "task_family": metadata.get("task_family"),
            "material_family": metadata.get("material_family"),
            "challenge_type": metadata.get("challenge_type"),
        },
        "status": "missing_candidate",
        "candidate_path": str(candidate_path),
        "minimum_task_runnable": None,
        "minimum_task_runnable_reasons": [],
    }


def summarize_grade_subset(grades: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [grade for grade in grades if grade.get("status") == "graded"]
    average_must_match_score = (
        round(sum(grade["score_breakdown"]["must_match"] for grade in graded) / len(graded), 2)
        if graded
        else None
    )
    average_semantic_score = (
        round(sum(grade["score_breakdown"]["must_match_semantic"] for grade in graded) / len(graded), 2)
        if graded
        else None
    )
    average_policy_score = (
        round(sum(grade["score_breakdown"]["must_match_policy"] for grade in graded) / len(graded), 2)
        if graded
        else None
    )
    average_optional_match_score = (
        round(sum(grade["score_breakdown"]["optional_match"] for grade in graded) / len(graded), 2)
        if graded
        else None
    )
    average_extra_keys_score = (
        round(sum(grade["score_breakdown"]["extra_keys"] for grade in graded) / len(graded), 2)
        if graded
        else None
    )
    perfect_cases = sum(1 for grade in graded if grade.get("perfect_case"))
    runnable_cases = sum(1 for grade in graded if grade.get("minimum_task_runnable") is True)

    return {
        "total_cases": len(grades),
        "graded_cases": len(graded),
        "missing_cases": len(grades) - len(graded),
        "average_scores": {
            "must_match": average_must_match_score,
            "must_match_semantic": average_semantic_score,
            "must_match_policy": average_policy_score,
            "optional_match": average_optional_match_score,
            "extra_keys": average_extra_keys_score,
        },
        "perfect_cases": perfect_cases,
        "perfect_case_rate": round(perfect_cases / len(graded), 4) if graded else None,
        "minimum_task_runnable_rate": round(runnable_cases / len(graded), 4) if graded else None,
        "case_ids": [grade["case_id"] for grade in grades],
    }


def _group_sort_key(value: str) -> tuple[int, str]:
    if value.startswith("L") and value[1:].isdigit():
        return (0, f"{int(value[1:]):04d}")
    preferred_task_order = {
        "static_scf": 0,
        "geometry_relax": 1,
        "line_mode_bands": 2,
        "dos_nscf": 3,
    }
    if value in preferred_task_order:
        return (1, f"{preferred_task_order[value]:04d}")
    return (2, value)


def grouped_generation_summaries(
    grades: list[dict[str, Any]],
    *,
    group_key: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for grade in grades:
        case_context = grade.get("case_context") or {}
        group_value = case_context.get(group_key) or "unknown"
        grouped.setdefault(str(group_value), []).append(grade)

    return {
        group_value: summarize_grade_subset(grouped[group_value])
        for group_value in sorted(grouped, key=_group_sort_key)
    }


def summarize_generation_grades(
    *,
    benchmark_root: Path,
    model_name: str,
    grades: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_core = summarize_grade_subset(grades)

    return {
        "generated_at_utc": utc_now(),
        "benchmark_root": str(benchmark_root),
        "model_name": model_name,
        **summary_core,
        "by_difficulty": grouped_generation_summaries(grades, group_key="difficulty"),
        "by_task_type": grouped_generation_summaries(grades, group_key="task_type"),
        "by_task_family": grouped_generation_summaries(grades, group_key="task_family"),
        "by_material_family": grouped_generation_summaries(grades, group_key="material_family"),
        "by_challenge_type": grouped_generation_summaries(grades, group_key="challenge_type"),
        "cases": [
            {
                "case_id": grade["case_id"],
                "status": grade["status"],
                "case_context": grade.get("case_context", {}),
                "scores": grade.get("score_breakdown", {}),
                "perfect_case": grade.get("perfect_case"),
                "minimum_task_runnable": grade.get("minimum_task_runnable"),
                "minimum_task_runnable_reasons": grade.get("minimum_task_runnable_reasons", []),
                "missing_required_keys": grade.get("missing_required_keys", {}),
                "extra_keys": grade.get("extra_keys", []),
                "default_imputed_keys": grade.get("default_imputed_keys", {}),
                "optional_missed_keys": [
                    item.get("parameter")
                    for item in (grade.get("optional_match", {}) or {}).get("items", [])
                    if not item.get("matched") and item.get("parameter")
                ],
            }
            for grade in grades
        ],
    }


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "REPO_ROOT",
    "build_case_metadata",
    "build_scoring_payload",
    "dump_json",
    "enabled_models",
    "ensure_output_root",
    "extract_incar_from_response",
    "fetch_mp_seed_data",
    "fetch_seed_data",
    "fetch_local_seed_data",
    "apply_reference_adjustments",
    "incar_text_from_dict",
    "invoke_model",
    "load_json",
    "load_problem_csv",
    "missing_generation_grade",
    "mentioned_scoring_keys_union",
    "normalize_incar_dict",
    "prompt_messages_for_case",
    "resolve_input_path",
    "row_mentioned_scoring_keys",
    "row_source_kind",
    "score_generated_incar",
    "split_csv_list",
    "summarize_generation_grades",
    "to_poscar_string",
]
