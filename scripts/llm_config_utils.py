#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark_utils import load_json


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "llm_benchmark_config.json"


def load_llm_benchmark_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or default_config_path()
    payload = load_json(config_path)
    validate_llm_benchmark_config(payload, config_path)
    return payload


def validate_llm_benchmark_config(payload: dict[str, Any], config_path: Path | None = None) -> None:
    def fail(message: str) -> None:
        prefix = f"{config_path}: " if config_path else ""
        raise ValueError(prefix + message)

    if not isinstance(payload, dict):
        fail("config must be a JSON object")

    materials_project = payload.get("materials_project")
    if not isinstance(materials_project, dict):
        fail("materials_project must be an object")

    if "api_key_env" in materials_project or "use_env_first" in materials_project:
        fail("materials_project must not define environment-variable-based key settings")

    for required_key in ("api_key", "endpoint"):
        if required_key not in materials_project:
            fail(f"materials_project.{required_key} is required")

    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict):
        fail("benchmark must be an object")

    for required_key in ("static_benchmark_root", "dataset_root", "default_prompt_style"):
        if required_key not in benchmark:
            fail(f"benchmark.{required_key} is required")

    models = payload.get("models")
    if not isinstance(models, list) or not models:
        fail("models must be a non-empty list")

    seen_names: set[str] = set()
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            fail(f"models[{index}] must be an object")

        for forbidden_key in ("api_key_env", "use_env_first"):
            if forbidden_key in model:
                fail(f"models[{index}] must not define {forbidden_key}")

        for required_key in ("name", "enabled", "provider", "access_method", "model_id"):
            if required_key not in model:
                fail(f"models[{index}].{required_key} is required")

        name = model["name"]
        if not isinstance(name, str) or not name:
            fail(f"models[{index}].name must be a non-empty string")
        if name in seen_names:
            fail(f"duplicate model name: {name}")
        seen_names.add(name)

        if not isinstance(model["enabled"], bool):
            fail(f"models[{index}].enabled must be a boolean")
        if not isinstance(model.get("headers", {}), dict):
            fail(f"models[{index}].headers must be an object")
        if not isinstance(model.get("default_params", {}), dict):
            fail(f"models[{index}].default_params must be an object")
        if not isinstance(model.get("use_proxy", False), bool):
            fail(f"models[{index}].use_proxy must be a boolean when provided")
        if "proxy_url" in model and not isinstance(model.get("proxy_url"), str):
            fail(f"models[{index}].proxy_url must be a string when provided")

        vertexai = model.get("vertexai", False)
        if not isinstance(vertexai, bool):
            fail(f"models[{index}].vertexai must be a boolean when provided")

        access_method = model.get("access_method")

        if access_method == "third_party_via_vertex":
            for required_key in ("endpoint", "region", "project_id"):
                value = model.get(required_key)
                if not isinstance(value, str) or not value.strip():
                    fail(f"models[{index}].{required_key} is required when access_method=third_party_via_vertex")
            if "api_key" in model and not isinstance(model.get("api_key"), str):
                fail(f"models[{index}].api_key must be a string when provided")
            if "base_url" in model and not isinstance(model.get("base_url"), str):
                fail(f"models[{index}].base_url must be a string when provided")
            continue

        if vertexai:
            if model.get("access_method") not in {"openai_sdk", "openai_compatible_http", "google_genai_sdk"}:
                fail(
                    f"models[{index}] with vertexai=true must use access_method openai_sdk, openai_compatible_http, or google_genai_sdk"
                )
            if model.get("access_method") in {"openai_sdk", "openai_compatible_http"}:
                for required_key in ("project", "location"):
                    value = model.get(required_key)
                    if not isinstance(value, str) or not value.strip():
                        fail(f"models[{index}].{required_key} is required when vertexai=true and access_method is OpenAI-compatible")
            else:
                for optional_key in ("project", "location"):
                    if optional_key in model and not isinstance(model.get(optional_key), str):
                        fail(f"models[{index}].{optional_key} must be a string when provided")
            if "api_key" in model and not isinstance(model.get("api_key"), str):
                fail(f"models[{index}].api_key must be a string when provided")
            if "base_url" in model and not isinstance(model.get("base_url"), str):
                fail(f"models[{index}].base_url must be a string when provided")
        else:
            for required_key in ("api_key", "base_url"):
                value = model.get(required_key)
                if not isinstance(value, str) or not value:
                    fail(f"models[{index}].{required_key} is required")


__all__ = ["default_config_path", "load_llm_benchmark_config", "validate_llm_benchmark_config"]
