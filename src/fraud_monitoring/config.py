"""Config loading with optional per-environment overlay.

configs/monitoring.yaml holds every default. configs/<env>.yaml may override any
subset of it; the two are deep-merged. `FRAUD_MONITORING_ENV` picks the overlay,
which is how the same code runs in dev, staging, and prod without a branch.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
BASE_CONFIG_PATH = CONFIG_DIR / "monitoring.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge. Override wins on scalars and lists."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(env: str | None = None, config_dir: Path = CONFIG_DIR) -> dict:
    with open(config_dir / "monitoring.yaml") as f:
        config = yaml.safe_load(f)

    env = env or os.environ.get("FRAUD_MONITORING_ENV") or config.get("environment", "dev")
    overlay_path = config_dir / f"{env}.yaml"
    if overlay_path.exists():
        with open(overlay_path) as f:
            overlay = yaml.safe_load(f) or {}
        config = _deep_merge(config, overlay)
    config["environment"] = env
    return config


def resolve(config: dict, dotted_key: str, default: Any = None) -> Any:
    """resolve(cfg, "drift.psi.alert") -> 0.25. Keeps call sites readable."""
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def abspath(config: dict, dotted_key: str) -> Path:
    """Config paths are repo-relative; make them absolute for file IO."""
    value = resolve(config, dotted_key)
    if value is None:
        raise KeyError(f"No path configured at {dotted_key!r}")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


CONFIG = load_config()
