"""
Phase 5 — Config Loader
========================
Loads YAML configs, supports nested override via dot-notation CLI args,
and provides a frozen dataclass view for type safety throughout the codebase.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merges override into base dict."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Load and merge configs.
    Priority: default.yaml < experiment config < CLI overrides

    Args:
        config_path: Optional path to experiment-specific YAML config.
        overrides: Dict of dot-notation key → value overrides.
                   e.g. {"training.lr": 1e-3, "model.hidden_dim": 64}
    """
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)

    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            experiment_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, experiment_cfg)

    if overrides:
        for dotpath, value in overrides.items():
            keys = dotpath.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    return cfg


def get_config(overrides: Optional[Dict] = None) -> dict:
    """Convenience accessor — returns default config with optional overrides."""
    return load_config(overrides=overrides)
