"""Loads config.yaml into attribute-style settings, with optional dotted overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    return value


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> SimpleNamespace:
    """Read the YAML file and apply overrides such as {"generate.candidates": 1}."""
    raw = yaml.safe_load(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    cfg = _namespace(copy.deepcopy(raw))
    cfg.raw = raw
    return cfg
