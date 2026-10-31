"""The optional LLM naming call, tested with a fake OpenAI client (no network, no key, no cost)."""
from __future__ import annotations

import sys
import types

import pytest
from PIL import Image

from sketch2shape.config import load_config
from sketch2shape.contracts import SketchAsset
from sketch2shape.runstore import RunStore
from sketch2shape.stages.understand import name_with_llm, parse_naming, understand

GOOD = ('{"object": "Headphones", "details": "arched band, two round ear cups", '
        '"material": "matte black plastic", "confidence": 0.9}')


def fake_openai(behaviour: dict) -> types.ModuleType:
    """A stand-in for the openai package; behaviour maps model name to 'ok', 'missing', 'bad key' or 'junk'."""
    module = types.ModuleType("openai")
    module.APIStatusError = type("APIStatusError", (Exception,), {"status_code": 500})
    for name in ("NotFoundError", "BadRequestError", "PermissionDeniedError", "AuthenticationError", "RateLimitError"):
        setattr(module, name, type(name, (module.APIStatusError,), {}))
    module.APIConnectionError = type("APIConnectionError", (Exception,), {})
    module.calls = []

    class Completions:
        def create(self, model, messages, response_format):
            module.calls.append(model)
            assert response_format == {"type": "json_object"}
            assert messages[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
            what = behaviour[model]
            if what == "missing":
                raise module.NotFoundError(model)
            if what == "bad key":
                raise module.AuthenticationError("invalid key")
            content = GOOD if what == "ok" else "I cannot tell."
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])

    class OpenAI:
        def __init__(self, **options):
            self.chat = types.SimpleNamespace(completions=Completions())

    module.OpenAI = OpenAI
    return module


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides={"runs_dir": str(tmp_path / "runs"),
                                  "understand.llm.openai_models": ["model-a", "model-b"]})


@pytest.fixture
def sketch_png(tmp_path):
    path = tmp_path / "lineart.png"
    Image.new("RGB", (64, 64), "white").save(path)
    return path


def use_openai(monkeypatch, behaviour):
    module = fake_openai(behaviour)
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return module


def test_openai_names_the_sketch_and_skips_unavailable_models(cfg, sketch_png, monkeypatch):
    module = use_openai(monkeypatch, {"model-a": "missing", "model-b": "ok"})
    named, note = name_with_llm(sketch_png, cfg)
    assert note == "" and module.calls == ["model-a", "model-b"]
    assert named["label"] == "headphones" and named["source"] == "openai:model-b"
    assert "arched band" in named["details"]


def test_a_rejected_key_falls_back_to_clip(cfg, sketch_png, monkeypatch):
    use_openai(monkeypatch, {"model-a": "bad key", "model-b": "ok"})
    named, note = name_with_llm(sketch_png, cfg)
    assert named is None and "rejected" in note


def test_an_unusable_answer_falls_back_to_clip(cfg, sketch_png, monkeypatch):
    use_openai(monkeypatch, {"model-a": "junk", "model-b": "ok"})
    named, note = name_with_llm(sketch_png, cfg)
    assert named is None and "no usable name" in note


def test_no_key_means_clip(cfg, sketch_png, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    named, note = name_with_llm(sketch_png, cfg)
    assert named is None and "No LLM API key" in note


def test_understand_uses_the_llm_name_in_the_prompt(cfg, sketch_png, monkeypatch, tmp_path):
    use_openai(monkeypatch, {"model-a": "ok", "model-b": "ok"})
    sketch = SketchAsset(sketch_png, sketch_png, sketch_png, 0.05, "upload")
    intent = understand(models=None, sketch=sketch, user_prompt="", cfg=cfg, store=RunStore(tmp_path / "r"))
    assert intent.source == "openai:model-a" and intent.label == "headphones"
    assert intent.prompt.startswith("a headphones, matte black plastic, arched band, two round ear cups")
    # a typed description decides appearance: the LLM's material suggestion is not added
    typed = understand(models=None, sketch=sketch, user_prompt="red leather headphones", cfg=cfg,
                       store=RunStore(tmp_path / "r2"))
    assert "matte black plastic" not in typed.prompt and typed.prompt.startswith("red leather headphones")


def test_parse_naming_accepts_fenced_json():
    named = parse_naming("```json\n" + GOOD + "\n```")
    assert named["label"] == "headphones" and named["confidence"] == 0.9
    assert parse_naming("no json here") is None
