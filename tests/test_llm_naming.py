"""The optional vision-LLM calls (naming, image check), tested with a fake OpenAI client (no network, no key, no cost)."""
from __future__ import annotations

import sys
import types

import pytest
from PIL import Image

from sketch2shape.config import load_config
from sketch2shape.contracts import Intent, SketchAsset
from sketch2shape.runstore import RunStore
from sketch2shape.stages.generate import sketch_control
from sketch2shape.stages.understand import name_with_llm, parse_naming, realistic, understand

GOOD = ('{"object": "Headphones", "details": "arched band, two round ear cups", '
        '"material": "matte black plastic", "drawing": "clean", "kind": "object", "up": "top", "confidence": 0.9}')
ROUGH = GOOD.replace('"clean"', '"rough"')


def fake_openai(behaviour: dict) -> types.ModuleType:
    """A stand-in for the openai package; behaviour maps model name to 'ok', 'rough', 'missing', 'bad key',
    'junk' or a JSON reply to send as it is."""
    module = types.ModuleType("openai")
    module.APIStatusError = type("APIStatusError", (Exception,), {"status_code": 500})
    for name in ("NotFoundError", "BadRequestError", "PermissionDeniedError", "AuthenticationError", "RateLimitError"):
        setattr(module, name, type(name, (module.APIStatusError,), {}))
    module.APIConnectionError = type("APIConnectionError", (Exception,), {})
    module.calls, module.texts, module.images = [], [], []

    class Completions:
        def create(self, model, messages, response_format):
            module.calls.append(model)
            module.texts.append(messages[0]["content"][0]["text"])
            assert response_format == {"type": "json_object"}
            images = [part["image_url"]["url"] for part in messages[0]["content"] if part["type"] == "image_url"]
            assert images and all(url.startswith("data:image/png;base64,") for url in images)
            module.images.append(len(images))
            what = behaviour[model]
            if what == "missing":
                raise module.NotFoundError(model)
            if what == "bad key":
                raise module.AuthenticationError("invalid key")
            content = {"ok": GOOD, "rough": ROUGH}.get(what, what if what.startswith("{") else "I cannot tell.")
            if "red leather" in module.texts[-1]:       # like the real models: keep the colours the user typed
                content = content.replace("matte black plastic", "red leather")
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


def test_a_clean_drawing_gets_the_plain_product_photo_prompt(cfg, sketch_png, monkeypatch, tmp_path):
    use_openai(monkeypatch, {"model-a": "ok", "model-b": "ok"})
    sketch = SketchAsset(sketch_png, sketch_png, sketch_png, 0.05, "upload")
    intent = understand(models=None, sketch=sketch, user_prompt="", cfg=cfg, store=RunStore(tmp_path / "r"))
    g = cfg.generate
    assert intent.source == "openai:model-a" and intent.label == "headphones" and intent.drawing == "clean"
    # the comparison: GPT's colours and details added mistakes to clean drawings, so only the name is used
    assert intent.prompt == f"a headphones, {g.clean.prompt_suffix}"
    assert intent.negative_prompt == g.clean.negative_prompt
    typed = understand(models=None, sketch=sketch, user_prompt="red leather headphones", cfg=cfg,
                       store=RunStore(tmp_path / "r2"))
    assert typed.prompt == f"red leather headphones, {g.clean.prompt_suffix}"


def test_a_rough_doodle_gets_gpt_colours_and_realism(cfg, sketch_png, monkeypatch, tmp_path):
    module = use_openai(monkeypatch, {"model-a": "rough", "model-b": "rough"})
    sketch = SketchAsset(sketch_png, sketch_png, sketch_png, 0.05, "upload")
    intent = understand(models=None, sketch=sketch, user_prompt="", cfg=cfg, store=RunStore(tmp_path / "r"))
    g = cfg.generate
    assert intent.prompt.startswith("a headphones, matte black plastic, arched band, two round ear cups, "
                                    + g.realism.prompt)
    assert intent.negative_prompt == f"{g.negative_prompt}, {g.realism.negative_prompt}"
    # a typed description comes first and goes to the naming call, whose material keeps the user's colours
    typed = understand(models=None, sketch=sketch, user_prompt="red leather headphones", cfg=cfg,
                       store=RunStore(tmp_path / "r2"))
    assert '"red leather headphones"' in module.texts[-1]
    assert typed.prompt.startswith(f"red leather headphones, {g.realism.prompt}")      # "red leather" not repeated
    assert "matte black plastic" not in typed.prompt
    # a short typed description that names no colours (the cat case) still gets them from the naming call
    short = understand(models=None, sketch=sketch, user_prompt="headphones on a stand", cfg=cfg,
                       store=RunStore(tmp_path / "r3"))
    assert short.prompt.startswith(f"headphones on a stand, matte black plastic, {g.realism.prompt}")


def test_a_rough_doodle_gets_lighter_control(cfg, sketch_png, monkeypatch, tmp_path):
    use_openai(monkeypatch, {"model-a": "rough", "model-b": "ok"})
    sketch = SketchAsset(sketch_png, sketch_png, sketch_png, 0.05, "upload")
    rough = understand(models=None, sketch=sketch, user_prompt="", cfg=cfg, store=RunStore(tmp_path / "r"))
    g = cfg.generate
    assert rough.drawing == "rough" and g.realism.prompt in rough.prompt
    # lighter, shorter control until an image drifts from the lines; then every step follows them again
    assert sketch_control(rough, 0.7, False, g) == (g.rough.control_scale, g.rough.control_end)
    assert sketch_control(rough, 0.8, True, g) == (0.8, 1.0)
    for drawing in ("clean", ""):                             # clean, or no LLM judgement: unchanged control
        assert sketch_control(Intent("mug", 0.9, "clip", "a mug", "", drawing=drawing), 0.7, False, g) == (0.7, 1.0)
    # a typed request for a style turns the realism words off; the doodle still gets lighter control
    styled = understand(models=None, sketch=sketch, user_prompt="a cartoon cat", cfg=cfg, store=RunStore(tmp_path / "r2"))
    assert g.realism.prompt not in styled.prompt and styled.negative_prompt == g.negative_prompt
    assert sketch_control(styled, 0.7, False, g)[1] == g.rough.control_end


def test_only_style_words_turn_realism_off():
    for text in ("", "cat sleeping", "toyota car", "red leather headphones"):
        assert realistic(text), text
    for text in ("a cartoon cat", "Cartoonish dog", "toy car", "low-poly tree", "stone statue", "rendered vase"):
        assert not realistic(text), text


def test_parse_naming_accepts_fenced_json():
    named = parse_naming("```json\n" + GOOD + "\n```")
    assert named["label"] == "headphones" and named["confidence"] == 0.9 and named["drawing"] == "clean"
    assert parse_naming(GOOD.replace('"clean"', '"sloppy"'))["drawing"] == ""    # unknown answers count as unknown
    assert parse_naming("no json here") is None


def test_parse_naming_reads_kind_and_up_with_safe_defaults():
    named = parse_naming(GOOD.replace('"object", "up": "top"', '"scene", "up": "right"'))
    assert named["kind"] == "scene" and named["up"] == "right"
    odd = parse_naming('{"object": "mug", "kind": "thing", "up": "sideways"}')
    assert odd["kind"] == "" and odd["up"] == "top"          # unknown answers: treated as an upright object


def test_a_scene_becomes_a_diorama_with_light_sketch_control(cfg, sketch_png, monkeypatch, tmp_path):
    scene = GOOD.replace('"Headphones"', '"mountain valley with a cabin"').replace('"object", "up"', '"scene", "up"')
    use_openai(monkeypatch, {"model-a": scene, "model-b": "ok"})
    sketch = SketchAsset(sketch_png, sketch_png, sketch_png, 0.05, "upload")
    intent = understand(models=None, sketch=sketch, user_prompt="", cfg=cfg, store=RunStore(tmp_path / "r"))
    g = cfg.generate
    assert intent.kind == "scene"
    assert intent.prompt.startswith("a miniature diorama of a mountain valley with a cabin, matte black plastic, "
                                    "on a small round base")
    assert "arched band" not in intent.prompt                 # a scene's parts are left to the sketch
    assert intent.negative_prompt.startswith(g.scene.negative_prompt)
    assert sketch_control(intent, 0.7, True, g) == (g.scene.control_scale, g.scene.control_end)


def test_turn_upright_rotates_a_sideways_sketch(tmp_path):
    import numpy as np
    from sketch2shape.stages.preprocess import turn_upright
    bar = np.zeros((8, 8), np.uint8)
    bar[:, 7] = 255                                           # the drawing's top is at the right edge
    paths = [tmp_path / f"{name}.png" for name in ("lineart", "control", "mask")]
    for path in paths:
        Image.fromarray(bar).save(path)
    assert turn_upright(SketchAsset(*paths, 0.1, "upload"), "right") == 1
    for path in paths:
        assert (np.array(Image.open(path))[0] == 255).all()   # ...and is now at the top
    assert turn_upright(SketchAsset(*paths, 0.1, "upload"), "top") == 0


# ------------------------------------------------------------ the vision-LLM image check (gate 1)

def candidates_for(tmp_path, n=2, round_=0):
    from sketch2shape.contracts import Candidate
    paths = []
    for i in range(n):
        path = tmp_path / f"05_candidate_r{round_}_{i}.png"
        Image.new("RGB", (32, 32), "white").save(path)
        paths.append(path)
    return [Candidate(path=p, seed=42, index=i, round=round_, control_scale=0.7) for i, p in enumerate(paths)]


def test_the_image_check_picks_one_and_sends_the_sketch_and_every_photo(cfg, sketch_png, monkeypatch, tmp_path):
    from sketch2shape.stages.image_check import check_images
    module = use_openai(monkeypatch, {"model-a": '{"best": "B", "problems": "A: ball inside a ring", "avoid": ""}',
                                      "model-b": "ok"})
    sketch = SketchAsset(sketch_png, sketch_png, sketch_png, 0.05, "upload")
    found = candidates_for(tmp_path)
    store = RunStore(tmp_path / "r")
    verdict = check_images(sketch, Intent("soccer ball", 0.9, "openai", "", "", kind="object"), found, cfg, store)
    assert verdict["best"] is found[1] and [c.judged for c in found] == ["rejected", "picked"]
    assert module.images[-1] == 3                             # the sketch and both photos
    assert "one soccer ball" in module.texts[-1] and store.manifest["image_checks"][0]["picked"] == found[1].path.name


def test_the_image_check_can_reject_everything_and_say_what_to_avoid(cfg, sketch_png, monkeypatch, tmp_path):
    from sketch2shape.stages.image_check import check_images
    use_openai(monkeypatch, {"model-a": '{"best": "none", "problems": "both framed", '
                                        '"avoid": "Circular frame around the ball, and a rim!"}', "model-b": "ok"})
    sketch = SketchAsset(sketch_png, sketch_png, sketch_png, 0.05, "upload")
    found = candidates_for(tmp_path)
    verdict = check_images(sketch, Intent("soccer ball", 0.9, "openai", "", ""), found, cfg, RunStore(tmp_path / "r"))
    assert verdict["best"] is None and all(c.judged == "rejected" for c in found)
    assert verdict["avoid"] == "circular frame around the ball and"   # at most 6 plain words


def test_without_a_usable_answer_the_check_steps_aside(cfg, sketch_png, monkeypatch, tmp_path):
    from sketch2shape.stages.image_check import check_images
    use_openai(monkeypatch, {"model-a": "junk", "model-b": "ok"})
    sketch = SketchAsset(sketch_png, sketch_png, sketch_png, 0.05, "upload")
    found = candidates_for(tmp_path)
    store = RunStore(tmp_path / "r")
    assert check_images(sketch, Intent("ball", 0.9, "openai", "", ""), found, cfg, store) is None
    assert all(c.judged == "" for c in found) and "Image check skipped" in store.manifest["warnings"][-1]


def score_step(cfg, tmp_path, monkeypatch, verdicts, totals, kind="object"):
    """The pipeline's gate-1 handler with a scripted image check and scripted line scores."""
    from sketch2shape import pipeline as pipeline_module
    from sketch2shape.pipeline import Sketch2Shape

    def fake_check(sketch, intent, fresh, cfg_, store):
        verdict = verdicts.pop(0)
        if verdict is None:
            return None
        picked = fresh[verdict["best"]] if verdict["best"] is not None else None
        for c in fresh:
            c.judged = "picked" if c is picked else "rejected"
        return {"best": picked, "problems": "", "avoid": verdict.get("avoid", ""), "source": "fake"}

    def fake_scores(models, sketch, intent, found, cfg_, store):
        for c in found:
            c.scores, c.total = {"edges": 1.0}, totals[c.path.name]

    monkeypatch.setattr(pipeline_module, "check_images", fake_check)
    monkeypatch.setattr(pipeline_module, "score_candidates", fake_scores)
    pipe = Sketch2Shape(cfg, models=object())
    ctx = pipe.new_context(image=None)
    ctx.intent = Intent("ball", 0.9, "openai", "a ball", "base negative", kind=kind)
    ctx.negative_base = "base negative"
    ctx.candidates = candidates_for(tmp_path)
    return pipe, ctx


class RecordingModels:
    """Records the worker calls the pipeline makes around gate 1."""
    def __init__(self):
        self.calls = []

    def release_kept(self):
        self.calls.append("release_kept")

    def load_early(self, task, role, store, **load):
        self.calls.append(f"load_early:{task}")

    def drop_early(self):
        self.calls.append("drop_early")


def test_gate_one_passes_the_checked_pick_when_it_follows_the_lines(cfg, tmp_path, monkeypatch):
    pipe, ctx = score_step(cfg, tmp_path, monkeypatch, [{"best": 1}],
                           {"05_candidate_r0_0.png": 0.9, "05_candidate_r0_1.png": 0.6})
    pipe.models = RecordingModels()
    assert pipe._score(ctx) == "pass"
    assert ctx.best is ctx.candidates[1]                   # the check's pick, not the higher line score
    assert ctx.candidates[0].scores == {}                  # a rejected image is not even cut out
    # once picked, SDXL makes way and the shape model starts loading during the line check
    assert pipe.models.calls == ["release_kept", "load_early:shape"]


def test_a_failed_pick_drops_the_early_shape_worker_before_retrying(cfg, tmp_path, monkeypatch):
    pipe, ctx = score_step(cfg, tmp_path, monkeypatch, [{"best": 0}],
                           {"05_candidate_r0_0.png": 0.1, "05_candidate_r0_1.png": 0.9})
    pipe.models = RecordingModels()
    assert pipe._score(ctx) == "retry"
    assert pipe.models.calls == ["release_kept", "load_early:shape", "drop_early"]


def test_gate_one_retries_with_avoid_words_when_everything_is_rejected(cfg, tmp_path, monkeypatch):
    pipe, ctx = score_step(cfg, tmp_path, monkeypatch, [{"best": None, "avoid": "ring around the ball"}], {})
    assert pipe._score(ctx) == "retry"
    assert ctx.intent.negative_prompt == "ring around the ball, base negative"
    assert not ctx.strict and ctx.control_scale == float(cfg.generate.control_scale)   # same sketch control


def test_gate_one_follows_the_lines_harder_when_the_pick_drifted(cfg, tmp_path, monkeypatch):
    pipe, ctx = score_step(cfg, tmp_path, monkeypatch, [{"best": 0}],
                           {"05_candidate_r0_0.png": 0.1, "05_candidate_r0_1.png": 0.9})
    assert pipe._score(ctx) == "retry" and ctx.strict
    assert ctx.control_scale > float(cfg.generate.control_scale)


def test_gate_one_accepts_a_checked_diorama_whatever_its_line_score(cfg, tmp_path, monkeypatch):
    pipe, ctx = score_step(cfg, tmp_path, monkeypatch, [{"best": 0}],
                           {"05_candidate_r0_0.png": 0.05, "05_candidate_r0_1.png": 0.05}, kind="scene")
    assert pipe._score(ctx) == "pass"


def test_gate_one_without_the_check_works_as_before(cfg, tmp_path, monkeypatch):
    pipe, ctx = score_step(cfg, tmp_path, monkeypatch, [None],
                           {"05_candidate_r0_0.png": 0.5, "05_candidate_r0_1.png": 0.7})
    assert pipe._score(ctx) == "pass" and ctx.best is ctx.candidates[1]
