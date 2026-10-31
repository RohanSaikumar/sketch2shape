"""Stage 2: name the sketched object and build the image-generation prompt.

With an LLM API key set, one vision call (sketch2shape/llm.py) names the object, notes its shape, a
plausible real colour and surface, whether the drawing is clean or rough, whether it shows one object
or a whole scene, and which way is up. Without a key, or on any API error or refusal, CLIP picks a name
from a fixed list instead, so the pipeline never depends on a key. A typed user prompt always wins over
the guessed name; the call sees it, so its colour and material suggestion keeps what the user asked for.
The prompt asks for a photorealistic object unless the typed description asks for a style such as a
cartoon or a toy. A scene (a landscape, several separate things) is made as a miniature diorama.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts import Intent
from ..llm import ask, parse_json
from ..vocab import OBJECTS

NAMING_PROMPT = (
    "This is a hand-drawn sketch that will be turned into a realistic 3D model. "
    "Name the object in one to three plain words, for example 'coffee mug', 'armchair' or 'sports car'. "
    "In details, list at most 10 words about the real object's shape and pose as the sketch shows it, for "
    "example 'tall tapered body, round handle on the right'; describe the object, not the drawing style. "
    "In material, give at most 8 words on the real colours, pattern and surface it would most plausibly have "
    "in real life, for example 'white glazed ceramic', 'brushed steel and black rubber' or 'orange tabby "
    "fur, white chest'. Shaded or hatched areas are dark colours or materials: say so in material (for "
    "example 'black pentagon panels'), never treat them as texture, grooves or a rim. "
    "In drawing, answer 'clean' if the lines already show the real object's true shape and proportions, or "
    "'rough' if it is a quick, wobbly, childlike or cartoon-style doodle (odd proportions, a smiley face, "
    "lumpy outlines) that the real object should not copy line for line. "
    "In kind, answer 'object' if the sketch shows one thing, or 'scene' if it shows a landscape or a view "
    "with several separate things (for example hills, trees and a house); for a scene, name the whole view "
    "in object (for example 'mountain valley with a cabin') and list its main parts in details. "
    "In up, say which edge of the image the top of the drawing points to: 'top' if it is upright or has no "
    "clear up direction (like a ball), otherwise 'right', 'bottom' or 'left' (a photo taken sideways or "
    "upside down). Set confidence between 0 and 1."
)

NAMING_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string"},
        "details": {"type": "string"},
        "material": {"type": "string"},
        "drawing": {"type": "string", "enum": ["clean", "rough"]},
        "kind": {"type": "string", "enum": ["object", "scene"]},
        "up": {"type": "string", "enum": ["top", "right", "bottom", "left"]},
        "confidence": {"type": "number"},
    },
    "required": ["object", "details", "material", "drawing", "kind", "up", "confidence"],
    "additionalProperties": False,
}

USER_TEXT = (" The user also typed this description: \"{}\". In material, keep every colour, material or "
             "style it names and only fill in what it leaves open.")

JSON_REPLY = (" Reply only with a JSON object with the keys object (string), details (string), "
              "material (string), drawing ('clean' or 'rough'), kind ('object' or 'scene'), up ('top', "
              "'right', 'bottom' or 'left') and confidence (number).")

# A typed description asking for one of these looks turns the realism words off.
STYLE_WORDS = re.compile(r"\b(cartoon\w*|anime|chibi|toys?|figurines?|statues?|sculptures?|lego|origami|"
                         r"low[- ]?poly|voxel\w*|styli[sz]ed|render\w*)\b", re.I)


def naming_text(user_prompt: str) -> str:
    return NAMING_PROMPT + (USER_TEXT.format(user_prompt.strip()) if user_prompt.strip() else "")


def realistic(user_prompt: str) -> bool:
    """False when a typed description asks for a non-realistic look, such as a cartoon or a toy."""
    return not STYLE_WORDS.search(user_prompt or "")


def parse_naming(text: str) -> dict | None:
    """The naming answer from a model's JSON reply, or None if it holds no usable name."""
    data = parse_json(text)
    name = str((data or {}).get("object", "")).strip().lower()
    if not name:
        return None
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5

    def choice(key: str, allowed: tuple, default: str = "") -> str:
        value = str(data.get(key, "")).strip().lower()
        return value if value in allowed else default

    return {"label": name, "details": str(data.get("details", "")).strip(),
            "material": str(data.get("material", "")).strip()[:60],
            "drawing": choice("drawing", ("clean", "rough")), "kind": choice("kind", ("object", "scene")),
            "up": choice("up", ("top", "right", "bottom", "left"), "top"), "confidence": confidence}


def name_with_llm(image_path: Path, cfg, user_prompt: str = "") -> tuple[dict | None, str]:
    """Ask the vision LLM what the sketch shows. Returns (result or None, note for the run log)."""
    text, source, note = ask([naming_text(user_prompt) + JSON_REPLY, Path(image_path)], cfg, NAMING_SCHEMA)
    if text is None:
        return None, f"{note}; using CLIP instead." if note else ""
    named = parse_naming(text)
    if not named:
        return None, f"{source} gave no usable name; using CLIP instead."
    named["source"] = source
    return named, ""


def clip_zero_shot(models, images: list[Image.Image], labels: list[str], template: str) -> np.ndarray:
    """Probability of each label for each image, from CLIP image-text similarity."""
    import torch
    model, processor = models.clip()
    texts = [template.format(with_article(label)) for label in labels]
    try:                                    # fast path: label embeddings are computed once and cached
        text = models.clip_text_features(texts)
        image = models.clip_image_features(images)
        with torch.no_grad():
            logits = model.logit_scale.exp() * image @ text.T    # same as CLIPModel's logits_per_image
    except Exception:                       # e.g. a transformers version with different internals
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image
    return logits.float().softmax(dim=-1).cpu().numpy()


def with_article(noun: str) -> str:
    return f"{'an' if noun[:1] in 'aeiou' else 'a'} {noun}"


def build_prompt(label: str | None, details: str, user_prompt: str, cfg, material: str = "",
                 kind: str = "") -> str:
    g = cfg.generate
    if user_prompt:                         # the user's words come first; the naming call kept their colours
        subject = user_prompt
        if label and label not in user_prompt.lower():
            subject = f"{with_article(label)}, {user_prompt}"
        if material and material.lower() not in user_prompt.lower():
            subject += f", {material}"
    elif label:
        # a scene's parts are many words; its name and colours are enough next to the sketch
        parts = (material,) if kind == "scene" else (material, details)
        subject = with_article(label) + "".join(f", {part}" for part in parts if part)
    else:
        subject = "a single everyday object"
    if kind == "scene":                     # a landscape is not one object: build it as a diorama
        subject = f"a miniature diorama of {subject}, on a small round base"
    realism = f"{g.realism.prompt}, " if realistic(user_prompt) else ""
    return f"{subject}, {realism}{g.prompt_suffix}"


def understand(models, sketch, user_prompt: str, cfg, store) -> Intent:
    named, note = name_with_llm(sketch.lineart_path, cfg, user_prompt)
    if note:
        store.warn(note)
    material, drawing, kind, up = "", "", "", ""
    if named:
        label, details, confidence, source = named["label"], named["details"], named["confidence"], named["source"]
        material, drawing, kind, up = named["material"], named["drawing"], named["kind"], named["up"]
    else:
        lineart = Image.open(sketch.lineart_path).convert("RGB")
        probs = clip_zero_shot(models, [lineart], OBJECTS, "a black and white sketch of {}")[0]
        best = int(probs.argmax())
        confidence, details = float(probs[best]), ""
        if confidence >= float(cfg.understand.clip_min_confidence):
            label, source = OBJECTS[best], "clip"
        else:
            label, source = None, "none"
    g = cfg.generate
    negative_prompt = g.scene.negative_prompt if kind == "scene" else g.negative_prompt
    if realistic(user_prompt):
        negative_prompt += f", {g.realism.negative_prompt}"
    return Intent(
        label=label, confidence=round(confidence, 3), source=source,
        prompt=build_prompt(label, details, user_prompt, cfg, material, kind),
        negative_prompt=negative_prompt, details=details, drawing=drawing, kind=kind, up=up,
    )
