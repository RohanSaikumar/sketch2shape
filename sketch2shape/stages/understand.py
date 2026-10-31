"""Stage 2: name the sketched object and build the image-generation prompt.

With ANTHROPIC_API_KEY set, one Claude call names the object and notes its visible shape features.
Without a key, or on any API error or refusal, CLIP picks a name from a fixed list instead, so the
pipeline never depends on the key. A typed user prompt always wins over the guessed name.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts import Intent
from ..vocab import OBJECTS

NAMING_PROMPT = (
    "This is a rough hand-drawn sketch of one object that will be turned into a 3D model. "
    "Name the object in one to three plain words, for example 'coffee mug', 'armchair' or 'sports car'. "
    "In details, list at most 12 words about its visible shape, for example 'tall tapered body, round "
    "handle on the right'. Do not invent colours or materials. Set confidence between 0 and 1."
)

NAMING_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string"},
        "details": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["object", "details", "confidence"],
    "additionalProperties": False,
}


def claude_name(image_path: Path, cfg) -> tuple[dict | None, str]:
    """Ask Claude what the sketch shows. Returns (result or None, note for the run log)."""
    llm = cfg.understand.llm
    if not llm.enabled:
        return None, ""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, "No ANTHROPIC_API_KEY set; using CLIP instead."
    try:
        import anthropic
    except ImportError:
        return None, "The anthropic package is not installed; using CLIP instead."

    client = anthropic.Anthropic(timeout=float(llm.timeout_s), max_retries=1)
    image_data = base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")
    request = dict(
        model=llm.model,
        max_tokens=4096,
        output_config={"effort": llm.effort, "format": {"type": "json_schema", "schema": NAMING_SCHEMA}},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
                {"type": "text", "text": NAMING_PROMPT},
            ],
        }],
    )
    try:
        if llm.server_fallback:
            response = client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request)
        else:
            response = client.messages.create(**request)
    except anthropic.RateLimitError:
        return None, "Claude rate limit reached; using CLIP instead."
    except anthropic.APIStatusError as exc:
        return None, f"Claude API error {exc.status_code}; using CLIP instead."
    except anthropic.APIConnectionError:
        return None, "Could not reach the Claude API; using CLIP instead."
    except Exception as exc:  # e.g. an older SDK without these parameters
        return None, f"Claude call failed ({exc!r}); using CLIP instead."

    if response.stop_reason == "refusal":
        return None, "Claude declined to name the sketch; using CLIP instead."
    text = next((block.text for block in response.content if block.type == "text"), None)
    if not text:
        return None, "Claude returned no text; using CLIP instead."
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, "Claude returned malformed JSON; using CLIP instead."
    name = str(data.get("object", "")).strip().lower()
    if not name:
        return None, "Claude could not name the object; using CLIP instead."
    return {"label": name, "details": str(data.get("details", "")).strip(),
            "confidence": float(data.get("confidence", 0.5))}, ""


def clip_zero_shot(models, images: list[Image.Image], labels: list[str], template: str) -> np.ndarray:
    """Probability of each label for each image, from CLIP image-text similarity."""
    import torch
    model, processor = models.clip()
    inputs = processor(text=[template.format(with_article(label)) for label in labels], images=images,
                       return_tensors="pt", padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image.float()
    return logits.softmax(dim=-1).cpu().numpy()


def with_article(noun: str) -> str:
    return f"{'an' if noun[:1] in 'aeiou' else 'a'} {noun}"


def build_prompt(label: str | None, details: str, user_prompt: str, cfg) -> str:
    if user_prompt:
        subject = user_prompt
        if label and label not in user_prompt.lower():
            subject = f"{with_article(label)}, {user_prompt}"
    elif label:
        subject = with_article(label) + (f", {details}" if details else "")
    else:
        subject = "a single everyday object"
    return f"{subject}, {cfg.generate.prompt_suffix}"


def understand(models, sketch, user_prompt: str, cfg, store) -> Intent:
    named, note = claude_name(sketch.lineart_path, cfg)
    if note:
        store.warn(note)
    if named:
        label, details, confidence, source = named["label"], named["details"], named["confidence"], "llm"
    else:
        lineart = Image.open(sketch.lineart_path).convert("RGB")
        probs = clip_zero_shot(models, [lineart], OBJECTS, "a black and white sketch of {}")[0]
        best = int(probs.argmax())
        confidence, details = float(probs[best]), ""
        if confidence >= float(cfg.understand.clip_min_confidence):
            label, source = OBJECTS[best], "clip"
        else:
            label, source = None, "none"
    return Intent(
        label=label, confidence=round(confidence, 3), source=source,
        prompt=build_prompt(label, details, user_prompt, cfg),
        negative_prompt=cfg.generate.negative_prompt, details=details,
    )
