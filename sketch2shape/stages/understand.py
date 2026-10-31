"""Stage 2: name the sketched object and build the image-generation prompt.

With an LLM API key set, one vision call names the object and notes its visible shape features:
OpenAI when OPENAI_API_KEY is set, otherwise Claude when ANTHROPIC_API_KEY is set (config can force
either). Without a key, or on any API error or refusal, CLIP picks a name from a fixed list instead, so
the pipeline never depends on a key. A typed user prompt always wins over the guessed name.
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts import Intent
from ..vocab import OBJECTS

NAMING_PROMPT = (
    "This is a rough hand-drawn sketch of one object that will be turned into a 3D model. "
    "Name the object in one to three plain words, for example 'coffee mug', 'armchair' or 'sports car'. "
    "In details, list at most 12 words about its visible shape only, for example 'tall tapered body, round "
    "handle on the right'. In material, give one short phrase for a plausible real-world material and "
    "finish for this object, for example 'white glazed ceramic', 'brushed steel' or 'soft grey plush "
    "fabric'. Set confidence between 0 and 1."
)

NAMING_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string"},
        "details": {"type": "string"},
        "material": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["object", "details", "material", "confidence"],
    "additionalProperties": False,
}


JSON_REPLY = (" Reply only with a JSON object with the keys object (string), details (string), "
              "material (string) and confidence (number).")


def parse_naming(text: str) -> dict | None:
    """The naming answer from a model's JSON reply, or None if it holds no usable name."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    name = str(data.get("object", "")).strip().lower()
    if not name:
        return None
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return {"label": name, "details": str(data.get("details", "")).strip(),
            "material": str(data.get("material", "")).strip()[:60], "confidence": confidence}


def name_with_llm(image_path: Path, cfg) -> tuple[dict | None, str]:
    """Pick the provider from config and the keys present. Returns (result or None, note for the run log)."""
    llm = cfg.understand.llm
    if not llm.enabled:
        return None, ""
    provider = llm.provider
    if provider == "auto":
        provider = ("openai" if os.environ.get("OPENAI_API_KEY")
                    else "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else None)
    if provider == "openai":
        return openai_name(image_path, cfg)
    if provider == "anthropic":
        return claude_name(image_path, cfg)
    return None, "No LLM API key set (OPENAI_API_KEY or ANTHROPIC_API_KEY); using CLIP instead."


def openai_name(image_path: Path, cfg) -> tuple[dict | None, str]:
    """Ask an OpenAI vision model what the sketch shows, trying the configured models in order."""
    llm = cfg.understand.llm
    if not os.environ.get("OPENAI_API_KEY"):
        return None, "No OPENAI_API_KEY set; using CLIP instead."
    try:
        import openai
    except ImportError:
        return None, "The openai package is not installed; using CLIP instead."

    client = openai.OpenAI(timeout=float(llm.timeout_s), max_retries=1)
    image_url = "data:image/png;base64," + base64.standard_b64encode(Path(image_path).read_bytes()).decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": NAMING_PROMPT + JSON_REPLY},
        {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},   # low detail: cheap, enough for a sketch
    ]}]
    skipped = []
    for model in llm.openai_models:
        try:
            response = client.chat.completions.create(model=model, messages=messages,
                                                      response_format={"type": "json_object"})
        except (openai.NotFoundError, openai.BadRequestError, openai.PermissionDeniedError) as exc:
            skipped.append(f"{model}: {type(exc).__name__}")       # model not available to this key: try the next
            continue
        except openai.AuthenticationError:
            return None, "OpenAI rejected the API key; using CLIP instead."
        except openai.RateLimitError:
            return None, "OpenAI rate limit or quota reached; using CLIP instead."
        except openai.APIConnectionError:
            return None, "Could not reach the OpenAI API; using CLIP instead."
        except openai.APIStatusError as exc:
            return None, f"OpenAI API error {exc.status_code}; using CLIP instead."
        named = parse_naming(response.choices[0].message.content)
        if not named:
            return None, f"{model} gave no usable name; using CLIP instead."
        named["source"] = f"openai:{model}"
        return named, ""
    return None, f"No configured OpenAI model was usable ({'; '.join(skipped)}); using CLIP instead."


def claude_name(image_path: Path, cfg) -> tuple[dict | None, str]:
    """Ask Claude what the sketch shows. Returns (result or None, note for the run log)."""
    llm = cfg.understand.llm
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, "No ANTHROPIC_API_KEY set; using CLIP instead."
    try:
        import anthropic
    except ImportError:
        return None, "The anthropic package is not installed; using CLIP instead."

    client = anthropic.Anthropic(timeout=float(llm.timeout_s), max_retries=1)
    image_data = base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")
    request = dict(
        model=llm.anthropic_model,
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
    named = parse_naming(text)
    if not named:
        return None, "Claude gave no usable name; using CLIP instead."
    named["source"] = f"anthropic:{llm.anthropic_model}"
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


def build_prompt(label: str | None, details: str, user_prompt: str, cfg, material: str = "") -> str:
    if user_prompt:                         # the user's words decide appearance; the LLM's material is not used
        subject = user_prompt
        if label and label not in user_prompt.lower():
            subject = f"{with_article(label)}, {user_prompt}"
    elif label:
        subject = with_article(label) + "".join(f", {part}" for part in (material, details) if part)
    else:
        subject = "a single everyday object"
    return f"{subject}, {cfg.generate.prompt_suffix}"


def understand(models, sketch, user_prompt: str, cfg, store) -> Intent:
    named, note = name_with_llm(sketch.lineart_path, cfg)
    if note:
        store.warn(note)
    material = ""
    if named:
        label, details, confidence, source = named["label"], named["details"], named["confidence"], named["source"]
        material = named.get("material", "")
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
        prompt=build_prompt(label, details, user_prompt, cfg, material),
        negative_prompt=cfg.generate.negative_prompt, details=details,
    )
