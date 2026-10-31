"""The optional vision-LLM calls: naming the sketch (stage 2) and checking the generated images (gate 1).

One request sends text and images in order and gets one JSON object back. OpenAI is used when
OPENAI_API_KEY is set, otherwise Claude when ANTHROPIC_API_KEY is set (config can force either). A
missing key, an API error or a refusal returns no answer and a short note; the callers then fall back
(CLIP for naming, the line scores alone for the image check), so the pipeline never depends on a key.
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def provider(cfg) -> str | None:
    """"openai", "anthropic", or None when the call is switched off or no key is set."""
    llm = cfg.understand.llm
    if not llm.enabled:
        return None
    if llm.provider != "auto":
        return llm.provider
    return ("openai" if os.environ.get("OPENAI_API_KEY")
            else "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else None)


def parse_json(text: str) -> dict | None:
    """The first JSON object in a reply (models sometimes wrap it in a code fence), or None."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def ask(parts: list, cfg, schema: dict) -> tuple[str | None, str, str]:
    """Send the parts (strings, and image files as Paths) in order. Returns (reply text or None,
    "openai:<model>" or "anthropic:<model>", note saying why there is no reply)."""
    if not cfg.understand.llm.enabled:
        return None, "", ""
    name = provider(cfg)
    if name == "openai":
        return ask_openai(parts, cfg)
    if name == "anthropic":
        return ask_claude(parts, cfg, schema)
    return None, "", "No LLM API key set (OPENAI_API_KEY or ANTHROPIC_API_KEY)"


def _image_b64(path: Path) -> tuple[str, str]:
    return (MEDIA_TYPES.get(Path(path).suffix.lower(), "image/png"),
            base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8"))


def ask_openai(parts: list, cfg) -> tuple[str | None, str, str]:
    """An OpenAI vision model, trying the configured models in order until one is available to the key."""
    llm = cfg.understand.llm
    if not os.environ.get("OPENAI_API_KEY"):
        return None, "", "No OPENAI_API_KEY set"
    try:
        import openai
    except ImportError:
        return None, "", "The openai package is not installed"

    client = openai.OpenAI(timeout=float(llm.timeout_s), max_retries=1)
    content = []
    for part in parts:
        if isinstance(part, Path):
            media, data = _image_b64(part)          # low detail: cheap, and enough for sketches and checks
            content.append({"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}", "detail": "low"}})
        else:
            content.append({"type": "text", "text": part})
    messages = [{"role": "user", "content": content}]
    skipped = []
    for model in llm.openai_models:
        try:
            response = client.chat.completions.create(model=model, messages=messages,
                                                      response_format={"type": "json_object"})
        except (openai.NotFoundError, openai.BadRequestError, openai.PermissionDeniedError) as exc:
            skipped.append(f"{model}: {type(exc).__name__}")       # model not available to this key: try the next
            continue
        except openai.AuthenticationError:
            return None, "", "OpenAI rejected the API key"
        except openai.RateLimitError:
            return None, "", "OpenAI rate limit or quota reached"
        except openai.APIConnectionError:
            return None, "", "Could not reach the OpenAI API"
        except openai.APIStatusError as exc:
            return None, "", f"OpenAI API error {exc.status_code}"
        return response.choices[0].message.content, f"openai:{model}", ""
    return None, "", f"No configured OpenAI model was usable ({'; '.join(skipped)})"


def ask_claude(parts: list, cfg, schema: dict) -> tuple[str | None, str, str]:
    """Claude with a JSON-schema answer format."""
    llm = cfg.understand.llm
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, "", "No ANTHROPIC_API_KEY set"
    try:
        import anthropic
    except ImportError:
        return None, "", "The anthropic package is not installed"

    client = anthropic.Anthropic(timeout=float(llm.timeout_s), max_retries=1)
    content = []
    for part in parts:
        if isinstance(part, Path):
            media, data = _image_b64(part)
            content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
        else:
            content.append({"type": "text", "text": part})
    request = dict(
        model=llm.anthropic_model,
        max_tokens=4096,
        output_config={"effort": llm.effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": content}],
    )
    try:
        if llm.server_fallback:
            response = client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request)
        else:
            response = client.messages.create(**request)
    except anthropic.RateLimitError:
        return None, "", "Claude rate limit reached"
    except anthropic.APIStatusError as exc:
        return None, "", f"Claude API error {exc.status_code}"
    except anthropic.APIConnectionError:
        return None, "", "Could not reach the Claude API"
    except Exception as exc:  # e.g. an older SDK without these parameters
        return None, "", f"Claude call failed ({exc!r})"

    if response.stop_reason == "refusal":
        return None, "", "Claude declined the request"
    text = next((block.text for block in response.content if block.type == "text"), None)
    if not text:
        return None, "", "Claude returned no text"
    return text, f"anthropic:{llm.anthropic_model}", ""
