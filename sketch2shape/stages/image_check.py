"""Gate 1, first check: the vision LLM picks the generated image the 3D model should be built from.

The line scores (score.py) only measure how closely an image follows the sketch, so a ball drawn inside
a ring, or a cabin painted with the hills across its walls in front of a forest, passed them, and the
3D model then faithfully built the drum or the slab. This check sends the sketch and the round's images
to the vision LLM, which picks the one showing a single, complete, real object (or diorama) alone on a
plain background, or rejects them all with a few words on what to avoid next time. Without an LLM key,
or if the call fails, it returns None and the line scores decide alone, as before.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from ..llm import ask, parse_json

LETTERS = "ABCDEFGH"

CHECK_PROMPT = (
    "Below are a hand-drawn sketch and {n} photo(s) generated from it, labelled {letters}. A 3D model will be "
    "built from the photo you pick, so a usable photo shows {subject} and nothing else: complete, with no part "
    "cut off at the edges, alone on a plain light background, with real depth (not a flat plaque, relief, "
    "sticker or drawing), with no frame, ring, border, stand, second object or background scenery around it, "
    "and with roughly the overall shape of the sketch. In best, give the letter of the best usable photo, or "
    "'none' if none is usable. In problems, say in a few words what is wrong with each photo you did not "
    "pick. In avoid, give at most 6 words the image generator should avoid next time, for example "
    "'circular frame around the ball', or '' if nothing. Reply only with a JSON object with the keys best, "
    "problems and avoid."
)


def check_images(sketch, intent, candidates, cfg, store) -> dict | None:
    """Returns {"best": the picked candidate or None if all were rejected, "problems", "avoid", "source"},
    or None when the check could not run. Also marks each candidate "picked" or "rejected"."""
    if not candidates:
        return None
    letters = list(LETTERS[:len(candidates)])     # a list: an empty answer must not match ("" is in any str)
    if intent.kind == "scene":
        subject = f"one miniature diorama of {intent.label} on a small base"
    else:
        subject = f"one {intent.label}" if intent.label else "the one object the sketch shows"
    parts = [CHECK_PROMPT.format(n=len(candidates), letters=", ".join(letters), subject=subject),
             "Sketch:", Path(sketch.lineart_path)]
    for letter, candidate in zip(letters, candidates):
        parts += [f"Photo {letter}:", Path(candidate.path)]
    schema = {"type": "object", "additionalProperties": False, "required": ["best", "problems", "avoid"],
              "properties": {"best": {"type": "string", "enum": [*letters, "none"]},
                             "problems": {"type": "string"}, "avoid": {"type": "string"}}}
    started = time.monotonic()
    text, source, note = ask(parts, cfg, schema)
    store.manifest.setdefault("score_timings_s", []).append({"image_check": round(time.monotonic() - started, 1)})
    if text is None:
        if note:
            store.warn(f"Image check skipped: {note}.")
        return None
    data = parse_json(text) or {}
    best = str(data.get("best", "")).strip().upper()
    if best not in letters and best != "NONE":
        store.warn(f"Image check skipped: {source} gave no usable answer.")
        return None
    picked = candidates[letters.index(best)] if best in letters else None
    for candidate in candidates:
        candidate.judged = "picked" if candidate is picked else "rejected"
    verdict = {"best": picked, "problems": str(data.get("problems", "")).strip()[:300],
               "avoid": short_phrase(data.get("avoid", "")), "source": source}
    store.manifest.setdefault("image_checks", []).append(
        {"round": candidates[0].round, "picked": picked.path.name if picked else None,
         "problems": verdict["problems"], "avoid": verdict["avoid"], "by": source})
    return verdict


def short_phrase(text, words: int = 6) -> str:
    """At most `words` plain words, so the negative prompt stays within CLIP's 77 tokens."""
    return " ".join(re.sub(r"[^\w\s'-]", " ", str(text or "")).split()[:words]).lower()
