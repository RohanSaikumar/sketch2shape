"""One folder per run: numbered intermediates plus a manifest.json that describes the whole run."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


class RunStore:
    def __init__(self, root: str | Path, run_id: str | None = None):
        self.run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.dir = Path(root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "run_id": self.run_id,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "transitions": [],
            "timings_s": {},
            "warnings": [],
        }

    def path(self, name: str) -> Path:
        return self.dir / name

    def save_image(self, image, name: str) -> Path:
        target = self.path(name)
        image.save(target)
        return target

    def warn(self, message: str) -> None:
        self.manifest["warnings"].append(message)

    def add_time(self, key: str, seconds: float) -> None:
        timings = self.manifest["timings_s"]
        timings[key] = round(timings.get(key, 0.0) + seconds, 2)

    def set(self, key: str, value) -> None:
        self.manifest[key] = value

    def write(self) -> Path:
        target = self.path("manifest.json")
        target.write_text(json.dumps(self.manifest, indent=2, default=str), encoding="utf-8")
        return target
