"""Per-photo job bookkeeping, deliberately free of heavy imports.

The ``run`` orchestrator stays alive while every stage subprocess runs, so it must not pull in
numpy/OpenCV/MediaPipe/torch: on a machine short of commit, every resident MB counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
# Files whose presence means a stage finished for a photo (stages resume / retry per photo).
STAGE_ARTIFACTS = {
    "analyze": ("face.json", "report.analysis.json"),
    "matting": ("matte.npz", "framing.json", "report.matting.json"),
    "strict": ("subject.npz", "thumb.npz", "report.strict-gov.json"),
    "studio": ("render.json",),  # render.png is transient: deleted once studio-finish used it
    "studio-finish": ("report.studio-ai.json",),
    "identity": ("report.identity.json",),
}
# Extra files a stage owns besides its completion markers.
STAGE_EXTRAS = {
    "analyze": ("report.source.json",),
    "matting": ("report.framing.json",),
    "studio": ("render_error.txt", "render.png"),
}


def _default(o):
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")


@dataclass
class Job:
    src: Path
    out: Path
    cache: Path
    report: dict = field(default_factory=dict)
    failed: str | None = None

    @property
    def stem(self) -> str:
        return self.src.stem

    def path(self, mode: str, kind: str, ext: str = "jpg") -> Path:
        return self.out / mode / f"{self.stem}_{mode}_{kind}.{ext}"

    def save(self, key: str, data: dict) -> None:
        """Record a report section in memory and on disk (stages run in separate processes)."""
        self.report[key] = data
        write_json(self.cache / f"report.{key}.json", data)

    def fail(self, msg: str) -> None:
        self.failed = msg
        self.cache.mkdir(parents=True, exist_ok=True)
        (self.cache / "failed.txt").write_text(msg, encoding="utf-8")

    def load(self) -> "Job":
        self.report = {}
        for f in sorted(self.cache.glob("report.*.json")):
            self.report[f.stem.split(".", 1)[1]] = json.loads(f.read_text(encoding="utf-8"))
        marker = self.cache / "failed.txt"
        self.failed = marker.read_text(encoding="utf-8") if marker.exists() else None
        return self

    def done(self, stage: str) -> bool:
        if stage == "studio" and self.skipped_studio():
            return True
        return all((self.cache / f).exists() for f in STAGE_ARTIFACTS[stage])

    def skipped_studio(self) -> str | None:
        """Reason the generative render was deliberately skipped (low memory), if it was."""
        f = self.cache / "render_error.txt"
        if f.exists():
            text = f.read_text(encoding="utf-8")
            if text.startswith("skipped:"):
                return text
        return None

    def clear(self, stage: str) -> None:
        """Forget a stage's results so it is recomputed."""
        for f in STAGE_ARTIFACTS[stage] + STAGE_EXTRAS.get(stage, ()):
            (self.cache / f).unlink(missing_ok=True)
            if f.startswith("report."):
                self.report.pop(f[len("report.") : -len(".json")], None)
        (self.cache / "failed.txt").unlink(missing_ok=True)
        self.failed = None


def make_jobs(input_dir: Path, output_dir: Path, only: list[str] | None = None, limit: int | None = None) -> list[Job]:
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if only:
        files = [p for p in files if any(o in p.stem for o in only)]
    if limit:
        files = files[:limit]
    return [Job(p, output_dir / p.stem, output_dir / ".cache" / p.stem) for p in files]
