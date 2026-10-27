"""Command line entry point: ``govkadr run`` / ``govkadr fetch-models``.

``run`` executes every pipeline stage in its own Python process. On Windows, VRAM is backed by
system commit and a process rarely hands freed heap back to the OS, so isolating the BiRefNet and
IC-Light stages guarantees each one starts with the full memory budget. Stages resume per photo,
so a stage process that dies (e.g. a sticky CUDA error when the OS runs out of commit) is simply
restarted for the photos still missing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Must be set before torch initialises CUDA: returns freed blocks to the driver promptly, which on
# Windows also returns the system commit that backs them.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# numpy and scipy each bundle OpenBLAS, which commits a buffer per thread at import time
# (~0.38 GB each with 16 threads); 4 threads is plenty for this pipeline's small solves.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

STAGES = ("analyze", "matting", "strict", "studio", "studio-finish", "identity")


def _jobs(args):
    from .jobs import make_jobs

    return make_jobs(args.input, args.output, args.only, args.limit)


def _stage(args: argparse.Namespace) -> int:
    """Run one stage in this process (invoked by ``run`` as a subprocess)."""
    from . import pipeline

    jobs = [j.load() for j in _jobs(args)]
    name = args.name
    analyzer = None
    if name in ("analyze", "strict", "studio-finish"):
        from .landmarks import FaceAnalyzer
        from .weights import FACE_LANDMARKER, fetch_url

        analyzer = FaceAnalyzer(fetch_url(FACE_LANDMARKER))
    if name == "analyze":
        pipeline.stage_analyze(jobs, analyzer)
    elif name == "matting":
        pipeline.stage_matting(jobs)
    elif name == "strict":
        pipeline.stage_strict(jobs, analyzer)
    elif name == "studio":
        pipeline.stage_studio(jobs)
    elif name == "studio-finish":
        pipeline.stage_studio_finish(jobs, analyzer)
    elif name == "identity":
        pipeline.stage_identity(jobs)
    return 0


def _run(args: argparse.Namespace) -> int:
    from . import report
    from .resources import free_commit_gb

    jobs = _jobs(args)
    if not jobs:
        print(f"no images in {args.input}")
        return 1
    if args.allow_low_memory:
        os.environ["GOVKADR_ALLOW_LOW_MEMORY"] = "1"  # inherited by the stage subprocesses
    stages = [s for s in STAGES if args.mode != "strict-gov" or not s.startswith("studio")]
    stages = stages[stages.index(args.from_stage) :] if args.from_stage else stages
    if args.stages:
        stages = [s for s in STAGES if s in args.stages]
    for job in jobs:
        job.load()
        for stage in stages:
            job.clear(stage)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()
    print(f"govkadr: {len(jobs)} photo(s), mode {args.mode}, stages {', '.join(stages)}, free commit {free_commit_gb():.1f} GB", flush=True)
    timings = {}
    for stage in stages:
        ts = time.perf_counter()
        cmd = [sys.executable, "-m", "govkadr", "_stage", stage, "--input", str(args.input), "--output", str(args.output)]
        if args.only:
            cmd += ["--only", *args.only]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        for attempt in range(1, 4):
            rc = subprocess.call(cmd)
            pending = [j.stem for j in (job.load() for job in _jobs(args)) if not j.failed and not j.done(stage)]
            if rc == 0 and not pending:
                break
            print(f"stage {stage}: exit {rc}, {len(pending)} photo(s) pending" + (" - restarting" if attempt < 3 else " - giving up"), flush=True)
            if attempt < 3:
                time.sleep(15)
        timings[stage] = round(time.perf_counter() - ts, 1)
    jobs = [j.load() for j in _jobs(args)]
    baseline_file = args.output / ".cache" / "identity_baseline.json"
    baseline = json.loads(baseline_file.read_text(encoding="utf-8")) if baseline_file.exists() else {}
    run_info = {"started": started, "photos": len(jobs), "mode": args.mode, "stages": stages, "seconds": round(time.perf_counter() - t0, 1), "stage_seconds": timings}
    report.write_reports(jobs, args.output, baseline, run_info)
    failed = [j for j in jobs if j.failed]
    print(f"done in {run_info['seconds']} s: {len(jobs) - len(failed)} ok, {len(failed)} failed -> {args.output / 'summary.md'}", flush=True)
    return 0 if not failed else 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="govkadr", description="35x45 mm biometric photos (gov.pl / ICAO 9303) from phone portraits")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--input", type=Path, default=Path("photos-from-camera"))
        sp.add_argument("--output", type=Path, default=Path("output"))
        sp.add_argument("--only", nargs="*", help="substrings of file names to process")
        sp.add_argument("--limit", type=int, help="process at most N photos")

    r = sub.add_parser("run", help="process a folder of portraits")
    common(r)
    r.add_argument(
        "--mode",
        choices=("both", "strict-gov", "studio-ai"),
        default="both",
        help="strict-gov: deterministic, submission-safe; studio-ai: generative relighting on top of it (R&D only)",
    )
    r.add_argument("--from-stage", choices=STAGES, help="re-run from this stage, reusing earlier cached stages")
    r.add_argument("--stages", nargs="+", choices=STAGES, help="re-run exactly these stages, reusing every other cached stage")
    r.add_argument(
        "--allow-low-memory",
        action="store_true",
        help="run IC-Light even when free system commit is below ~5.2 GB (may starve other applications)",
    )
    s = sub.add_parser("_stage", help=argparse.SUPPRESS)
    s.add_argument("name", choices=STAGES)
    common(s)
    f = sub.add_parser("fetch-models", help="download and verify all model weights")
    f.add_argument("--no-generative", action="store_true", help="skip the ~2 GB IC-Light / SD1.5 weights")
    args = p.parse_args(argv)
    if args.cmd == "fetch-models":
        from .weights import fetch_all

        fetch_all(include_generative=not args.no_generative)
        return 0
    if args.cmd == "_stage":
        return _stage(args)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
