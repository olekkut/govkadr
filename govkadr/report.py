"""Per-photo report.json plus batch summary.json, summary.md and a local index.html gallery."""

from __future__ import annotations

import html
import os
from pathlib import Path

from .jobs import Job, write_json
from .spec import PHOTO


def _rel(p: str | None, base: Path) -> str | None:
    return Path(os.path.relpath(p, base)).as_posix() if p else None


def _row(job: Job) -> dict:
    r = job.report
    a, b, ident = r.get("strict-gov", {}), r.get("studio-ai", {}), r.get("identity", {})
    am = a.get("measurements", {})
    warns = [k for k, c in a.get("checks", {}).items() if c["status"] != "PASS"]
    return {
        "photo": job.src.name,
        "camera_focal_35mm": r.get("source", {}).get("focal_35mm"),
        "rotation_applied_deg_ccw": r.get("analysis", {}).get("applied_rotation_deg_ccw"),
        "strict_gov_verdict": a.get("verdict", "ERROR" if job.failed else None),
        "head_height_mm": am.get("head_height_mm"),
        "head_height_px": am.get("head_height_px"),
        "top_margin_mm": am.get("top_margin_mm"),
        "horizontal_offset_mm": am.get("horizontal_offset_mm"),
        "roll_residual_deg": am.get("roll_residual_deg"),
        "background_white_pct": round(100 * am.get("background", {}).get("pure_white_fraction", 0), 3) if am else None,
        "file_kb": round(a.get("file", {}).get("file_bytes", 0) / 1024) if a else None,
        "studio_ai_verdict": b.get("verdict"),
        "virtual_studio_verdict": r.get("virtual-studio", {}).get("verdict"),
        "cos_source_virtual": ident.get("cosine_source_vs_virtual_studio"),
        "studio_flux_verdict": r.get("studio-flux", {}).get("verdict"),
        "cos_source_flux": ident.get("cosine_source_vs_studio_flux"),
        "cos_source_strict": ident.get("cosine_source_vs_strict_gov"),
        "cos_source_studio": ident.get("cosine_source_vs_studio_ai"),
        "strict_gov_non_pass_checks": warns,
        "error": job.failed,
    }


def write_reports(jobs: list[Job], out_dir: Path, baseline: dict, run_info: dict) -> None:
    for job in jobs:
        write_json(job.out / "report.json", {"photo": job.src.name, "error": job.failed, **job.report})
    rows = [_row(j) for j in jobs]
    write_json(out_dir / "summary.json", {"run": run_info, "spec": PHOTO.__dict__, "identity_baseline": baseline, "photos": rows})

    def f(v, fmt="{}"):
        return "-" if v is None else fmt.format(v)

    lines = [
        "# govkadr batch summary",
        "",
        f"Last run: {run_info.get('started')} · {run_info.get('photos')} photos · mode {run_info.get('mode')} · "
        f"stages {', '.join(run_info.get('stages', []))} in {run_info.get('seconds')} s (earlier stages reused from cache)",
        "",
        f"Identity baseline (ArcFace cosine between different raw shots of the same person): median {baseline.get('median')}, "
        f"min {baseline.get('min')}, max {baseline.get('max')} over {baseline.get('pairs')} pairs.",
        "",
        "| Photo | f(35mm) | Rotation | strict-gov | Head mm | Top mm | Offset mm | Roll res. | BG #FFF % | cos A | studio-ai | cos B | virtual-studio | cos V | studio-flux | cos F | Non-PASS checks (A) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['photo']} | {f(r['camera_focal_35mm'])} | {f(r['rotation_applied_deg_ccw'], '{:+.2f}°')} | {f(r['strict_gov_verdict'])} | "
            f"{f(r['head_height_mm'], '{:.2f}')} | {f(r['top_margin_mm'], '{:.2f}')} | {f(r['horizontal_offset_mm'], '{:+.2f}')} | "
            f"{f(r['roll_residual_deg'], '{:+.3f}°')} | {f(r['background_white_pct'], '{:.2f}')} | {f(r['cos_source_strict'], '{:.3f}')} | "
            f"{f(r['studio_ai_verdict'])} | {f(r['cos_source_studio'], '{:.3f}')} | {f(r['virtual_studio_verdict'])} | {f(r['cos_source_virtual'], '{:.3f}')} | {f(r['studio_flux_verdict'])} | {f(r['cos_source_flux'], '{:.3f}')} | "
            f"{', '.join(r['strict_gov_non_pass_checks']) or '-'}{' ERROR: ' + r['error'] if r['error'] else ''} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_gallery(jobs, rows, out_dir, baseline)


def _write_gallery(jobs: list[Job], rows: list[dict], out_dir: Path, baseline: dict) -> None:
    cards = []
    for job, r in zip(jobs, rows):
        a = job.report.get("strict-gov", {}).get("files", {})
        b = job.report.get("studio-ai", {}).get("files", {})

        def img(p, label):
            rel = _rel(p, out_dir)
            return f'<figure><a href="{rel}"><img loading="lazy" src="{rel}" alt="{label}"></a><figcaption>{label}</figcaption></figure>' if rel else ""

        def badge(v):
            cls = {"PASS": "ok", "PASS_WITH_WARNINGS": "warn", "FAIL": "bad", "ERROR": "bad", "SKIPPED": "na"}.get(v or "", "na")
            return f'<span class="b {cls}">{html.escape((v or "n/a").replace("_", " "))}</span>'

        links = " · ".join(f'<a href="{_rel(p, out_dir)}">{n}</a>' for n, p in (("A diagnostic", a.get("diagnostic")), ("A sheet 10x15", a.get("sheet_10x15cm")), ("B diagnostic", b.get("diagnostic")), ("B comparison", b.get("comparison")), ("report.json", str(job.out / "report.json"))) if p)
        cards.append(
            f'<section><h2>{html.escape(r["photo"])}</h2><p>strict-gov {badge(r["strict_gov_verdict"])} studio-ai {badge(r["studio_ai_verdict"])} '
            f'head {r["head_height_mm"]} mm · offset {r["horizontal_offset_mm"]} mm · rotation {r["rotation_applied_deg_ccw"]}° · '
            f'cos(src,A) {r["cos_source_strict"]} · cos(src,B) {r["cos_source_studio"]}</p>'
            f'<div class="row">{img(a.get("photo"), "strict-gov")}{img(job.report.get("virtual-studio", {}).get("files", {}).get("photo"), "virtual-studio")}{img(b.get("photo"), "studio-ai")}{img(job.report.get("studio-flux", {}).get("files", {}).get("photo"), "studio-flux")}</div>'
            f"<p class=links>{links}</p></section>"
        )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>govkadr results</title><style>
:root{{--bg:#f6f6f8;--fg:#1d1d24;--muted:#666;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16161a;--fg:#ececf1;--muted:#a0a0aa;--card:#202027}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,Segoe UI,sans-serif}}
section{{background:var(--card);border-radius:12px;padding:12px 16px;margin:0 0 16px}}
h1{{font-size:22px}} h2{{font-size:16px;margin:4px 0}} p{{margin:6px 0;color:var(--muted)}}
.row{{display:flex;gap:12px;flex-wrap:wrap}} figure{{margin:0}} img{{height:300px;max-width:100%;border:1px solid #ccc}}
figcaption{{font-size:12px;color:var(--muted)}} .b{{padding:1px 8px;border-radius:9px;color:#fff;font-size:12px}}
.ok{{background:#22964a}} .warn{{background:#e08c00}} .bad{{background:#d22d2d}} .na{{background:#888}} a{{color:#2f6fd6}}
</style></head><body><h1>govkadr · 35x45 mm biometric photos</h1>
<p>Identity baseline between different raw shots: median cosine {baseline.get("median")} (min {baseline.get("min")}). See summary.md for the table.</p>
{"".join(cards)}</body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")
