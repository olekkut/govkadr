"""Print manager: a local web GUI to pick photos (and modes) for a kiosk print sheet.

``govkadr gui`` serves a single page on 127.0.0.1 (never on the network) using only the standard
library. It reads the per-photo reports under ``output/``, shows every mode's 35x45 photo with its
verdict, and builds 10x15 cm or 4x6" sheets from any mix of photos into ``output/prints/``.
"""

from __future__ import annotations

import json
import os
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .fill import _session_and_time
from .jobs import make_jobs

MODES = ("strict-gov", "virtual-studio", "studio-ai", "studio-flux")


def _catalog(input_dir: Path, out_dir: Path) -> list[dict]:
    root = out_dir.resolve()
    items = []
    for job in make_jobs(input_dir, out_dir):
        job.load()
        if "strict-gov" not in job.report:
            continue
        ident = job.report.get("identity", {})
        fill = job.cache / "fill.json"
        fill_info = json.loads(fill.read_text(encoding="utf-8")) if fill.exists() else {}
        modes = {}
        for mode in MODES:
            r = job.report.get(mode, {})
            photo = r.get("files", {}).get("photo")
            if not photo or not Path(photo).exists():
                continue
            rel = Path(os.path.relpath(Path(photo).resolve(), root)).as_posix()
            m = r.get("measurements", {})
            modes[mode] = {
                "url": "/files/" + rel,
                "diagnostic": "/files/" + Path(os.path.relpath(Path(r["files"]["diagnostic"]).resolve(), root)).as_posix(),
                "verdict": r.get("verdict"),
                "warnings": [k.replace("_", " ") for k, c in r.get("checks", {}).items() if c["status"] != "PASS"],
                "head_mm": m.get("head_height_mm"),
                "cos": ident.get(f"cosine_source_vs_{mode.replace('-', '_')}"),
            }
        src = job.report.get("source", {})
        session, secs = _session_and_time(job.stem)
        items.append(
            {
                "stem": job.stem,
                "session": f"{session[:4]}-{session[4:6]}-{session[6:]}" if session else "",
                "time": f"{secs // 3600:02d}:{secs // 60 % 60:02d}:{secs % 60:02d}" if session else "",
                "focal": src.get("focal_35mm"),
                "edge_fill_mm2": fill_info.get("area_mm2_on_print") if fill_info.get("applied") else None,
                "modes": modes,
            }
        )
    return items


def _export(out_dir: Path, tiles: list[dict], sheet_name: str) -> Path:
    import numpy as np
    from PIL import Image

    from . import imageio, sheet
    from .spec import SHEET_4x6, SHEET_10x15

    spec = SHEET_4x6 if sheet_name == "4x6" else SHEET_10x15
    root = out_dir.resolve()
    photos = []
    for t in tiles[: spec.cols * spec.rows]:
        p = (root / unquote(t["url"]).removeprefix("/files/")).resolve()
        if root not in p.parents:
            raise ValueError("photo outside the output folder")
        photos.append(np.asarray(Image.open(p).convert("RGB")))
    if not photos:
        raise ValueError("no photos selected")
    dest = root / "prints" / f"{time.strftime('%Y%m%d-%H%M%S')}_{spec.name}.jpg"
    imageio.save_jpeg(dest, sheet.build(photos, spec), quality=95)
    manifest = dest.with_suffix(".json")
    manifest.write_text(json.dumps({"sheet": spec.label, "tiles": tiles}, indent=2), encoding="utf-8")
    return dest


def make_handler(input_dir: Path, out_dir: Path):
    root = out_dir.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet console
            pass

        def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = 200) -> None:
            self._send(json.dumps(data).encode("utf-8"), "application/json", status)

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/photos":
                self._json(_catalog(input_dir, out_dir))
            elif path.startswith("/files/"):
                f = (root / unquote(path[len("/files/") :])).resolve()
                if root not in f.parents or not f.is_file() or f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".json"):
                    self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                    return
                ctype = "application/json" if f.suffix == ".json" else "image/png" if f.suffix == ".png" else "image/jpeg"
                self._send(f.read_bytes(), ctype)
            else:
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/api/export":
                self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                dest = _export(out_dir, body.get("tiles", []), body.get("sheet", "10x15"))
                self._json({"url": "/files/" + dest.relative_to(root).as_posix(), "path": str(dest)})
            except Exception as e:  # noqa: BLE001
                self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

    return Handler


def serve(input_dir: Path, out_dir: Path, port: int = 8765, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(input_dir, out_dir))
    url = f"http://127.0.0.1:{port}/"
    print(f"govkadr print manager on {url} (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>govkadr print manager</title>
<style>
:root{--bg:#f4f4f6;--card:#fff;--fg:#1c1c22;--muted:#6b6b76;--line:#dcdce2;--accent:#2f6fd6;--ok:#22964a;--warn:#d98a00;--bad:#d22d2d}
@media (prefers-color-scheme:dark){:root{--bg:#141418;--card:#1f1f26;--fg:#ececf1;--muted:#9a9aa6;--line:#34343e;--accent:#6d9cff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 16px;background:var(--card);border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0 8px 0 0}select,button{font:inherit;padding:6px 10px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button:disabled{opacity:.5;cursor:default}
main{display:grid;grid-template-columns:1fr 380px;gap:16px;padding:16px}@media (max-width:900px){main{grid-template-columns:1fr}}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px;align-content:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px;display:flex;flex-direction:column;gap:6px}
.card img{width:100%;aspect-ratio:826/1062;object-fit:cover;border-radius:6px;border:1px solid var(--line);background:#fff;cursor:zoom-in}
.tabs{display:flex;gap:4px;flex-wrap:wrap}.tabs button{padding:2px 7px;font-size:12px}.tabs button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.meta{font-size:12px;color:var(--muted)}.b{display:inline-block;padding:0 7px;border-radius:9px;color:#fff;font-size:11px}
.PASS{background:var(--ok)}.PASS_WITH_WARNINGS{background:var(--warn)}.FAIL,.ERROR{background:var(--bad)}.SKIPPED{background:#888}
.row{display:flex;gap:6px}.row button{flex:1}
aside{position:sticky;top:70px;align-self:start;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
#sheet{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;background:#fff;padding:8px;border-radius:6px;border:1px solid var(--line)}
.slot{aspect-ratio:826/1062;border:1px dashed #bbb;border-radius:3px;position:relative;overflow:hidden;background:#fafafa}
.slot img{width:100%;height:100%;object-fit:cover;cursor:pointer}.slot span{position:absolute;left:2px;bottom:2px;font-size:9px;background:#000a;color:#fff;padding:0 3px;border-radius:3px}
.note{font-size:12px;color:var(--muted);margin:8px 0}#status a{color:var(--accent)}
dialog{border:none;border-radius:12px;padding:0;max-width:95vw}dialog img{max-width:95vw;max-height:90vh;display:block}
</style></head><body>
<header><h1>govkadr · print manager</h1>
<label>Session <select id="session"><option value="">all</option></select></label>
<label><input type="checkbox" id="okonly"> hide FAIL/ERROR</label>
<label>Sheet <select id="sheetsize"><option value="4x6">4×6 in (102×152 mm kiosk paper)</option><option value="10x15">150×100 mm exact</option></select></label>
<span id="status" class="note"></span></header>
<main><section id="grid"></section>
<aside><b>Sheet (8 photos)</b><div class="note">Click a tile to remove it. Unused tiles print blank with crop marks.</div>
<div id="sheet"></div>
<div class="row" style="margin-top:10px"><button id="clear">Clear</button><button id="repeat">Repeat to fill</button></div>
<button id="export" class="primary" style="width:100%;margin-top:8px">Create print sheet</button>
<div class="note">Only <b>strict-gov</b> photos are meant for an official ID / passport application. Print at 100 % with "no cropping".</div></aside></main>
<dialog id="zoom"><img alt=""></dialog>
<script>
const LABEL={"strict-gov":"strict-gov","virtual-studio":"virtual","studio-ai":"IC-Light","studio-flux":"FLUX"};
let photos=[], tiles=[];
const $=s=>document.querySelector(s);
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
async function load(){photos=await (await fetch("/api/photos")).json();
  const ss=[...new Set(photos.map(p=>p.session))];ss.forEach(s=>$("#session").insertAdjacentHTML("beforeend",`<option value="${s}">${s||"no date"}</option>`));render();
  if(new URLSearchParams(location.search).has("demo")) demo(); renderSheet();}
// ?demo=1 (used for the README screenshot): cards in different modes and a partly filled sheet.
function demo(){const ms=["studio-flux","virtual-studio","strict-gov"];photos.forEach((p,i)=>{if(p.modes[ms[i%3]]) p.cur=ms[i%3]});
  tiles=[];photos.forEach(p=>{for(let i=0;i<2;i++) tiles.push({url:p.modes[p.cur].url,stem:p.stem,mode:p.cur})});render();}
function render(){const sess=$("#session").value, ok=$("#okonly").checked; $("#grid").innerHTML="";
  for(const p of photos){ if(sess&&p.session!==sess) continue;
    const modes=Object.keys(p.modes); if(!modes.length) continue; p.cur=p.cur&&p.modes[p.cur]?p.cur:modes[0];
    const m=p.modes[p.cur]; if(ok&&["FAIL","ERROR"].includes(m.verdict)) continue;
    const el=document.createElement("div"); el.className="card";
    el.innerHTML=`<img loading="lazy" src="${m.url}" alt="${esc(p.stem)}">
      <div class="tabs">${modes.map(k=>`<button data-m="${k}" class="${k===p.cur?"on":""}">${LABEL[k]}</button>`).join("")}</div>
      <div><span class="b ${m.verdict}">${esc((m.verdict||"").replaceAll("_"," "))}</span> <b>${esc(p.stem)}</b></div>
      <div class="meta">${[p.session&&`${p.session} ${p.time}`, p.focal&&`${p.focal} mm lens`, `head ${m.head_mm??"-"} mm`, `id ${m.cos??"-"}`].filter(Boolean).join(" · ")}</div>
      ${m.warnings.length?`<div class="meta">⚠ ${esc(m.warnings.join(", "))}</div>`:""}
      ${p.edge_fill_mm2&&p.cur!=="strict-gov"?`<div class="meta">edge fill: ${p.edge_fill_mm2} mm² generated</div>`:""}
      <div class="row"><button data-a="1">+1 to sheet</button><button data-a="8">Fill sheet ×8</button></div>
      <a class="meta" href="${m.diagnostic}" target="_blank">diagnostic</a>`;
    el.querySelectorAll(".tabs button").forEach(b=>b.onclick=()=>{p.cur=b.dataset.m;render()});
    el.querySelector("img").onclick=()=>{$("#zoom img").src=m.url;$("#zoom").showModal()};
    el.querySelectorAll("[data-a]").forEach(b=>b.onclick=()=>{const n=+b.dataset.a; if(n===8) tiles=[];
      for(let i=0;i<n&&tiles.length<8;i++) tiles.push({url:m.url,stem:p.stem,mode:p.cur}); renderSheet()});
    $("#grid").append(el);} }
function renderSheet(){$("#sheet").innerHTML="";for(let i=0;i<8;i++){const t=tiles[i];const s=document.createElement("div");s.className="slot";
  if(t){s.innerHTML=`<img src="${t.url}" alt=""><span>${LABEL[t.mode]}</span>`;s.onclick=()=>{tiles.splice(i,1);renderSheet()}} $("#sheet").append(s)}
  $("#export").disabled=!tiles.length;}
$("#session").onchange=render;$("#okonly").onchange=render;$("#zoom").onclick=()=>$("#zoom").close();
$("#clear").onclick=()=>{tiles=[];renderSheet()};
$("#repeat").onclick=()=>{if(!tiles.length)return;const base=[...tiles];while(tiles.length<8)tiles.push(base[tiles.length%base.length]);renderSheet()};
$("#export").onclick=async()=>{$("#status").textContent="Building sheet…";
  const r=await fetch("/api/export",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({sheet:$("#sheetsize").value,tiles})});
  const j=await r.json(); $("#status").innerHTML=j.error?`Error: ${esc(j.error)}`:`Saved <a href="${j.url}" target="_blank">${esc(j.path)}</a>`;};
load();
</script></body></html>"""
