import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from govkadr import gui
from govkadr.jobs import make_jobs


def test_make_jobs_filters_images_and_tracks_stage_state(tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    for name in ("a.jpg", "b.JPEG", "notes.txt"):
        (inp / name).write_bytes(b"x")
    jobs = make_jobs(inp, out)
    assert [j.stem for j in jobs] == ["a", "b"]
    job = jobs[0]
    assert not job.done("identity")
    job.save("identity", {"cos": 0.9})
    assert job.done("identity")
    assert job.load().report["identity"]["cos"] == 0.9
    job.clear("identity")
    assert not job.done("identity")


@pytest.fixture
def server(tmp_path):
    (tmp_path / "in").mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "secret.txt").write_text("nope")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), gui.make_handler(tmp_path / "in", out))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_gui_serves_page_and_empty_catalog(server):
    assert b"print manager" in urllib.request.urlopen(server + "/").read()
    assert json.loads(urllib.request.urlopen(server + "/api/photos").read()) == []


def test_gui_rejects_path_traversal(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(server + "/files/..%2Fsecret.txt")
    assert e.value.code == 404


def test_gui_export_rejects_empty_selection(server):
    req = urllib.request.Request(server + "/api/export", data=json.dumps({"tiles": []}).encode(), headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400
