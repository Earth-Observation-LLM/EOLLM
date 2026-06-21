#!/usr/bin/env python3
"""serve.py — local server viewer (on-demand images + attention).

The static viewer inlines every image as base64, which blows up to hundreds of
MB and chokes the browser at ~1400 records. This server instead keeps the data
in memory and serves it lazily: the page loads ALL records' metadata instantly
(small JSON), and a record's images + attention are fetched only when you click
it. Images are rebuilt on demand (byte-identical to what the model saw) and
LRU-cached, so the page stays snappy across the full benchmark.

Run:
    python viewer/serve.py --results results/qwen35_4b_awq_attn100 [--mode full] [--port 8077]
then open the printed URL.

Endpoints:
    GET /                     -> app shell (HTML/JS, no inlined data)
    GET /api/meta             -> model/mode/topic summary + aggregate stats
    GET /api/records          -> compact list for the left pane (all records)
    GET /api/record/<idx>     -> full detail (question, options, attention block)
    GET /api/image/<idx>/<k>  -> JPEG of image k for record idx (rebuilt + cached)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import OrderedDict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

SUITE = Path(__file__).resolve().parent.parent
REPO = SUITE.parent
sys.path.insert(0, str(SUITE))

import images as suite_images
import modes as suite_modes
import stats as suite_stats   # aggregate helper stats (see stats.py)

THUMB_PX = 360


class Store:
    """Loads one model's results dir and rebuilds images on demand."""

    def __init__(self, results_dir: Path, mode: str | None):
        self.dir = results_dir
        self.meta = json.loads((results_dir / "meta.json").read_text()) if (results_dir / "meta.json").exists() else {}
        modes_present = sorted(
            m for m in (p.name.split("_predictions.jsonl")[0]
                        for p in results_dir.glob("*_predictions.jsonl")) if m != "all")
        self.mode = mode or (modes_present[0] if modes_present else None)
        self.modes_present = modes_present
        if not self.mode:
            raise SystemExit(f"no *_predictions.jsonl in {results_dir}")
        self.rows = [json.loads(l) for l in open(results_dir / f"{self.mode}_predictions.jsonl") if l.strip()]

        # benchmark index for image rebuilds
        bench = self.meta.get("dataset_path")
        default_bench = REPO / "dataset_content" / "EODATA_compressed_final" / "benchmark"
        self.image_root = default_bench
        data_path = default_bench / "benchmark_with_answers.jsonl"
        self.bench = {}
        for line in open(data_path):
            if line.strip():
                r = json.loads(line)
                self.bench[r["question_id"]] = r
        self.max_edge = self.meta.get("image_max_edge", 512)
        self._img_cache: "OrderedDict[tuple,bytes]" = OrderedDict()
        self._img_cache_cap = 512

    # ---- list + detail ---------------------------------------------------
    def records_list(self):
        out = []
        for i, r in enumerate(self.rows):
            out.append({
                "i": i, "qid": r["question_id"], "topic": r["topic"],
                "gold": r["gold"], "pred": r["prediction"], "correct": r["is_correct"],
                "difficulty": r.get("difficulty"), "city": r.get("city"),
                "city_type": r.get("city_type"), "image_mode": r.get("image_mode"),
                "n_images": r["attention"]["n_images"] if r.get("attention") else len(r.get("image_roles", [])),
            })
        return out

    def record_detail(self, idx):
        r = self.rows[idx]
        return {
            "i": idx, "qid": r["question_id"], "topic": r["topic"],
            "city": r.get("city"), "difficulty": r.get("difficulty"),
            "question": r["question"], "options": r["options"],
            "gold": r["gold"], "prediction": r["prediction"], "correct": r["is_correct"],
            "prob_dict": r.get("prob_dict"), "raw": r.get("raw_response", ""),
            "image_roles": r.get("image_roles", []),
            "attention": r.get("attention"),
            "extras": suite_stats.per_record_extras(r.get("attention"), r),
        }

    # ---- on-demand image rebuild ----------------------------------------
    def image_jpeg(self, idx, k) -> bytes | None:
        key = (idx, k)
        if key in self._img_cache:
            self._img_cache.move_to_end(key)
            return self._img_cache[key]
        r = self.rows[idx]
        rec = self.bench.get(r["question_id"])
        if rec is None:
            return None
        full = suite_images.build_images(rec, str(self.image_root), self.max_edge)
        sel = suite_modes.filter_images_for_mode(full, r["ablation_mode"])
        if k < 0 or k >= len(sel):
            return None
        im = sel[k]["image"].convert("RGB")
        im.thumbnail((THUMB_PX, THUMB_PX))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
        data = buf.getvalue()
        self._img_cache[key] = data
        if len(self._img_cache) > self._img_cache_cap:
            self._img_cache.popitem(last=False)
        return data

    def meta_payload(self):
        return {
            "model": self.meta.get("key", self.dir.name),
            "display": self.meta.get("display"),
            "backend": self.meta.get("backend"),
            "mode": self.mode, "modes_present": self.modes_present,
            "n_records": len(self.rows),
            "stats": suite_stats.aggregate(self.rows),
        }


STORE: Store | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/":
                return self._send(200, APP_HTML, "text/html; charset=utf-8")
            if path == "/api/meta":
                return self._send(200, STORE.meta_payload())
            if path == "/api/records":
                return self._send(200, STORE.records_list())
            if path.startswith("/api/record/"):
                idx = int(path.rsplit("/", 1)[1])
                return self._send(200, STORE.record_detail(idx))
            if path.startswith("/api/image/"):
                _, _, _, sidx, sk = path.split("/")
                data = STORE.image_jpeg(int(sidx), int(sk))
                if data is None:
                    return self._send(404, {"error": "no image"})
                return self._send(200, data, "image/jpeg")
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    global STORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--mode", default=None)
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    STORE = Store(Path(args.results), args.mode)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"serving {STORE.dir.name} ({STORE.meta_payload()['n_records']} records, mode={STORE.mode})")
    print(f"  -> {url}")
    print("  Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


# APP_HTML is defined in app_shell.py to keep this file readable.
from app_shell import APP_HTML  # noqa: E402

if __name__ == "__main__":
    main()
