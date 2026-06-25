#!/usr/bin/env python3
"""serve.py — open the built viewer.html in a local browser.

viewer.html is fully self-contained (images base64-inlined), so this just serves
the solved/<model> dir over http and opens the page. Build it first with
build_viewer.py.

Usage:
    python serve.py --solved ../solved/qwen36_27b
    python serve.py --solved ../solved/qwen36_27b --port 8011
"""
from __future__ import annotations

import argparse
import http.server
import socketserver
import webbrowser
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solved", required=True)
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()

    solved_dir = Path(args.solved)
    if not solved_dir.is_absolute():
        solved_dir = (HERE / args.solved).resolve()
    html = solved_dir / "viewer.html"
    if not html.exists():
        raise SystemExit(f"{html} not found; run build_viewer.py --solved {args.solved} first")

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(solved_dir))
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://127.0.0.1:{args.port}/viewer.html"
        print(f"serving {html} at {url}  (Ctrl-C to stop)")
        webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
