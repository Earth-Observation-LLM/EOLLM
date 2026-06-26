#!/usr/bin/env python3
"""serve.py — open the built viewer.html in a local browser.

viewer.html is fully self-contained (images base64-inlined), so this just serves
the solved/<model> dir over http and opens the page. Build it first with
build_viewer.py.

Usage:
    python serve.py --solved ../solved/qwen36_27b
    python serve.py --solved ../solved/qwen36_27b --port 8011
    # share over your Tailscale tailnet (bind all interfaces, no auto-open):
    python serve.py --solved ../solved/qwen36_27b --host 0.0.0.0 --no-open
"""
from __future__ import annotations

import argparse
import http.server
import socketserver
import webbrowser
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent


class _Server(socketserver.TCPServer):
    allow_reuse_address = True            # reclaim the port quickly on restart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solved", required=True)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; use 0.0.0.0 to expose over the tailnet/LAN")
    ap.add_argument("--no-open", action="store_true",
                    help="don't auto-open a local browser (for headless/shared serving)")
    args = ap.parse_args()

    solved_dir = Path(args.solved)
    if not solved_dir.is_absolute():
        solved_dir = (HERE / args.solved).resolve()
    html = solved_dir / "viewer.html"
    if not html.exists():
        raise SystemExit(f"{html} not found; run build_viewer.py --solved {args.solved} first")

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(solved_dir))
    with _Server((args.host, args.port), handler) as httpd:
        shown = "127.0.0.1" if args.host in ("127.0.0.1", "localhost") else args.host
        url = f"http://{shown}:{args.port}/viewer.html"
        print(f"serving {html} at {url}  (Ctrl-C to stop)")
        if not args.no_open:
            webbrowser.open(f"http://127.0.0.1:{args.port}/viewer.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
