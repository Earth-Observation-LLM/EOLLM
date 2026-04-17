#!/usr/bin/env bash
# Serve geo.html viewer on localhost
# Usage: ./serve_geo.sh [port]
PORT="${1:-8080}"
echo "Serving geo viewer at http://localhost:${PORT}/geo.html"
python3 -m http.server "$PORT" --directory "$(dirname "$0")"
