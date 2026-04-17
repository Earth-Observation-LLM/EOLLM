#!/bin/bash
cd "$(dirname "$0")/.."
echo "Starting Urban VQA Sample Viewer..."
echo "Open: http://localhost:8765"
python3 -m http.server 8765 --directory . &
SERVER_PID=$!
sleep 1
xdg-open http://localhost:8765/viewer/ 2>/dev/null || open http://localhost:8765/viewer/ 2>/dev/null || echo "Open http://localhost:8765/viewer/ in your browser"
echo "Press Ctrl+C to stop"
wait $SERVER_PID
