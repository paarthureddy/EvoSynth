#!/bin/bash
# EvoSynth — Launch the full web server (UI + backend API)

cd "$(dirname "$0")"

echo "=========================================================="
echo "    EvoSynth — Hybrid AutoML Dashboard"
echo "=========================================================="
echo "  Open: http://localhost:8000"
echo "  Stop: Ctrl+C"
echo "=========================================================="

python3 server.py
