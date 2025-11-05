#!/bin/bash
# SpectralPredict GUI Launcher

echo "================================================"
echo "  SpectralPredict.jl GUI Launcher"
echo "================================================"
echo ""
echo "Starting GUI server..."
echo ""

cd "$(dirname "$0")"
julia --project=. gui.jl
