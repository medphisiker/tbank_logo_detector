#!/bin/bash

# Synthesis Pipeline Runner - runs all synthesis scripts in Docker
# Usage: ./run_synthesis_pipeline.sh [N] — N optional, default 20

N_VAL=${1:-20}

echo "Starting T-Bank Logo Synthesis Pipeline..."
echo "=========================================="

echo "Step 1: Cropping logos from official images..."
docker run -v "$(pwd)/data_preparation/synthesis:/app/synthesis" -v "$(pwd)/data:/app/data" --rm tbank-synth python crop_logos.py

echo ""
echo "Step 2: Downloading background images..."
docker run -v "$(pwd)/data_preparation/synthesis:/app/synthesis" -v "$(pwd)/data:/app/data" --rm tbank-synth python download_backgrounds.py

echo ""
echo "Step 3: Generating $N_VAL synthetic images..."
docker run -v "$(pwd)/data_preparation/synthesis:/app/synthesis" -v "$(pwd)/data:/app/data" --rm tbank-synth python gen_synth.py --N $N_VAL

echo ""
echo "Synthesis pipeline completed!"
echo "Generated data saved to: data/data_synt/"
echo "=========================================="