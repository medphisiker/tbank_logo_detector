#!/bin/bash

# Synthesis Pipeline Runner - runs all synthesis scripts in Docker
# Usage: ./run_synthesis_pipeline.sh [N] [min_scale_down] [iou_threshold] [max_neg]
#   N: number of images (default: 1000)
#   min_scale_down: min background scale (default: 0.5)
#   iou_threshold: IoU threshold for logos (default: 0.4)
#   max_neg: max distractors per image (default: 15)

N_VAL=${1:-1000}
SCALE_VAL=${2:-0.5}
IOU_VAL=${3:-0.4}
NEG_VAL=${4:-15}

echo "Starting T-Bank Logo Synthesis Pipeline..."
echo "=========================================="
echo "Parameters: N=$N_VAL, scale_down=$SCALE_VAL, iou=$IOU_VAL, max_neg=$NEG_VAL"
echo "=========================================="

echo "Step 1: Cropping logos from official images..."
docker run -v "$(pwd)/data_preparation/synthesis:/app/synthesis" -v "$(pwd)/data:/app/data" --rm tbank-synth python crop_logos.py

echo ""
echo "Step 2: Downloading high-res background images..."
docker run -v "$(pwd)/data_preparation/synthesis:/app/synthesis" -v "$(pwd)/data:/app/data" --rm tbank-synth python download_backgrounds.py --num 1000 --size 1920

echo ""
echo "Step 3: Preparing background objects (distractors)..."
docker run -v "$(pwd)/data_preparation/synthesis:/app/synthesis" -v "$(pwd)/data:/app/data" --rm tbank-synth python prepare_background_objects.py --num 200

echo ""
echo "Step 4: Generating $N_VAL synthetic images with advanced features..."
docker run -v "$(pwd)/data_preparation/synthesis:/app/synthesis" -v "$(pwd)/data:/app/data" --rm tbank-synth python gen_synth.py --N $N_VAL --min_scale_down $SCALE_VAL --iou_threshold $IOU_VAL --max_neg $NEG_VAL

echo ""
echo "Synthesis pipeline completed!"
echo "Generated data saved to: data/data_synt/"
echo "Features: multi-logo, distractors, random resize, advanced augmentations"
echo "=========================================="