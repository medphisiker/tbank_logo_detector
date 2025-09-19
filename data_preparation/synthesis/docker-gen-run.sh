#!/bin/bash

# Docker pull and run for synthesis test (small N=10) with pre-built image


docker pull medphisiker/tbank-synth:latest
docker run -v "$(pwd):/app/synthesis" -v "$(pwd)/../../data:/app/data" --rm medphisiker/tbank-synth python gen_synth.py --N 10 --min_scale_down 0.5 --iou_threshold 0.4 --max_neg 15