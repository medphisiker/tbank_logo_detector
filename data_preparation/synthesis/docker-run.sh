#!/bin/bash

# Docker pull and run for synthesis test (small N=10) with pre-built image


docker pull medphisiker/tbank-synth:latest
docker run -v "$(pwd):/app/synthesis" -v "$(pwd)/../../data:/app/data" --rm medphisiker/tbank-synth python gen_synth.py --N 10