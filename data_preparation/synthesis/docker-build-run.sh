#!/bin/bash

# Docker build and run for synthesis test (small N=10)

docker build -t tbank-synth .. 
docker run -v $(pwd)/data_preparation/synthesis:/app/synthesis -v $(pwd)/data:/app/data --rm tbank-synth python gen_synth.py --N 10