@echo off
REM Docker build and run for synthesis test (small N=10) on Windows

docker build -t tbank-synth -f Dockerfile .
docker run -v "%cd%:/app/synthesis" -v "%cd%/../../data:/app/data" --rm tbank-synth python gen_synth.py --N 10