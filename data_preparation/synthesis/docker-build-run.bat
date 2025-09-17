@echo off
REM Docker pull and run for synthesis test (small N=10) with pre-built image on Windows

REM Альтернатива: Соберите свой образ
REM docker build -t tbank-synth -f Dockerfile .

docker pull medphisiker/tbank-synth:latest
docker run -v "%cd%:/app/synthesis" -v "%cd%/../../data:/app/data" --rm medphisiker/tbank-synth python gen_synth.py --N 10