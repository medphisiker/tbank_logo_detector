@echo off
REM Docker pull and run for synthesis test (small N=10) with pre-built image on Windows

REM Переходим в корень проекта
cd /d "%~dp0..\.."


docker pull medphisiker/tbank-synth:latest

REM Нормализуем пути для Docker (заменяем \ на /)
set "ROOT_PATH=%cd:/=/%"
set "SYNTH_PATH=%ROOT_PATH%/data_preparation/synthesis"
set "DATA_PATH=%ROOT_PATH%/data"

REM Отладка путей
echo Root: %ROOT_PATH%
echo Synthesis: %SYNTH_PATH%
echo Data: %DATA_PATH%

docker run -v "%SYNTH_PATH%:/app/synthesis" -v "%DATA_PATH%:/app/data" --rm medphisiker/tbank-synth python gen_synth.py --N 10 --min_scale_down 0.5 --iou_threshold 0.4 --max_neg 15