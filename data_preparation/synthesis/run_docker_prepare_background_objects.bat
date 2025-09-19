@echo off
REM Docker script for preparing background objects (distractors)
REM Usage: run_docker_prepare_background_objects.bat [num]
REM   num: number of distractor images (default: 200)

set NUM_VAL=%1
if "%NUM_VAL%"=="" set NUM_VAL=200

echo Starting background objects preparation...
echo ==========================================
echo Parameters: num=%NUM_VAL%
echo ==========================================

REM Переходим в корень проекта
cd /d "%~dp0..\.."

REM Нормализуем пути для Docker (заменяем \ на /)
set "ROOT_PATH=%cd:/=/%"
set "SYNTH_PATH=%ROOT_PATH%/data_preparation/synthesis"
set "DATA_PATH=%ROOT_PATH%/data"

REM Отладка путей
echo Root: %ROOT_PATH%
echo Synthesis: %SYNTH_PATH%
echo Data: %DATA_PATH%

docker run -v "%SYNTH_PATH%:/app/synthesis" -v "%DATA_PATH%:/app/data" --rm medphisiker/tbank-synth:latest python prepare_background_objects.py --num %NUM_VAL%

echo.
echo Background objects preparation completed!
echo Distractors saved to: data_preparation/synthesis/background_objects/
echo ==========================================