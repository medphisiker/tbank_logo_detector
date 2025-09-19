@echo off
REM Docker script for cropping logos from official images
REM Usage: run_docker_crop_logos.bat

echo Starting logo cropping...
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

docker run -v "%SYNTH_PATH%:/app/synthesis" -v "%DATA_PATH%:/app/data" --rm medphisiker/tbank-synth:latest python crop_logos.py

echo.
echo Logo cropping completed!
echo Cropped logos saved to: data_preparation/synthesis/crops/
echo ==========================================