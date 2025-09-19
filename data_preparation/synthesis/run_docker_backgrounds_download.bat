@echo off
REM Docker script for downloading background images
REM Usage: run_docker_backgrounds_download.bat [num] [size] [thematic]
REM   num: number of backgrounds (default: 1000)
REM   size: image size (default: 1920)
REM   thematic: use thematic backgrounds (default: false)

set NUM_VAL=%1
if "%NUM_VAL%"=="" set NUM_VAL=1000

set SIZE_VAL=%2
if "%SIZE_VAL%"=="" set SIZE_VAL=1920

set THEMATIC_VAL=%3
if "%THEMATIC_VAL%"=="" set THEMATIC_VAL=false

echo Starting background download...
echo ==========================================
echo Parameters: num=%NUM_VAL%, size=%SIZE_VAL%, thematic=%THEMATIC_VAL%
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

if "%THEMATIC_VAL%"=="true" (
    docker run -v "%SYNTH_PATH%:/app/synthesis" -v "%DATA_PATH%:/app/data" --rm medphisiker/tbank-synth:latest python download_backgrounds.py --num %NUM_VAL% --size %SIZE_VAL% --thematic
) else (
    docker run -v "%SYNTH_PATH%:/app/synthesis" -v "%DATA_PATH%:/app/data" --rm medphisiker/tbank-synth:latest python download_backgrounds.py --num %NUM_VAL% --size %SIZE_VAL%
)

echo.
echo Background download completed!
echo Images saved to: data_preparation/synthesis/backgrounds/
echo ==========================================