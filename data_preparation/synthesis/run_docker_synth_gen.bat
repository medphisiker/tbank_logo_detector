@echo off
REM Docker script for generating synthetic data
REM Usage: run_docker_synth_gen.bat [N] [min_scale_down] [iou_threshold] [max_neg]
REM   N: number of images (default: 1000)
REM   min_scale_down: min background scale (default: 0.5)
REM   iou_threshold: IoU threshold for logos (default: 0.4)
REM   max_neg: max distractors per image (default: 15)

set N_VAL=%1
if "%N_VAL%"=="" set N_VAL=1000

set SCALE_VAL=%2
if "%SCALE_VAL%"=="" set SCALE_VAL=0.5

set IOU_VAL=%3
if "%IOU_VAL%"=="" set IOU_VAL=0.4

set NEG_VAL=%4
if "%NEG_VAL%"=="" set NEG_VAL=15

echo Starting synthetic data generation...
echo ==========================================
echo Parameters: N=%N_VAL%, scale_down=%SCALE_VAL%, iou=%IOU_VAL%, max_neg=%NEG_VAL%
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

docker run -v "%SYNTH_PATH%:/app/synthesis" -v "%DATA_PATH%:/app/data" --rm medphisiker/tbank-synth:latest python gen_synth.py --N %N_VAL% --min_scale_down %SCALE_VAL% --iou_threshold %IOU_VAL% --max_neg %NEG_VAL%

echo.
echo Synthetic data generation completed!
echo Generated data saved to: data/data_synt/
echo Features: multi-logo, distractors, random resize, advanced augmentations
echo ==========================================