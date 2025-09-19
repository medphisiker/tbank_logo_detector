@echo off
REM Synthesis Pipeline Runner - runs all synthesis scripts in Docker
REM Usage: run_synthesis_pipeline.bat [N] [min_scale_down] [iou_threshold] [max_neg]
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

echo Starting T-Bank Logo Synthesis Pipeline...
echo ==========================================
echo Parameters: N=%N_VAL%, scale_down=%SCALE_VAL%, iou=%IOU_VAL%, max_neg=%NEG_VAL%
echo ==========================================

echo Step 1: Cropping logos from official images...
docker run -v "%CD%/data_preparation/synthesis:/app/synthesis" -v "%CD%/data:/app/data" --rm tbank-synth python crop_logos.py

echo.
echo Step 2: Downloading high-res background images...
docker run -v "%CD%/data_preparation/synthesis:/app/synthesis" -v "%CD%/data:/app/data" --rm tbank-synth python download_backgrounds.py --num 1000 --size 1920

echo.
echo Step 3: Preparing background objects (distractors)...
docker run -v "%CD%/data_preparation/synthesis:/app/synthesis" -v "%CD%/data:/app/data" --rm tbank-synth python prepare_background_objects.py --num 200

echo.
echo Step 4: Generating %N_VAL% synthetic images with advanced features...
docker run -v "%CD%/data_preparation/synthesis:/app/synthesis" -v "%CD%/data:/app/data" --rm tbank-synth python gen_synth.py --N %N_VAL% --min_scale_down %SCALE_VAL% --iou_threshold %IOU_VAL% --max_neg %NEG_VAL%

echo.
echo Synthesis pipeline completed!
echo Generated data saved to: data/data_synt/
echo Features: multi-logo, distractors, random resize, advanced augmentations
echo ==========================================