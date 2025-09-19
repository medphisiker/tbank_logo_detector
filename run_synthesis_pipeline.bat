@echo off
REM Synthesis Pipeline Runner - runs all synthesis scripts in Docker
REM Usage: run_synthesis_pipeline.bat [N] — N optional, default 20

set N_VAL=%1
if "%N_VAL%"=="" set N_VAL=20

echo Starting T-Bank Logo Synthesis Pipeline...
echo ==========================================

echo Step 1: Cropping logos from official images...
docker run -v "%CD%/data_preparation/synthesis:/app/synthesis" -v "%CD%/data:/app/data" --rm tbank-synth python crop_logos.py

echo.
echo Step 2: Downloading background images...
docker run -v "%CD%/data_preparation/synthesis:/app/synthesis" -v "%CD%/data:/app/data" --rm tbank-synth python download_backgrounds.py

echo.
echo Step 3: Generating %N_VAL% synthetic images...
docker run -v "%CD%/data_preparation/synthesis:/app/synthesis" -v "%CD%/data:/app/data" --rm tbank-synth python gen_synth.py --N %N_VAL%

echo.
echo Synthesis pipeline completed!
echo Generated data saved to: data/data_synt/
echo ==========================================