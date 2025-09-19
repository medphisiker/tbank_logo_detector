@echo off
REM Build and run the synth container
REM Usage: run_synth.bat [N] — N optional, default 20

set DIR=%~dp0
cd /d "%DIR%.."


REM Run with volume mount of entire project and N env
set N_VAL=%1
if "%N_VAL%"=="" set N_VAL=20
docker run --rm -v "%CD%":/workspace -e N=%N_VAL% tbank-synth

REM Example: run_synth.bat 10 for test