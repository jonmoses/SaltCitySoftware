@echo off
REM One-click ESM-2 3B embedding for the viral + bacterial datasets (Windows + RTX 4080).
REM Double-click this file, or run it from a terminal. Window stays open at the end.

REM Move to the repo root (this .bat lives in scripts\).
cd /d "%~dp0\.."

echo Running ESM-2 3B embedding. First run downloads ~5.6 GB of model weights.
echo This is a multi-hour job for the full bacterial set. It checkpoints and
echo can be safely re-run to resume if interrupted.
echo.

REM Prefer a local virtualenv if present, else fall back to whatever `python` is on PATH.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\embed_3b.py
) else (
    python scripts\embed_3b.py
)

echo.
echo Finished (or stopped). Press any key to close this window.
pause >nul
