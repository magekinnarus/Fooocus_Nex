@echo off

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo venv not found. Please ensure you have a virtual environment at .\venv
    pause
    exit /b
)

python tools\model_extractor.py %*

echo.
echo Extraction complete.
pause
