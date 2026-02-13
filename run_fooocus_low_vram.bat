@echo off

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo venv not found. Please ensure you have a virtual environment at .\venv
    pause
    exit /b
)

python launch.py --skip-model-load %*

echo.
echo Fooocus closed.
pause
