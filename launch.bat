@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

echo ========================================
echo   Nexfocus Launcher
echo ========================================
echo.

REM --- Check system Python ---
set "PYTHON_CMD="
for /f "tokens=*" %%p in ('where python 2^>nul') do (
    set "PYTHON_CMD=%%p"
    goto :found_python
)
for /f "tokens=*" %%p in ('where python3 2^>nul') do (
    set "PYTHON_CMD=%%p"
    goto :found_python
)

:found_python
if "%PYTHON_CMD%"=="" (
    echo [FAIL] Python not found.
    echo        Install Python 3.10+ ^(recommended: 3.12^).
    echo        Download: https://www.python.org/downloads/
    echo        See INSTALL.md for detailed instructions.
    pause
    exit /b 1
)

"%PYTHON_CMD%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>nul
if errorlevel 1 (
    for /f "tokens=*" %%v in ('"%PYTHON_CMD%" --version 2^>^&1') do set "PYTHON_VER=%%v"
    echo [FAIL] Python 3.10 or newer is required. Found: !PYTHON_VER!
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('"%PYTHON_CMD%" --version 2^>^&1') do set "PYTHON_VER=%%v"
echo [OK] !PYTHON_VER!

REM --- Check NVIDIA driver ---
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARN] nvidia-smi not found. GPU acceleration may not be available.
    echo        Ensure NVIDIA drivers are installed: https://www.nvidia.com/download/
) else (
    echo [OK] NVIDIA driver
)

REM --- Check venv ---
if not exist "%VENV_PYTHON%" (
    echo [FAIL] Virtual environment not found at venv\
    echo        Create it: python -m venv venv
    echo        Then install dependencies: see INSTALL.md
    pause
    exit /b 1
)

"%VENV_PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>nul
if errorlevel 1 (
    echo [FAIL] The virtual environment must use Python 3.10 or newer.
    echo        Recreate venv with a supported Python version.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('"%VENV_PYTHON%" --version 2^>^&1') do set "VENV_PYTHON_VER=%%v"
echo [OK] Virtual environment ^(!VENV_PYTHON_VER!^)

REM --- Check PyTorch ---
"%VENV_PYTHON%" -c "import torch; version = tuple(map(int, torch.__version__.split('+', 1)[0].split('.')[:3])); assert version >= (2, 5, 1), 'PyTorch below 2.5.1'; assert torch.cuda.is_available(), 'CUDA not available'" 2>nul
if errorlevel 1 (
    echo [FAIL] PyTorch 2.5.1+ with CUDA not found in venv.
    echo        Install the validated baseline: see INSTALL.md
    echo        Guide: https://pytorch.org/get-started/locally/
    pause
    exit /b 1
)
"%VENV_PYTHON%" -c "import torch; print('[OK] PyTorch', torch.__version__, 'with CUDA')"

REM --- Check xformers ---
"%VENV_PYTHON%" -c "import xformers" 2>nul
if errorlevel 1 (
    echo [WARN] xformers not found. Attention will be slower.
    echo        Install: pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124
) else (
    echo [OK] xformers
)

REM --- Check Aria2 ---
where aria2c >nul 2>&1
if errorlevel 1 if not exist "%VENV_DIR%\Scripts\aria2c.exe" (
    echo [INFO] Installing Aria2 download manager ...
    set "ARIA2_URL=https://github.com/aria2/aria2/releases/download/release-1.37.0/aria2-1.37.0-win-64bit-build1.zip"
    set "ARIA2_ZIP=%TEMP%\nexfocus_aria2.zip"
    set "ARIA2_DIR=%TEMP%\nexfocus_aria2_extract"
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '!ARIA2_URL!' -OutFile '!ARIA2_ZIP!' } catch { exit 1 }" >nul 2>&1
    if not errorlevel 1 (
        powershell -Command "Expand-Archive -Path '!ARIA2_ZIP!' -DestinationPath '!ARIA2_DIR!' -Force" >nul 2>&1
        for /d %%d in ("!ARIA2_DIR!\aria2-*") do (
            if exist "%%d\aria2c.exe" copy /Y "%%d\aria2c.exe" "%VENV_DIR%\Scripts\aria2c.exe" >nul 2>&1
        )
        del /Q "!ARIA2_ZIP!" 2>nul
        rmdir /S /Q "!ARIA2_DIR!" 2>nul
    )
)

where aria2c >nul 2>&1
if not errorlevel 1 (
    echo [OK] aria2c
) else if exist "%VENV_DIR%\Scripts\aria2c.exe" (
    echo [OK] aria2c
) else (
    echo [WARN] Could not install Aria2 automatically.
    echo        Model downloads will use the slower fallback.
    echo        Install manually: https://github.com/aria2/aria2/releases
)

REM --- Check uv ---
"%VENV_PYTHON%" -m pip show uv >nul 2>&1
if errorlevel 1 (
    echo [FAIL] uv package manager not found in venv.
    echo        Install: python -m pip install uv
    pause
    exit /b 1
)
echo [OK] uv

echo.
echo All checks passed. Launching Nexfocus ...
echo.
"%VENV_PYTHON%" "%SCRIPT_DIR%launch.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%
