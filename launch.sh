 #!/usr/bin/env bash
 set -euo pipefail
 
 SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
 VENV_DIR="$SCRIPT_DIR/venv"
 
 echo "========================================"
 echo "  Nexfocus Launcher"
 echo "========================================"
 echo ""
 
 # --- Check Python ---
 PYTHON_CMD=""
 for cmd in python3 python; do
     if command -v "$cmd" &>/dev/null; then
         PYTHON_CMD="$cmd"
         break
     fi
 done
 
 if [ -z "$PYTHON_CMD" ]; then
     echo "[FAIL] Python not found."
     echo "       Install Python 3.10+ (recommended: 3.12)."
     echo "       Download: https://www.python.org/downloads/"
     echo "       See INSTALL.md for detailed instructions."
     exit 1
 fi
 
 echo "[ OK ] $($PYTHON_CMD --version)"
 
 # --- Check NVIDIA driver ---
 if ! nvidia-smi &>/dev/null; then
     echo "[WARN] nvidia-smi not found. GPU acceleration may not be available."
     echo "       Ensure NVIDIA drivers are installed: https://www.nvidia.com/download/"
 else
     echo "[ OK ] NVIDIA driver"
 fi
 
 # --- Check venv ---
 if [ ! -f "$VENV_DIR/bin/python" ]; then
     echo "[FAIL] Virtual environment not found at venv/"
     echo "       Create it: python3 -m venv venv"
     echo "       Then install dependencies: see INSTALL.md"
     exit 1
 fi
 echo "[ OK ] Virtual environment"
 
 source "$VENV_DIR/bin/activate"
 
 # --- Check PyTorch ---
 if python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
     TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
     echo "[ OK ] PyTorch $TORCH_VER with CUDA"
 else
     echo "[FAIL] PyTorch with CUDA not found in venv."
     echo "       Install PyTorch: see INSTALL.md"
     echo "       Guide: https://pytorch.org/get-started/locally/"
     exit 1
 fi
 
 # --- Check xformers ---
 if python -c "import xformers" 2>/dev/null; then
     echo "[ OK ] xformers"
 else
     echo "[WARN] xformers not found. Attention will be slower."
     echo "       Install: pip install xformers --index-url https://download.pytorch.org/whl/cu124"
 fi
 
# --- Check Aria2 ---
if command -v aria2c &>/dev/null; then
    echo "[ OK ] aria2c"
else
     echo "[INFO] Installing Aria2 from bundled packages ..."
     ARIA2_DIR="$SCRIPT_DIR/extras/aria2_packages"
     if [ -d "$ARIA2_DIR" ]; then
         if dpkg -i "$ARIA2_DIR"/libc-ares2_*.deb &>/dev/null && \
            dpkg -i "$ARIA2_DIR"/libaria2-0_*.deb &>/dev/null && \
            dpkg -i "$ARIA2_DIR"/aria2_*.deb &>/dev/null; then
             echo "[ OK ] aria2c"
         elif command -v sudo &>/dev/null; then
             sudo dpkg -i "$ARIA2_DIR"/libc-ares2_*.deb &>/dev/null || true
             sudo dpkg -i "$ARIA2_DIR"/libaria2-0_*.deb &>/dev/null || true
             sudo dpkg -i "$ARIA2_DIR"/aria2_*.deb &>/dev/null || true
             if command -v aria2c &>/dev/null; then
                 echo "[ OK ] aria2c"
             else
                 echo "[WARN] Could not install Aria2. Downloads will use slower fallback."
                 echo "       Install manually: sudo apt install aria2"
             fi
         else
             echo "[WARN] Could not install Aria2 (sudo not available)."
             echo "       Install manually: sudo apt install aria2"
         fi
     else
         echo "[WARN] Aria2 packages not found. Downloads will use slower fallback."
         echo "       Install manually: sudo apt install aria2"
     fi
fi
 
 # --- Check uv ---
 if python -m pip show uv &>/dev/null; then
     echo "[ OK ] uv"
 else
     echo "[FAIL] uv package manager not found in venv."
     echo "       Install: pip install uv"
     exit 1
 fi
 
 echo ""
 echo "All checks passed. Launching Nexfocus ..."
 echo ""
 python "$SCRIPT_DIR/launch.py" "$@"
