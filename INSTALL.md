 # Installation Guide
 
 This guide walks through setting up Nexfocus from scratch. Each step is
 manual -- the launch scripts only verify, they do not install. If you get
 stuck, ask an AI assistant for help with your specific platform.
 
 ## Prerequisites
 
- An NVIDIA GPU with updated drivers
- Internet connection for downloading models and dependencies
- Approximately 30 GB of free disk space

 > **NVIDIA GPUs only.** Nexfocus currently supports NVIDIA GPUs exclusively.
 > If the next scout mission (a native C++ tensor layer) succeeds, it will
 > remove the PyTorch dependency that restricts us to NVIDIA hardware,
 > opening the door to AMD, Intel, and other GPU platforms.
 
 See the [System Requirements](readme.md#system-requirements) table for
 GPU and RAM requirements for your target workload.
 
 ---
 
 ## Step 1: Install Python
 
 **Recommended:** Python 3.12. Any Python 3.10 or newer works.
 
 Download from [python.org](https://www.python.org/downloads/).
 
 - **Windows:** Check "Add Python to PATH" during installation.
 - **Linux:** Use your package manager (`apt install python3.12` on
   Ubuntu/Debian) or the official installer.
 
 Verify:
 
 ```bash
 python --version
 ```
 
 For detailed platform-specific help, ask an AI assistant: "How do I install
 Python 3.12 on Windows?" or "How do I install Python 3.12 on Ubuntu?"
 
 ---
 
 ## Step 2: Create a Virtual Environment
 
 Open a terminal in the Nexfocus directory and run:
 
 ```bash
 python -m venv venv
 ```
 
 Activate the environment:
 
 **Windows:**
 ```bash
 venv\Scripts\activate
 ```
 
 **Linux:**
 ```bash
 source venv/bin/activate
 ```
 
 Your terminal prompt should now show `(venv)`. Keep this terminal open for
 the remaining steps.
 
 ---
 
 ## Step 3: Install PyTorch
 
 PyTorch must be installed with CUDA support. The exact command depends on
 your GPU and its CUDA compute capability.
 
 **Most modern GPUs (GTX 1660 / RTX 2000 series and newer):**
 
 ```bash
 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
 ```
 
 **Older GPUs (GTX 10-series, including GTX 1050):**
 
 ```bash
 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
 ```
 
 To find the right command for your hardware, visit
 [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
 or ask an AI assistant: "What PyTorch CUDA version should I install for my
 NVIDIA GPU?"
 
 Verify:
 
 ```bash
 python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
 ```
 
 ---
 
 ## Step 4: Install xformers
 
 xformers provides optimized attention kernels. Install the version that
 matches your PyTorch installation:
 
 ```bash
 pip install xformers --index-url https://download.pytorch.org/whl/cu124
 ```
 
 Replace `cu124` with `cu118` if you used the CUDA 11.8 PyTorch index.
 
 xformers is strongly recommended but not required. If installation fails,
 the application will use PyTorch's built-in attention (slower).
 
 Verify:
 
 ```bash
 python -c "import xformers; print('xformers', xformers.__version__)"
 ```
 
 ---
 
 ## Step 5: Install uv
 
 uv is a fast Python package manager used for dependency resolution:
 
 ```bash
 pip install uv
 ```
 
 ---
 
 ## Step 6: Install Python Dependencies
 
 ```bash
 uv pip install -r requirements_versions.txt
 ```
 
 This installs all required packages (Gradio, transformers, diffusers
 dependencies, and utilities). PyTorch and xformers must already be installed
 from Steps 3 and 4 before running this command.
 
 ---
 
 ## Step 7: Set Up API Keys (Optional)
 
 Copy the template file to create your environment configuration:
 
 ```bash
 copy .env_template .env    # Windows
 cp .env_template .env      # Linux
 ```
 
 Edit `.env` and add your API keys:
 
 - `HUGGINGFACE_TOKEN` -- for accessing gated models on Hugging Face.
   Generate at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
 - `CIVITAI_TOKEN` -- for downloading models from CivitAI.
   Generate at [civitai.com/user/account](https://civitai.com/user/account) (API Keys section).
 - `ZROK_TOKEN` -- optional, for exposing the Gradio UI publicly from Colab.
   Generate at [api.zrok.io](https://api.zrok.io).
 
 Tokens are optional -- the application works without them, but some gated
 models or higher-rate CivitAI downloads will be unavailable.
 
 ---
 
 ## Verify Installation
 
 Run the launch script to check that everything is correctly installed:
 
 **Windows:**
 ```bash
 launch.bat
 ```
 
**Linux:**
```bash
./launch.sh
```
 
 If you get a "permission denied" error, run `bash launch.sh` instead.
 
 The script verifies each component and reports `[OK]`, `[WARN]`, or
 `[FAIL]`. If all checks pass, the application starts automatically.
 
 The first launch downloads several model files (SDXL checkpoint, VAE,
 upscalers, and default LoRAs). This may take 10--30 minutes depending on
 your connection.
 
 ---
 
 ## Google Colab
 
 No local installation required. Click the badge to open the ready-to-run
 notebook:
 
 [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1J7ZF8gu2fquNAcrhsw0U5ITM2fp2muhtl#scrollTo=g8uPGq2Fgd5U)
 
 The notebook handles all installation steps automatically within Colab's
 environment.
