 # Installation Guide
 
 This guide walks through setting up Nexfocus from scratch. Each step is
 manual -- the launch scripts only verify, they do not install. If you get
 stuck, ask an AI assistant for help. You can copy-paste any section of this document as reference for AI to assist you.
 
## Prerequisites
 
 - An NVIDIA GPU with updated drivers.
 - Internet connection for downloading models and dependencies.
 
 > **Why no portable version?** A portable bundle locks the environment to a
 > fixed configuration, making it difficult to add components or adjust
 > settings. It is also unnecessarily heavy, packaging an entire Python
 > installation that most users already have. Following the steps below gives
 > you a working setup you actually understand -- which makes it easier to
 > tweak, troubleshoot, or adapt later.
 
 See the [System Requirements](readme.md#system-requirements) table for
 GPU and RAM requirements for your target workload.

 ## Before You Start
 
 Open a terminal in the directory where you want to clone the repository:
 
 - **Windows:** Open Command Prompt (`cmd`), PowerShell, or Windows Terminal.
 - **Linux:** Open your terminal application.
 
 All commands in this guide are run from this terminal.
 
 ---
 
## Step 1: Clone the Repository
 
 ```bash
 git clone https://github.com/magekinnarus/Nexfocus.git
 cd Nexfocus
 ```
 
 If you do not have Git installed, download it from
 [git-scm.com](https://git-scm.com/downloads) or download the repository as a
 ZIP from GitHub and extract it.
 
 All remaining steps are run from inside the `Nexfocus` directory.
 
 ---
 
## Step 2: Install Python (if you don't have one already installed in your system)
 
**Recommended:** Python 3.12. Any Python 3.10 or newer works.

 > **Python 3.13+:** PyTorch 2.5.x (our recommended baseline) only supports
 > Python 3.9--3.12. If you use Python 3.13 or later, you must install
 > PyTorch 2.6+ with CUDA 12.8 -- see the alternate command in Step 4.
 
 Download from [python.org](https://www.python.org/downloads/).
 
 - **Windows:** Check "Add Python to PATH" during installation.
   Also enable the **Python Launcher** (`py`) option so you can manage
   multiple Python versions on the same system.
 - **Linux:** Use your package manager (`apt install python3.12` on
   Ubuntu/Debian) or the official installer.
 
Your system Python version will create the virtual environment.
 
 Verify:
 
 ```bash
 python --version
 ```
 
 If you have more than one Python version installed, the launcher lets each
 project use its own version without conflict. The below is an example of running an app with py launcher. THE BELOW IS NOT A PART OF THE INSTALLATION. The actual venv creation happens in Step 3:
 
 ```bash
 py -3.10 your_app.py            # Windows: run with Python 3.10
 python3.11 your_app.py          # Linux: run with Python 3.11
 ```
 
 For detailed platform-specific help, ask an AI assistant: "How do I install
 Python 3.12 on Windows?" or "How do I install Python 3.12 on Ubuntu?"
 
 ---
 
 ## Step 3: Create a Virtual Environment
 
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
 
## Step 4: Install PyTorch
 
 PyTorch must be installed with CUDA support. Two versions are supported:
 
 **Recommended baseline -- PyTorch 2.5.x + CUDA 12.4 (validated on GTX 1050 and up):**
 
 ```bash
 pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
 ```
 
 This is the version we develop and validate against. It works on everything
 from a GTX 1050 (3 GB VRAM) to current-generation cards.
 
 **Newer GPUs -- PyTorch 2.6+ + CUDA 12.8 (RTX 3000 series and newer):**
 
 ```bash
 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
 ```
 
 Matches the Colab environment. May offer performance improvements on newer
 hardware.
 
 To find the right command for your hardware, visit
 [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
 or ask an AI assistant: "What PyTorch CUDA version should I install for my
 NVIDIA GPU?"
 
 Verify:
 
 ```bash
 python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
 ```
 
 ---
 
## Step 5: Install xformers
 
 xformers provides optimized attention kernels. Install the version that
 matches your PyTorch installation:
 
 **With PyTorch 2.5.x + CUDA 12.4:**
 
 ```bash
 pip install xformers --index-url https://download.pytorch.org/whl/cu124
 ```
 
 **With PyTorch 2.6+ + CUDA 12.8:**
 
 ```bash
 pip install xformers --index-url https://download.pytorch.org/whl/cu128
 ```
 
 If you installed a different PyTorch version, check the matching
 [xformers wheel](https://pytorch.org/get-started/locally/) or ask an AI
 assistant: "Which xformers version works with PyTorch X.Y.Z and CUDA A.B?"
 
 xformers is strongly recommended but not required. We use xformers as a wrapper for optimized kernels. If installation fails,
 the application will use PyTorch's built-in attention (slower).
 
 Verify:
 
 ```bash
 python -c "import xformers; print('xformers', xformers.__version__)"
 ```
 
 ---
 
 ## Step 6: Install uv
 
 uv is a fast Python package manager used for dependency resolution:
 
 ```bash
 pip install uv
 ```
 
 ---
 
 ## Step 7: Install Python Dependencies
 
 ```bash
 uv pip install -r requirements_versions.txt
 ```
 
 This installs all required packages (Gradio, transformers, diffusers
 dependencies, and utilities). PyTorch and xformers must already be installed
 from Steps 4 and 5 before running this command.
 
 ---
 
 ## Step 8: Set Up API Keys
 
Open .env_template in any text editor such as notepad or notepad++. 
Edit in your keys to create your environment configuration.
Save the edited file as .env in the Nexfocus directory.
 
Adding your API keys:
 
 - `HUGGINGFACE_TOKEN` -- optional for accessing gated models on Hugging Face.
   Generate at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
 - `CIVITAI_TOKEN` -- for downloading models from CivitAI.
   Generate at [civitai.com/user/account](https://civitai.com/user/account) (API Keys section).
 - `ZROK_TOKEN` -- optional, for tunneling in Colab for local browser access of your UI. There are other options (Cloudflare/LocalTunnel/GRadio public server) for tunneling included in the official colab notebook.
   Generate at [api.zrok.io](https://api.zrok.io).
 
 Tokens are optional -- the application works without them, but you will not be able to download CivitAI or gated Huggingface models.
 
 ---
 
 ## Verify installations and Launch App
 
 Run the launch script to start the app. It will automatically verify the installations:
 
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
 
 No local installation required. See the [Colab section in README.md](readme.md#run-on-google-colab).
