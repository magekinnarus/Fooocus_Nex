# Installation Guide

This guide walks through setting up Nexfocus from scratch. The launchers verify
the environment and install Aria2 when needed; they do not install Python,
PyTorch, xformers, uv, or the Python dependencies.

## Prerequisites

- An internet connection for downloading the repository, dependencies, and models.
- Enough storage for your chosen models. One SDXL checkpoint is about 6.5 GB,
  Flux Fill is about 12.7 GB, and FP16 T5-XXL is about 9.5 GB. Total usage
  depends on how many checkpoints and LoRAs you install.

> **NVIDIA GPUs only.** Nexfocus currently requires an NVIDIA GPU with current
> drivers. See the [System Requirements](readme.md#system-requirements) table
> for the GPU and RAM needed by each runtime posture.

Open Command Prompt, PowerShell, or Windows Terminal on Windows, or a terminal
application on Linux. Run each remaining command from that terminal.

---

## Step 1: Clone the Repository

**Windows and Linux:**

```bash
git clone https://github.com/magekinnarus/Nexfocus.git
cd Nexfocus
```

If Git is not installed, get it from [git-scm.com](https://git-scm.com/downloads)
or download and extract the repository ZIP from GitHub.

Verify that the terminal is inside the clone:

```bash
git remote get-url origin
```

Expected result: `https://github.com/magekinnarus/Nexfocus.git`.
If you used the ZIP instead, confirm that the extracted Nexfocus directory
contains `launch.py`.

---

## Step 2: Install Python

Python 3.12 is recommended. The launchers require Python 3.10 or newer. The
validated PyTorch 2.5.1 baseline supports Python 3.10 through 3.12; Python
3.13+ requires a newer compatible PyTorch/xformers combination and is not part
of the validated baseline.

- **Windows:** Download Python from [python.org](https://www.python.org/downloads/).
  Enable both **Add Python to PATH** and the **Python Launcher** during setup.
- **Ubuntu/Debian Linux:** Install the distribution's Python 3 and venv
  packages. Ubuntu 22.04 provides Python 3.10; Ubuntu 24.04 provides 3.12:

  ```bash
  sudo apt update
  sudo apt install python3 python3-venv
  ```

Verify:

**Windows:**

```bat
py -3 --version
```

**Linux:**

```bash
python3 --version
```

The result must be Python 3.10 or newer. See the
[official Python setup guides](https://www.python.org/about/gettingstarted/)
for other platforms.

---

## Step 3: Create a Virtual Environment

**Windows:**

```bat
py -3 -m venv venv
venv\Scripts\activate
venv\Scripts\python.exe --version
```

**Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
venv/bin/python --version
```

The verification command must report Python 3.10 or newer. Keep the
environment activated for Steps 4-8. For details, see the
[Python venv documentation](https://docs.python.org/3/library/venv.html).

---

## Step 4: Install PyTorch with CUDA

The validated development floor is:

- PyTorch `2.5.1+cu124`
- torchvision `0.20.1+cu124`
- CUDA runtime `12.4`

This combination was used throughout development from the GTX 1050 reference
machine through newer NVIDIA GPUs. PyTorch wheels include the CUDA runtime; a
separate CUDA Toolkit installation is not required.

**Windows and Linux, with the virtual environment activated:**

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The version must be at least `2.5.1`, and `CUDA available` must be `True`.
For another Python or GPU combination, use the
[official PyTorch selector](https://pytorch.org/get-started/locally/) and do
not install a PyTorch version below 2.5.1.

---

## Step 5: Install xformers

Install the xformers build validated with the PyTorch 2.5.1/cu124 baseline:

```bash
python -m pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import xformers; print(xformers.__version__)"
```

Expected result: `0.0.28.post3`. xformers is strongly recommended; without it,
Nexfocus falls back to slower PyTorch attention. See the
[xformers project](https://github.com/facebookresearch/xformers) for
compatibility information.

---

## Step 6: Install uv

Install uv inside the active virtual environment:

```bash
python -m pip install uv
```

Verify:

```bash
python -m pip show uv
```

See the [uv documentation](https://docs.astral.sh/uv/) for additional
installation options.

---

## Step 7: Install Python Dependencies

PyTorch and xformers must already be installed before this step:

```bash
uv pip install -r requirements_versions.txt
```

Verify representative application dependencies:

```bash
python -c "import gradio, transformers, safetensors, dotenv; print('Dependencies OK')"
```

If resolution fails, confirm that the virtual environment is active and review
the prerequisite header in
[requirements_versions.txt](requirements_versions.txt).

---

## Step 8: Set Up API Keys

All tokens are optional. Open `.env_template` in a text editor:

- **Windows:** Notepad, Notepad++, or another text editor.
- **Linux:** Text Editor, Gedit, Kate, or another text editor.

Add the credentials you use, then choose **Save As** and save the file as
`.env` in the Nexfocus repository folder. Ensure the editor does not append
`.txt` to the filename. Leave credentials you do not use as empty strings;
empty values remain disabled and do not cause authentication attempts.

Available keys:

- `HUGGINGFACE_TOKEN` -- gated Hugging Face models:
  [generate a token](https://huggingface.co/settings/tokens).
- `CIVITAI_TOKEN` -- authenticated CivitAI downloads:
  [generate an API key](https://civitai.com/user/account).
- `ZROK_TOKEN` -- optional Colab tunnel:
  [manage zrok credentials](https://api.zrok.io).

Reopen `.env` in the text editor and confirm that it contains the
`CIVITAI_TOKEN`, `HUGGINGFACE_TOKEN`, and `ZROK_TOKEN` lines and is saved in
the same folder as `launch.py`.

---

## Verify the Environment and Launch

**Windows:**

```bat
launch.bat
```

**Linux:**

```bash
./launch.sh
```

The launcher reports `[OK]`, `[WARN]`, or `[FAIL]` for every check. It installs
Aria2 automatically when possible and does not install the other prerequisites.
If all hard requirements pass, it starts Nexfocus.

The first launch downloads the selected model and required support files.
Download time depends on your connection and model selection.

---

## Google Colab

No local installation is required. Use the
[official Colab notebook](readme.md#run-on-google-colab) instead.
