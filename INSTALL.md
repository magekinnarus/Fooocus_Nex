# Installation Guide

## Need help? Ask an AI assistant

Copy and paste any installation step into an AI assistant and ask it to guide
you through the step for Windows or Linux. Include any error message you
receive, and ask the assistant to explain one command at a time.

> **Never share API keys, tokens, passwords, or the contents of your `.env`
> file with an AI assistant.**

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

Nexfocus supports compatible PyTorch/CUDA builds at or above the validated
development floor:

- PyTorch `2.5.1+cu124`
- torchvision `0.20.1+cu124`
- CUDA runtime `12.4`

This combination was used throughout development from the GTX 1050 reference
machine through newer NVIDIA GPUs. It is the safe baseline, not a requirement
to stay on an older build. PyTorch wheels include the CUDA runtime; a separate
CUDA Toolkit installation is not required.

### Option A: Use the validated baseline

**Windows and Linux, with the virtual environment activated:**

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

### Option B: Use a newer compatible build

Users with newer hardware may install a newer PyTorch and CUDA wheel. Open the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/),
choose your operating system, Pip, Python, and the CUDA version appropriate for
your hardware, then run the command it provides.

If you are unsure, copy this section into an AI assistant and provide your
operating system, Python version, and exact NVIDIA GPU name. Ask it to help
select an official PyTorch build that is version 2.5.1 or newer.

The Colab T4 environment has also been field-tested with PyTorch
`2.11.0+cu128`, demonstrating that Nexfocus is not restricted to the local
2.5.1/cu124 baseline.

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

The xformers build must be compatible with the PyTorch and CUDA build selected
in Step 4.

### If you installed the validated PyTorch 2.5.1/cu124 baseline

```bash
python -m pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124
```

### If you installed a newer PyTorch/CUDA build

Follow the
[official xformers installation instructions](https://github.com/facebookresearch/xformers#installing-xformers).
The wheel index must match your PyTorch CUDA build. If you are unsure, give an
AI assistant your operating system, Python version, PyTorch version (including
the `+cu...` suffix), and GPU name, then ask it to derive the command from the
official instructions.

The Colab T4 field environment uses xformers `0.0.35` with PyTorch
`2.11.0+cu128`.

Verify:

```bash
python -m xformers.info
```

Confirm that `pytorch.cuda` is available and that at least one
`memory_efficient_attention` implementation reports `available`. xformers is
strongly recommended; without it, Nexfocus falls back to slower PyTorch
attention.

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

The CivitAI token is required for model downloads through the Nexfocus
catalogue and CivitAI API. The Hugging Face and Zrok tokens are optional.

Open `.env_template` in a text editor:

- **Windows:** Notepad, Notepad++, or another text editor.
- **Linux:** Text Editor, Gedit, Kate, or another text editor.

Add your CivitAI token and any optional credentials you use, then choose
**Save As** and save the file as `.env` in the Nexfocus repository folder.
Ensure the editor does not append `.txt` to the filename. Leave optional
credentials you do not use as empty strings; empty values remain disabled and
do not cause authentication attempts.

Available keys:

- `HUGGINGFACE_TOKEN` -- gated Hugging Face models:
  [generate a token](https://huggingface.co/settings/tokens).
- `CIVITAI_TOKEN` -- **required** for catalogue/API model downloads from
  CivitAI:
  [generate an API key](https://civitai.com/user/account).
- `ZROK_TOKEN` -- optional Colab tunnel:
  [manage zrok credentials](https://api.zrok.io).

Reopen `.env` in the text editor and confirm that it contains the
`CIVITAI_TOKEN`, `HUGGINGFACE_TOKEN`, and `ZROK_TOKEN` lines and is saved in
the same folder as `launch.py`.

Without `CIVITAI_TOKEN`, Nexfocus cannot download CivitAI models through its
catalogue. Manual downloads from the CivitAI website are possible, but the
filename and destination must match Nexfocus's catalogue expectations. A file
saved under a different name or model directory may not be recognized, so
configuring the token and using the in-app catalogue is strongly recommended.

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

### Optional: Create a Desktop Shortcut

On Windows, you can double-click `launch.bat` in the Nexfocus repository
whenever you want to start the application. To make a desktop shortcut:

1. Open the Nexfocus repository in File Explorer.
2. Right-click `launch.bat`. On Windows 11, select **Show more options** if
   needed.
3. Select **Send to > Desktop (create shortcut)**.
4. On the desktop, right-click the new shortcut and select **Properties**.
5. On the **Shortcut** tab, select **Change Icon > Browse**, then choose
   `assets\images\Nexfocus_icon.ico` from the Nexfocus repository.
6. Rename the shortcut to **Nexfocus** if desired.

Linux desktops use `.desktop` launcher files instead of Windows shortcuts.
Open a text editor, paste the following block, and replace every
`/absolute/path/to/Nexfocus` with the full path to your cloned repository:

```ini
[Desktop Entry]
Type=Application
Name=Nexfocus
Comment=Launch Nexfocus
Exec="/absolute/path/to/Nexfocus/launch.sh"
Path=/absolute/path/to/Nexfocus
Icon=/absolute/path/to/Nexfocus/assets/images/Nexfocus_icon.png
Terminal=true
Categories=Graphics;
```

Save the file as `Nexfocus.desktop` on your desktop. Then right-click it and
enable **Allow executing file as program**, **Allow Launching**, or
**Trust and Launch**. The wording varies between Linux desktop environments.
If your desktop does not show launcher files, copy this section to an AI
assistant and tell it which Linux distribution and desktop environment you
use.

The first launch downloads the selected model and required support files.
Download time depends on your connection and model selection.

---

## Google Colab

No local installation is required. Use the
[official Colab notebook](readme.md#run-on-google-colab) instead.
