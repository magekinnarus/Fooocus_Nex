import os
import subprocess
import sys

def run_command(command, cwd=None, shell=True):
    print(f"Running: {command}")
    try:
        subprocess.run(command, cwd=cwd, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(e)
        sys.exit(1)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # If script is in 'tools', use current dir. If in root, append 'tools'.
    if os.path.basename(base_dir).lower() == "tools":
        tools_dir = base_dir
    else:
        tools_dir = os.path.join(base_dir, "tools")
        os.makedirs(tools_dir, exist_ok=True)
    
    llama_dir = os.path.join(tools_dir, "llama.cpp")

    # 1. Clone llama.cpp if it doesn't exist
    if not os.path.exists(llama_dir):
        print("Cloning llama.cpp...")
        run_command("git clone https://github.com/ggerganov/llama.cpp.git", cwd=tools_dir)
    else:
        print("llama.cpp directory already exists. Skipping clone.")

    # 1.5 Install dependencies
    print("Installing python dependencies...")
    run_command(f"{sys.executable} -m pip install gguf gguf-py", cwd=tools_dir)

    # 2. Checkout specific tag (b3962)
    print("Checking out tag b3962...")
    run_command("git checkout tags/b3962", cwd=llama_dir)

    # 3. Download patch file
    patch_url = "https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/tools/lcpp.patch"
    patch_file = os.path.join(tools_dir, "lcpp.patch")
    
    # Check if patch file exists
    if not os.path.exists(patch_file):
        print(f"Downloading patch from {patch_url}...")
        # Using curl to download
        run_command(f"curl -L --ssl-no-revoke -o {patch_file} {patch_url}", cwd=tools_dir)
    else:
        print("Patch file already exists.")

    # 3.5 Download convert-to-gguf.py
    convert_url = "https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/tools/convert.py"
    convert_file = os.path.join(llama_dir, "convert.py")
    print(f"Downloading convert.py from {convert_url}...")
    run_command(f"curl -L --ssl-no-revoke -o {convert_file} {convert_url}", cwd=tools_dir)

    # 4. Apply patch
    print("Applying patch...")
    # Check if patch is already applied or if we can apply it
    # We'll try to apply it. If it fails, it might be already applied or conflicting.
    # 'git apply' doesn't have a great "check if applied" flag easily without erroring.
    # We will try to apply and ignore whitespace errors.
    try:
        run_command(f"git apply --ignore-space-change --ignore-whitespace {patch_file}", cwd=llama_dir)
        print("Patch applied successfully.")
    except SystemExit:
        print("Warning: Failed to apply patch. It might have already been applied.")

    print("\n" + "="*50)
    print("Setup Complete!")
    print("Now you need to build llama.cpp.")
    print("Instructions for Windows:")
    print(f"1. cd {llama_dir}")
    print("2. mkdir build")
    print("3. cd build")
    print("4. cmake ..")
    print("5. cmake --build . --config Release -j 4 --target llama-quantize")
    print("="*50)

if __name__ == "__main__":
    main()
