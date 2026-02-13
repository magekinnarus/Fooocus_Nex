import os
import argparse
import subprocess
import glob
import torch
from safetensors.torch import load_file, save_file
import shutil

def run_command(command, cwd=None, shell=True):
    print(f"Running: {command}")
    try:
        subprocess.run(command, cwd=cwd, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        raise e

def extract_components(model_path, output_dir, model_name):
    print(f"Loading model: {model_path}")
    state_dict = load_file(model_path)
    
    unet_output = os.path.join(output_dir, 'unet', f'{model_name}_unet.safetensors')
    clips_output = os.path.join(output_dir, 'clips', f'{model_name}_clips.safetensors')
    
    # Create subdirs
    os.makedirs(os.path.dirname(unet_output), exist_ok=True)
    os.makedirs(os.path.dirname(clips_output), exist_ok=True)

    # 1. Extract UNet
    # SDXL UNet prefixes to keep: those NOT starting with conditioner or first_stage_model
    # But filtering by exclusion is safer for SDXL structure
    print("Extracting UNet...")
    unet_dict = {}
    for k, v in state_dict.items():
        if not k.startswith("conditioner.") and not k.startswith("first_stage_model."):
            unet_dict[k] = v
    
    save_file(unet_dict, unet_output)
    print(f"Saved UNet to {unet_output}")
    del unet_dict

    # 2. Extract and Combine CLIPs (CLIP-G and CLIP-L)
    # CLIP-L prefix: "conditioner.embedders.0.transformer."
    # CLIP-G prefix: "conditioner.embedders.1.model."
    
    print("Extracting and Combining CLIPs...")
    clips_dict = {}
    
    # We need to map these to the format expected by ComfyUI/Fooocus GGUF loader
    # Or keep them as is? User said "combine the clips into one safetensors file".
    # And "Simplify the universal convert... just need SDXL model extraction with the original layer names"
    # Keeping original names is simplest.
    
    for k, v in state_dict.items():
        if k.startswith("conditioner.embedders."):
             clips_dict[k] = v

    save_file(clips_dict, clips_output)
    print(f"Saved CLIPs to {clips_output}")
    del clips_dict
    del state_dict

    return unet_output

def convert_to_gguf(unet_path, output_dir, model_name, tools_dir):
    f16_output_dir = os.path.join(output_dir, 'fp16')
    os.makedirs(f16_output_dir, exist_ok=True)
    f16_output = os.path.join(f16_output_dir, f'{model_name}_F16.gguf')
    
    convert_script = os.path.join(tools_dir, 'llama.cpp', 'convert-to-gguf.py')
    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"Could not find convert-to-gguf.py at {convert_script}")

    # Call convert-to-gguf.py
    # --model-type unet for SDXL unet
    cmd = f'python "{convert_script}" --model-type unet --src "{unet_path}" --dst "{f16_output}"'
    run_command(cmd)
    
    return f16_output

def quantize_model(f16_path, output_dir, model_name, formats, tools_dir):
    quantized_output_dir = os.path.join(output_dir, 'quantized')
    os.makedirs(quantized_output_dir, exist_ok=True)
    
    # Hardcoded path as requested
    quantize_bin = os.path.join(tools_dir, 'llama.cpp', 'build', 'bin', 'Release', 'llama-quantize.exe')
    if not os.path.exists(quantize_bin):
        # Fallback to check without Release folder (in case of Debug or different build)
        quantize_bin_alt = os.path.join(tools_dir, 'llama.cpp', 'build', 'bin', 'llama-quantize.exe')
        if os.path.exists(quantize_bin_alt):
            quantize_bin = quantize_bin_alt
        else:
             raise FileNotFoundError(f"Could not find llama-quantize.exe at {quantize_bin} or {quantize_bin_alt}")

    for fmt in formats:
        # Naming convention: Q8_0 -> _Q8.gguf, others -> _{fmt}.gguf
        suffix = "Q8" if fmt == "Q8_0" else fmt
        out_file = os.path.join(quantized_output_dir, f'{model_name}_{suffix}.gguf')
        
        cmd = f'"{quantize_bin}" "{f16_path}" "{out_file}" {fmt}'
        print(f"Quantizing to {fmt}...")
        run_command(cmd)

def main():
    parser = argparse.ArgumentParser(description="Extract and Quantize SDXL Models")
    parser.add_argument("--input", required=True, help="Path to input safetensors model")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--formats", nargs="+", default=["Q8_0", "Q5_K_M", "Q4_K_M"], help="Quantization formats (default: Q8_0 Q5_K_M Q4_K_M)")
    
    args = parser.parse_args()
    
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.exists(args.input):
        print(f"Input model not found: {args.input}")
        return

    model_name = os.path.splitext(os.path.basename(args.input))[0]
    
    try:
        # 1. Extract
        unet_path = extract_components(args.input, args.output, model_name)
        
        # 2. Convert to GGUF F16
        f16_path = convert_to_gguf(unet_path, args.output, model_name, tools_dir)
        
        # 3. Quantize
        quantize_model(f16_path, args.output, model_name, args.formats, tools_dir)
        
        print("\nAll operations completed successfully!")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")

if __name__ == "__main__":
    main()
