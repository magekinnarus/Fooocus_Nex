import os
import argparse
import subprocess
import torch
import json
from safetensors.torch import load_file, save_file

def run_command(command, cwd=None, shell=True):
    print(f"Running: {command}")
    try:
        subprocess.run(command, cwd=cwd, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        raise e

def extract_components(model_path, output_dir, model_name, overwrite=False):
    print(f"Loading model: {model_path}")
    state_dict = load_file(model_path)
    
    unet_output = os.path.join(output_dir, 'unet', f'{model_name}_unet.safetensors')
    clips_output = os.path.join(output_dir, 'clips', f'{model_name}_clips.safetensors')
    
    os.makedirs(os.path.dirname(unet_output), exist_ok=True)
    os.makedirs(os.path.dirname(clips_output), exist_ok=True)

    # Overwrite check for extracted files
    for path in [unet_output, clips_output]:
        if os.path.exists(path) and overwrite:
            print(f"Overwriting existing file: {path}")
            os.remove(path)

    # 1. Extract UNet
    print("Extracting UNet...")
    unet_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model_ema"):
             continue
        if not k.startswith("conditioner.") and not k.startswith("first_stage_model."):
            key = k if k.startswith("model.diffusion_model.") else f"model.diffusion_model.{k}"
            unet_dict[key] = v
    
    save_file(unet_dict, unet_output)
    print(f"Saved UNet to {unet_output}")
    del unet_dict

    # 2. Extract and Combine CLIPs
    print("Extracting and Combining CLIPs...")
    clips_dict = {k: v for k, v in state_dict.items() if k.startswith("conditioner.embedders.")}
    save_file(clips_dict, clips_output)
    print(f"Saved Combined CLIPs to {clips_output}")
    
    del clips_dict
    del state_dict
    return unet_output

def convert_to_gguf(unet_path, output_dir, model_name, tools_dir, overwrite=False):
    f16_output_dir = os.path.join(output_dir, 'fp16')
    os.makedirs(f16_output_dir, exist_ok=True)
    f16_output = os.path.join(f16_output_dir, f'{model_name}_F16.gguf')
    
    if os.path.exists(f16_output) and overwrite:
        print(f"Overwriting existing FP16: {f16_output}")
        os.remove(f16_output)

    convert_script = os.path.join(tools_dir, 'llama.cpp', 'convert.py')
    # Use --src and --dst only
    cmd = f'python "{convert_script}" --src "{unet_path}" --dst "{f16_output}"'
    run_command(cmd)
    
    return f16_output

def quantize_model(f16_path, output_dir, model_name, formats, tools_dir, overwrite=False):
    quantized_output_dir = os.path.join(output_dir, 'quantized')
    os.makedirs(quantized_output_dir, exist_ok=True)
    
    # Path to your confirmed Debug executable
    quantize_bin = os.path.join(tools_dir, 'llama.cpp', 'build', 'bin', 'Debug', 'llama-quantize.exe')

    for fmt in formats:
        suffix = "Q8" if fmt == "Q8_0" else fmt
        out_file = os.path.join(quantized_output_dir, f'{model_name}_{suffix}.gguf')
        
        if os.path.exists(out_file) and overwrite:
            print(f"Overwriting existing quantized model: {out_file}")
            os.remove(out_file)
            
        cmd = f'"{quantize_bin}" "{f16_path}" "{out_file}" {fmt}'
        run_command(cmd)

def main():
    parser = argparse.ArgumentParser(description="SDXL Extractor & Quantizer")
    parser.add_argument("--config", default="extraction_output_config.json", help="Path to config JSON")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files if they exist")
    args = parser.parse_args()
    
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        input_path = config.get("input")
        output_dir = config.get("output")
        formats = config.get("formats", ["Q8_0", "Q5_K_M", "Q4_K_M"])
        # Check if overwrite is also defined in JSON, otherwise use CLI flag
        overwrite = config.get("overwrite", args.overwrite)
    else:
        print(f"Error: Configuration file {args.config} not found.")
        return

    model_name = os.path.splitext(os.path.basename(input_path))[0]
    
    try:
        unet_path = extract_components(input_path, output_dir, model_name, overwrite)
        f16_path = convert_to_gguf(unet_path, output_dir, model_name, tools_dir, overwrite)
        quantize_model(f16_path, output_dir, model_name, formats, tools_dir, overwrite)
        print("\nAll components processed successfully!")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")

if __name__ == "__main__":
    main()