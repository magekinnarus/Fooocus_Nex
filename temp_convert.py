# Apache-2.0 (apache.org/licenses/LICENSE-2.0)
# --- MODIFIED from City96's script FOR ggml_native_app PROJECT ---
# This script is now a multi-purpose component converter.
# It converts a single safetensors component (UNet, VAE, CLIP)
# into a single F16 GGUF file.

import os
import gguf
import torch
import logging
import argparse
from tqdm import tqdm
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO)

QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127
MAX_TENSOR_DIMS = 4

class ModelTemplate:
    arch = "invalid"
    shape_fix = False
    keys_detect = [("invalid",)]
    keys_banned = []
    keys_hiprec = []
    keys_ignore = []

    def handle_nd_tensor(self, key, data):
        raise NotImplementedError(f"Tensor detected that exceeds dims supported by C++ code! ({key} @ {data.shape})")

class ModelVAE(ModelTemplate):
    arch = "sd_vae"
    keys_hiprec = ["conv_in.weight", "conv_out.weight", "encoder.conv_in.weight", "decoder.conv_out.weight"]

class ModelClipL(ModelTemplate):
    arch = "clip_l"
    keys_hiprec = ["text_model.embeddings.token_embedding.weight", "text_model.embeddings.position_embedding.weight"]

class ModelClipG(ModelTemplate):
    arch = "clip_g"
    keys_hiprec = [
        "text_model2.embeddings.token_embedding.weight", "text_model2.embeddings.position_embedding.weight",
        "text_model2.final_layer_norm.weight", "text_model2.final_layer_norm.bias",
    ]

class ModelFlux(ModelTemplate):
    arch = "flux"
    keys_detect = [("transformer_blocks.0.attn.norm_added_k.weight",),("double_blocks.0.img_attn.proj.weight",),]
    keys_banned = ["transformer_blocks.0.attn.norm_added_k.weight",]

class ModelSDXL(ModelTemplate):
    arch = "sdxl" # Correct name for the C++ patch
    shape_fix = True
    # --- CORRECTION: Restoring the full, robust detection logic ---
    keys_detect = [
        # Diffusers-style keys
        ("down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight",),
        # Non-diffusers (A1111/ComfyUI) style keys
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight",
            "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight",
        ),
        # Another common key
        ("label_emb.0.0.weight",),
    ]

class ModelSD1(ModelTemplate):
    arch = "sd1" # Correct name for the C++ patch
    shape_fix = True
    # --- CORRECTION: Restoring the full, robust detection logic ---
    keys_detect = [
        # Diffusers-style key
        ("down_blocks.0.downsamplers.0.conv.weight",),
        # Non-diffusers (A1111/ComfyUI) style keys
        (
            "input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight", "input_blocks.9.0.op.weight",
            "output_blocks.2.1.conv.weight", "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"
        ),
    ]

unet_arch_list = [ModelFlux, ModelSDXL, ModelSD1]
direct_arch_map = {"vae": ModelVAE, "clip_l": ModelClipL, "clip_g": ModelClipG}

def detect_unet_arch(state_dict):
    model_arch = None
    for arch in unet_arch_list:
        matched = False
        invalid = False
        for match_list in arch.keys_detect:
            if all(key in state_dict for key in match_list):
                matched = True
                invalid = any(key in state_dict for key in arch.keys_banned)
                break
        if matched:
            assert not invalid, "Model architecture not allowed for conversion!"
            model_arch = arch()
            break
    assert model_arch is not None, "Unknown UNet model architecture!"
    return model_arch

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a single model component (UNet, VAE, CLIP) to a F16 GGUF file.")
    parser.add_argument("--model-type", required=True, choices=["unet", "vae", "clip_l", "clip_g"], help="The type of the model component to convert.")
    parser.add_argument("--src", required=True, help="Source model component .safetensors file.")
    parser.add_argument("--dst", help="Output GGUF file path. If not specified, it will be generated next to the source file.")
    return parser.parse_args()

def strip_unet_prefix(state_dict):
    prefix = None
    for pfx in ["model.diffusion_model.", "model."]:
        if any([x.startswith(pfx) for x in state_dict.keys()]):
            prefix = pfx
            break
    if prefix is not None:
        logging.info(f"UNet state dict prefix found: '{prefix}'. Stripping.")
        return {k.replace(prefix, ""): v for k, v in state_dict.items() if prefix in k}
    else:
        logging.debug("UNet state dict has no prefix to strip.")
        return state_dict

def load_state_dict(path, model_type):
    state_dict = load_file(path)
    if model_type == 'unet':
        return strip_unet_prefix(state_dict)
    return state_dict

def handle_tensors(writer, state_dict, model_arch):
    # --- MEMORY FIX ---
    # The logic below is restored from the original script. It is much more
    # memory-efficient as it avoids unnecessarily upcasting float16 tensors.
    tqdm.write("Writing tensors...")
    for key, data in tqdm(state_dict.items(), desc="Processing Tensors"):
        old_dtype = data.dtype

        # Skip ignored keys
        if any(x in key for x in model_arch.keys_ignore):
            continue

        # Convert tensor to numpy array intelligently
        if data.dtype == torch.bfloat16:
            # upcast bfloat16 to float32
            data = data.to(torch.float32).numpy()
        else:
            # directly convert float16 and float32 without changing dtype
            data = data.numpy()

        n_dims = len(data.shape)
        data_shape = data.shape
        
        # Determine the target quantization type
        if old_dtype == torch.bfloat16:
            data_qtype = gguf.GGMLQuantizationType.BF16
        else:
            data_qtype = gguf.GGMLQuantizationType.F16 # Default to F16 for float16/32

        n_params = data.size

        if old_dtype in (torch.float32, torch.bfloat16):
            if n_dims == 1 or n_params <= QUANTIZATION_THRESHOLD or any(x in key for x in model_arch.keys_hiprec):
                # Keep 1D, small, or high-precision tensors in F32
                data_qtype = gguf.GGMLQuantizationType.F32

        if len(data.shape) > MAX_TENSOR_DIMS:
            model_arch.handle_nd_tensor(key, data)
            continue

        # Add the tensor to the GGUF writer
        writer.add_tensor(key, data, raw_dtype=data_qtype)


def convert_file(args):
    state_dict = load_state_dict(args.src, args.model_type)
    model_arch = detect_unet_arch(state_dict) if args.model_type == 'unet' else direct_arch_map[args.model_type]()
    logging.info(f"* Architecture selected: {model_arch.arch}")

    dst_path = args.dst or f"{os.path.splitext(args.src)[0]}-F16.gguf"
    logging.info(f"Output path: {dst_path}")
    if os.path.exists(dst_path):
        logging.warning("Output file exists and will be overwritten.")

    writer = gguf.GGUFWriter(path=None, arch=model_arch.arch)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    
    handle_tensors(writer, state_dict, model_arch)

    writer.write_header_to_file(path=dst_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    logging.info(f"Conversion complete. File saved to {dst_path}")

if __name__ == "__main__":
    args = parse_args()
    convert_file(args)