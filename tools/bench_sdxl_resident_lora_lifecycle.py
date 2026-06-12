import os
import sys
import gc
import time
import tempfile
import argparse
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Optional
import torch
import safetensors.torch
try:
    import psutil
except Exception:  # pragma: no cover - optional runtime dependency for diagnostics only
    psutil = None

# Ensure REPO_ROOT is in path
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pre-mock args_parser and args_manager to avoid conflict
fake_args = SimpleNamespace(
    colab=False,
    preset="",
    output_path="",
    temp_path="",
    skip_model_load=True,
    disable_metadata=True,
)
sys.modules["args_manager"] = SimpleNamespace(
    args=fake_args,
    args_parser=SimpleNamespace(args=fake_args, parser=SimpleNamespace()),
)

from backend import lora as backend_lora, patching
from backend.cpu_compiler import SafeOpenHeaderOnly
from backend.gpu_compiler import GpuArtifactCompiler

class MockProjIn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Use slightly larger weights to make measurements visible
        self.weight = torch.nn.Parameter(torch.randn(1024, 1024, dtype=torch.float16))

class MockConvIn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(256, 256, 3, 3, dtype=torch.float16))

class MockUNetInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = MockProjIn()
        self.conv_in = MockConvIn()

class MockUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = MockUNetInner()
        self.model_config = SimpleNamespace(unet_config={
            "num_res_blocks": [1],
            "channel_mult": [1],
            "transformer_depth": [0],
            "transformer_depth_middle": 0,
            "transformer_depth_output": [0, 0],
        })
    def model_size(self) -> int:
        # 1024*1024*2 + 256*256*9*2 = 2,097,152 + 1,179,648 = 3,276,800 bytes
        return 3276800


def _current_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
    except Exception:
        return None


def _memory_observation(device: torch.device) -> Dict[str, float]:
    observation: Dict[str, float] = {}
    rss_mb = _current_rss_mb()
    if rss_mb is not None:
        observation["rss_mb"] = rss_mb
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        observation["cuda_allocated_mb"] = round(torch.cuda.memory_allocated(device) / (1024 * 1024), 2)
        observation["cuda_reserved_mb"] = round(torch.cuda.memory_reserved(device) / (1024 * 1024), 2)
        observation["cuda_peak_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
    return observation

def create_fake_lora_file(rank: int = 4) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors")
    mat1_lin = torch.randn(1024, rank, dtype=torch.float16)
    mat2_lin = torch.randn(rank, 1024, dtype=torch.float16)
    alpha_lin = torch.tensor(float(rank), dtype=torch.float32)

    mat1_conv = torch.randn(256, rank, dtype=torch.float16)
    mat2_conv = torch.randn(rank, 256, 3, 3, dtype=torch.float16)
    alpha_conv = torch.tensor(float(rank), dtype=torch.float32)

    tensors = {
        "lora_unet_proj_in.lora_up.weight": mat1_lin,
        "lora_unet_proj_in.lora_down.weight": mat2_lin,
        "lora_unet_proj_in.alpha": alpha_lin,
        "lora_unet_conv_in.lora_up.weight": mat1_conv,
        "lora_unet_conv_in.lora_down.weight": mat2_conv,
        "lora_unet_conv_in.alpha": alpha_conv,
    }
    safetensors.torch.save_file(tensors, tmp.name)
    tmp.close()
    return tmp.name

def run_lifecycle_benchmark(placement: str, device: torch.device):
    print(f"\n==========================================")
    print(f"Running Residency Matrix Lifecycle Benchmark")
    print(f"Clean-source placement: {placement.upper()}")
    print(f"Device: {device}")
    print(f"==========================================")

    # 1. Cold Load
    print("[Step 1] Cold Load: Initializing Working UNet on device...")
    working_model = MockUNet()
    working_model.to(device)
    patcher = patching.NexModelPatcher(
        working_model,
        load_device=device,
        offload_device=device,
        size=working_model.model_size(),
    )
    
    clean_source: Optional[Dict[str, torch.Tensor]] = None
    active_lora_sig = ()

    # Generate two mock LoRA files
    lora1_path = create_fake_lora_file(rank=4)
    lora2_path = create_fake_lora_file(rank=8)

    try:
        # Helper function to restore from clean_source
        def restore_from_clean():
            nonlocal clean_source
            if clean_source is not None:
                start_r = time.perf_counter()
                with torch.no_grad():
                    for name, param in patcher.model.named_parameters():
                        param.copy_(clean_source[name])
                    for name, buf in patcher.model.named_buffers():
                        buf.copy_(clean_source[name])
                end_r = time.perf_counter()
                print(f"  -> Restored clean weights from in-memory snapshot in {((end_r - start_r) * 1000):.2f} ms")

        # 2. First LoRA Apply
        print("\n[Step 2] Apply first LoRA (LoRA 1, strength=0.8)...")
        # Ensure clean source snapshot is lazy-created
        if clean_source is None:
            snap_start = time.perf_counter()
            clean_source = {}
            snapshot_device = torch.device("cpu") if placement == "cpu" else device
            for name, param in patcher.model.named_parameters():
                clean_source[name] = param.detach().to(device=snapshot_device, copy=True)
            for name, buf in patcher.model.named_buffers():
                clean_source[name] = buf.detach().to(device=snapshot_device, copy=True)
            snap_end = time.perf_counter()
            print(f"  -> Lazy clean source snapshot created on {snapshot_device} in {((snap_end - snap_start) * 1000):.2f} ms")
            print(f"  -> Snapshot memory observation: {_memory_observation(device)}")

        # Load and add patches
        key_map = backend_lora.model_lora_keys_unet(patcher.model)
        header = SafeOpenHeaderOnly(lora1_path)
        patch_dict = backend_lora.load_lora(header, key_map, log_missing=False)
        patcher.add_patches(patch_dict, 0.8)

        # Compile
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        comp_start = time.perf_counter()
        metrics = GpuArtifactCompiler.compile_patcher(
            patcher,
            clean_source=clean_source,
            target_device=device,
            intermediate_dtype=torch.float16,
        )
        comp_end = time.perf_counter()
        active_lora_sig = (lora1_path, 0.8)
        print(f"  -> Compiled successfully in {((comp_end - comp_start) * 1000):.2f} ms")
        print(f"  -> Patcher metrics: {metrics}")
        print(f"  -> Compile memory observation: {_memory_observation(device)}")

        # 3. Warm rerun with same stack
        print("\n[Step 3] Warm rerun with unchanged LoRA stack...")
        desired_sig = (lora1_path, 0.8)
        if desired_sig == active_lora_sig:
            cache_observation = {"cache_hit": True, **_memory_observation(device)}
            print("  -> Unchanged stack. Skipping compilation (zero CPU/GPU work).")
            print(f"  -> Cache observation: {cache_observation}")
        else:
            print("  -> Stack mismatch (unexpected). Compiling...")

        # 4. Stack change to different LoRA
        print("\n[Step 4] Stack change (LoRA 1 -> LoRA 2, strength=1.0)...")
        # Restore clean weights first in-memory
        restore_from_clean()

        # Load and add patches for LoRA 2
        header2 = SafeOpenHeaderOnly(lora2_path)
        patch_dict2 = backend_lora.load_lora(header2, key_map, log_missing=False)
        patcher.add_patches(patch_dict2, 1.0)

        # Compile
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        comp_start = time.perf_counter()
        metrics = GpuArtifactCompiler.compile_patcher(
            patcher,
            clean_source=clean_source,
            target_device=device,
            intermediate_dtype=torch.float16,
        )
        comp_end = time.perf_counter()
        active_lora_sig = (lora2_path, 1.0)
        print(f"  -> Compiled new stack in {((comp_end - comp_start) * 1000):.2f} ms")
        print(f"  -> Patcher metrics: {metrics}")
        print(f"  -> Stack-change memory observation: {_memory_observation(device)}")

        # 5. Return to no-LoRA state
        print("\n[Step 5] Return to empty stack (removing all LoRAs)...")
        # Restore clean weights
        restore_from_clean()
        # Discard lazy snapshot
        clean_source = None
        active_lora_sig = ()
        print("  -> Discarded clean source snapshot from memory.")
        print(f"  -> Final memory observation: {_memory_observation(device)}")

    finally:
        # Clean up files
        if os.path.exists(lora1_path):
            os.remove(lora1_path)
        if os.path.exists(lora2_path):
            os.remove(lora2_path)
        print("\nLifecycle benchmark completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Resident LoRA lifecycle benchmark harness.")
    parser.add_argument(
        "--placement",
        default="both",
        choices=("cpu", "gpu", "both"),
        help="Clean source placement profile: 'cpu' (8 GB), 'gpu' (Colab), or 'both'.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    placements = ["cpu", "gpu"] if args.placement == "both" else [args.placement]
    for placement in placements:
        run_lifecycle_benchmark(placement, device)

if __name__ == "__main__":
    main()
