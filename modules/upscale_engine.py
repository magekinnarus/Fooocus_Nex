import torch
import numpy as np
import math
from typing import Callable, List, Tuple
from dataclasses import dataclass
import modules.core as core
import gc
import time

@dataclass
class Segment:
    start: int
    end: int
    start_pad: int
    end_pad: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def padded_length(self) -> int:
        return (self.end + self.end_pad) - (self.start - self.start_pad)

def split_into_segments(length: int, tile_size: int, overlap: int) -> List[Segment]:
    if length <= tile_size:
        return [Segment(0, length, 0, 0)]
    
    result = []
    # First segment
    result.append(Segment(0, tile_size - overlap, 0, overlap))
    
    while result[-1].end < length:
        start = result[-1].end
        end = start + tile_size - overlap * 2
        
        if end >= length:
            # Last segment: pad backward to keep constant tile size
            end = length
            start_pad = tile_size - (end - start)
            result.append(Segment(start, end, start_pad, 0))
        else:
            result.append(Segment(start, end, overlap, overlap))
            
    return result

from modules.blending import sin_blend_1d

class RowBlender:
    def __init__(self, width: int, height: int, channels: int, device="cpu", dtype=torch.float32):
        self.output = torch.zeros((channels, height, width), device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def add_tile(self, tile: torch.Tensor, x: int, overlap_l: int):
        # tile is [C, H, W]
        c, h, w = tile.shape
        tile = tile.to(self.device, dtype=self.dtype)
        
        if x == 0:
            self.output[:, :, :w] = tile
        else:
            overlap_start = x - overlap_l
            if overlap_l > 0:
                mask = sin_blend_1d(overlap_l, self.device, self.dtype).view(1, 1, -1)
                overlap_region = self.output[:, :, overlap_start:x]
                self.output[:, :, overlap_start:x] = overlap_region * (1.0 - mask) + tile[:, :, :overlap_l] * mask
            
            unique_w = w - overlap_l
            self.output[:, :, x : x + unique_w] = tile[:, :, overlap_l:]

    def get_result(self) -> torch.Tensor:
        return self.output

class NexUpscaleEngine:
    def __init__(self):
        pass

    @torch.inference_mode()
    def process(self, img: np.ndarray, upscale_fn: Callable[[torch.Tensor], torch.Tensor], 
                scale: int, device: torch.device, is_bgr: bool = True, dtype: torch.dtype = None):
        
        m_dtype = dtype or torch.float32
        
        # Ensure correct CUDA device context
        if device.type == 'cuda' and device.index is not None:
            if torch.cuda.current_device() != device.index:
                torch.cuda.set_device(device)

        # Input to Tensor
        in_img = core.numpy_to_pytorch(img).movedim(-1, -3).to(device, dtype=m_dtype)
        if is_bgr: in_img = in_img[:, [2, 1, 0], :, :]
            
        h, w = in_img.shape[2:]
        oh, ow = h * scale, w * scale
        
        # Optimized Hardware-Aware Tiling
        tile_size = 512 
        overlap = 32
        
        if "cuda" in device.type:
            try:
                # Enable CuDNN benchmark for consistent tiled shapes
                torch.backends.cudnn.benchmark = True
                
                free, _ = torch.cuda.mem_get_info(device)
                # target 50% to stay well within physical VRAM (prevent paging)
                usable_vram = free * 0.45 
                
                # DYNAMIC MULTIPLIER (Model-Size Aware)
                # Heavy models (SwinIR/HAT) use significantly more intermediate VRAM than 
                # light models (ESRGAN-Lite). We inspect the closure to find the model size.
                model_params = 16 * 1024 * 1024 # Default fallback (16M parameters)
                
                # Look inside the upscale_fn closure for a torch.nn.Module
                if hasattr(upscale_fn, "__closure__") and upscale_fn.__closure__ is not None:
                    try:
                        for cell in upscale_fn.__closure__:
                            if hasattr(cell.cell_contents, "parameters"):
                                model_params = sum(p.numel() for p in cell.cell_contents.parameters())
                                break
                    except: pass
                
                # ChaiNNer-inspired safety multiplier: 
                # (model_params / constant) * element_size
                # We normalize it to a multiplier for the input pixels.
                # A 64MB model (SwinIR) results in roughly ~1200-1500 multiplier.
                safety_multiplier = max(800, int(model_params / 53248)) 
                
                pixel_size = 2 if m_dtype == torch.float16 else 4
                est_pixels = usable_vram / (pixel_size * safety_multiplier)
                
                # Maximum safe tile size based on available VRAM
                max_safe_tile = int(est_pixels**0.5)
                
                # BEST-FIT OPTIMIZER:
                # We check multiple tile candidates (multiples of 64) and pick the one 
                # that results in the SMALLEST total area to process.
                candidates = []
                # Search range: Floor 256 to Ceiling (capped at 1024 for 1050 stability)
                floor = 256
                ceil = max(256, min(1024, max_safe_tile))
                
                # Step by 64 (Industry standard for GPU hardware alignment)
                for ts in range(floor, ceil + 64, 64):
                    if ts > ceil: break
                    # Calculate tile counts if we used this tile size
                    x_count = math.ceil((w - overlap) / (ts - overlap)) if w > ts else 1
                    y_count = math.ceil((h - overlap) / (ts - overlap)) if h > ts else 1
                    # Area comparison metric: total pixels processed
                    total_area = (x_count * y_count) * (ts * ts)
                    candidates.append((total_area, ts))
                
                # Pick the most efficient candidate (least area)
                if candidates:
                    candidates.sort() # Sorts by total_area
                    tile_size = candidates[0][1]
                else:
                    tile_size = 256
                
                print(f"[Nex-Engine] VRAM: {free//1024**2}MB | Best-Fit Tile: {tile_size} | Precision: {m_dtype}")
            except: pass

        if h <= tile_size and w <= tile_size:
            start_t = time.time()
            out_img = upscale_fn(in_img)
            print(f"[Nex-Engine] Full image processed in {time.time() - start_t:.2f}s")
        else:
            x_segs = split_into_segments(w, tile_size, overlap)
            y_segs = split_into_segments(h, tile_size, overlap)
            
            # Final buffer ALWAYS on CPU to prevent OS-level VRAM paging slowdown
            final_output = torch.zeros((3, oh, ow), device="cpu", dtype=m_dtype)
            
            # Pre-emptive cleanup
            torch.cuda.empty_cache()
            gc.collect()

            for i, y_seg in enumerate(y_segs):
                row_start_t = time.time()
                row_h = y_seg.padded_length * scale
                row_blender = RowBlender(ow, row_h, 3, device=device, dtype=m_dtype)
                
                for x_seg in x_segs:
                    tx = x_seg.start - x_seg.start_pad
                    ty = y_seg.start - y_seg.start_pad
                    tile = in_img[:, :, ty:ty+y_seg.padded_length, tx:tx+x_seg.padded_length]
                    
                    # Upscale
                    upscaled = upscale_fn(tile)[0]
                    del tile
                    
                    row_blender.add_tile(upscaled, x_seg.start * scale, x_seg.start_pad * scale)
                    del upscaled
                
                # Blend Row into Final (Vertical) on CPU
                row_result = row_blender.get_result().to("cpu")
                y_start = y_seg.start * scale
                y_overlap = y_seg.start_pad * scale
                
                if y_start == 0:
                    final_output[:, :row_h, :] = row_result
                else:
                    y_blend_start = y_start - y_overlap
                    if y_overlap > 0:
                        mask = sin_blend_1d(y_overlap, "cpu", m_dtype).view(1, -1, 1)
                        overlap_region = final_output[:, y_blend_start:y_start, :]
                        final_output[:, y_blend_start:y_start, :] = overlap_region * (1.0 - mask) + row_result[:, :y_overlap, :] * mask
                    
                    unique_h = row_h - y_overlap
                    final_output[:, y_start : y_start + unique_h, :] = row_result[:, y_overlap:, :]
                
                del row_result
                # Sync-less loop (no inner empty_cache for maximum throughput)
                if (i+1) % 2 == 0: # Occasional sync to avoid complete fragmentation
                    torch.cuda.empty_cache()
                    
                print(f"[Nex-Engine] Row {i+1}/{len(y_segs)} completed in {time.time() - row_start_t:.2f}s")
            
            # Final resource release
            torch.cuda.empty_cache()
            gc.collect()
            print(f"[Nex-Engine] Tiled Upscale Completed.")
            out_img = final_output.unsqueeze(0)

        if is_bgr: out_img = out_img[:, [2, 1, 0], :, :]
            
        out_img = torch.clamp(out_img.movedim(-3, -1), 0, 1)
        return core.pytorch_to_numpy(out_img)[0]
