"""Compatibility shim for the renamed GPU compiler surface."""

from backend.gpu_compiler import GpuArtifactCompiler, ResidentLoRACompiler

__all__ = ["GpuArtifactCompiler", "ResidentLoRACompiler"]
