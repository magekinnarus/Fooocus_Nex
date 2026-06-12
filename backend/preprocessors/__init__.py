STRUCTURAL_PREPROCESSOR_ASSETS = {
    "Depth": "structural.depth.preprocessor",
    "MLSD": "structural.mlsd.preprocessor",
}

STRUCTURAL_CONTROLNET_ASSETS = {
    "PyraCanny": "structural.canny.controlnet",
    "CPDS": "structural.cpds.controlnet",
    "Depth": "structural.depth.controlnet",
    "MLSD": "structural.mlsd.controlnet",
}

from .runtime import apply_residency_policy, offload_cached_preprocessors, run_structural_preprocessor
