STRUCTURAL_PREPROCESSOR_ASSETS = {
    "Depth": "structural.depth.preprocessor",
    "MistoLine": "structural.mistoline.preprocessor",
    "MLSD": "structural.mlsd.preprocessor",
}

STRUCTURAL_CONTROLNET_ASSETS = {
    "PyraCanny": "structural.canny.controlnet",
    "CPDS": "structural.cpds.controlnet",
    "Depth": "structural.depth.controlnet",
    "MistoLine": "structural.mistoline.controlnet",
    "MLSD": "structural.mlsd.controlnet",
}

from .runtime import offload_cached_preprocessors, run_structural_preprocessor
