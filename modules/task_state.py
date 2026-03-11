from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np


@dataclass
class TaskState:
    # --- Generation Parameters ---
    generate_image_grid: bool = False
    prompt: str = ""
    negative_prompt: str = ""
    style_selections: List[str] = field(default_factory=list)
    steps: int = 30
    original_steps: int = 30
    aspect_ratios_selection: str = "1024x1024"
    image_number: int = 1
    output_format: str = "png"
    seed: int = -1
    sharpness: float = 2.0
    cfg_scale: float = 4.0
    base_model_name: str = ""
    vae_name: str = ""
    clip_model_name: str = ""
    loras: List[Any] = field(default_factory=list)
    input_image_checkbox: bool = False
    current_tab: str = "uov"
    uov_method: str = "Disabled"
    uov_input_image: Optional[np.ndarray] = None
    outpaint_selections: List[str] = field(default_factory=list)
    outpaint_input_image: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None
    outpaint_bb_image: Optional[np.ndarray] = None
    outpaint_bb_mask_data: str = ""
    outpaint_mask_image: Optional[np.ndarray] = None
    inpaint_input_image: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None
    inpaint_context_mask_image: Optional[np.ndarray] = None
    inpaint_additional_prompt: str = ""
    inpaint_mask_image: Optional[np.ndarray] = None
    inpaint_bb_image: Optional[np.ndarray] = None
    disable_preview: bool = False
    disable_intermediate_results: bool = False
    disable_seed_increment: bool = False
    adm_scaler_positive: float = 1.5
    adm_scaler_negative: float = 0.8
    adm_scaler_end: float = 0.3
    adaptive_cfg: float = 7.0
    clip_skip: int = 1
    sampler_name: str = "dpmpp_2m_sde_gpu"
    scheduler_name: str = "karras"
    overwrite_step: int = -1
    overwrite_width: int = -1
    overwrite_height: int = -1
    overwrite_vary_strength: float = -1.0
    overwrite_upscale_strength: float = -1.0
    mixing_image_prompt_and_vary_upscale: bool = False
    mixing_image_prompt_and_inpaint: bool = False
    debugging_cn_preprocessor: bool = False
    skipping_cn_preprocessor: bool = False
    canny_low_threshold: int = 64
    canny_high_threshold: int = 128
    controlnet_softness: float = 0.25
    debugging_inpaint_preprocessor: bool = False
    inpaint_disable_initial_latent: bool = False
    inpaint_engine: str = "None"
    inpaint_strength: float = 1.0
    inpaint_respective_field: float = 0.618
    inpaint_erode_or_dilate: int = 0
    inpaint_step2_checkbox: bool = False
    outpaint_step2_checkbox: bool = False
    outpaint_engine: str = "None"
    outpaint_strength: float = 1.0
    inpaint_outpaint_expansion_size: int = 384
    inpaint_pixelate_primer: bool = False
    context_mask: Optional[np.ndarray] = None
    outpaint_direction: Optional[str] = None
    save_metadata_to_images: bool = True
    metadata_scheme: Any = None # modules.flags.MetadataScheme
    cn_tasks: Dict[str, List[Any]] = field(default_factory=dict)
    
    # --- Removed Feature placeholders (Tech Debt to be purged) ---
    debugging_dino: bool = False
    dino_erode_or_dilate: int = 0
    debugging_enhance_masks_checkbox: bool = False
    enhance_input_image: Optional[np.ndarray] = None

    # --- Runtime State ---
    performance_loras: List[Any] = field(default_factory=list)
    yields: List[Any] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    last_stop: Union[bool, str] = False
    processing: bool = False
    goals: List[str] = field(default_factory=list)
    initial_latent: Optional[Dict[str, Any]] = None
    denoising_strength: float = 1.0
    tiled: bool = False
    positive_cond: Optional[Any] = None
    negative_cond: Optional[Any] = None
    width: int = 1024
    height: int = 1024
    use_expansion: bool = False
    inpaint_context: object = None
    use_style: bool = True

    def __post_init__(self):
        if not self.cn_tasks:
            from modules.flags import ip_list
            self.cn_tasks = {x: [] for x in ip_list}
