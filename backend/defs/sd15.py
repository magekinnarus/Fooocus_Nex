
PREFIXES = {
    "unet": ["model.diffusion_model."],
    "clip": ["cond_stage_model.transformer.", "cond_stage_model.model.", "cond_stage_model."], # Standard, Legacy, and Global
    "vae": ["first_stage_model."],
}

UNET_CONFIG = {
    "use_checkpoint": False,
    "image_size": 32,
    "out_channels": 4,
    "use_spatial_transformer": True,
    "legacy": False,
    "adm_in_channels": None,
    "in_channels": 4,
    "model_channels": 320,
    "num_res_blocks": [2, 2, 2, 2],
    "transformer_depth": [1, 1, 1, 1, 1, 1, 0, 0],
    "channel_mult": [1, 2, 4, 4],
    "transformer_depth_middle": 1,
    "use_linear_in_transformer": False,
    "context_dim": 768,
    "num_heads": 8,
    "transformer_depth_output": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    "use_temporal_attention": False,
    "use_temporal_resblock": False
}
