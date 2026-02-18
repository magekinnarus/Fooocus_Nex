
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
    "transformer_depth": [1, 1, 1, 1, 1, 1, 1, 1], # 1 per block for SD1.5
    "channel_mult": [1, 2, 4, 4],
    "transformer_depth_middle": 1,
    "use_linear_in_transformer": False,
    "context_dim": 768,
    "num_heads": 8,
    "transformer_depth_output": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Expanded from convert_config logic?
    "use_temporal_attention": False,
    "use_temporal_resblock": False
}

# Calculated from predict_unet_config in model_detection.py for SD15
# SD15 = {'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0], ...}
# convert_config expands this.
# Let's trust that passing the Base config is safer if we use model_detection.convert_config.
# But for a static def, we should probably use the expanded version or just use the base and call convert.
# backend/defs/sdxl.py seems to use the "raw" config which might be post-conversion or pre-conversion.
# SDXL in sdxl.py: "num_res_blocks": [2, 2, 2] -> matches detection.
# SDXL in sdxl.py: "transformer_depth": [0, 0, 2, 2, 10, 10] -> matches detection.
# So I should use the detection config for SD15 exactly.

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
