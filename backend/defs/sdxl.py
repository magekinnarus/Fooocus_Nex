PREFIXES = {
    "unet": ["model.diffusion_model."],
    "clip_l": [
        "conditioner.embedders.0.transformer.text_model.",
        "cond_stage_model.clip_l.transformer.text_model.",
        "cond_stage_model.clip_l.text_model.",
        "clip_l.transformer.text_model.",
    ],
    "clip_g": [
        "conditioner.embedders.1.model.",
        "cond_stage_model.clip_g.model.",
        "conditioner.embedders.1.transformer.text_model.",
        "cond_stage_model.clip_g.transformer.text_model.",
        "clip_g.model.",
    ],
    "vae": ["first_stage_model.", "vae."],
}

UNET_CONFIG = {
    "use_checkpoint": False,
    "image_size": 32,
    "out_channels": 4,
    "use_spatial_transformer": True,
    "legacy": False,
    "num_classes": "sequential",
    "adm_in_channels": 2816,
    "in_channels": 4,
    "model_channels": 320,
    "num_res_blocks": [2, 2, 2],
    "transformer_depth": [0, 0, 2, 2, 10, 10],
    "channel_mult": [1, 2, 4],
    "transformer_depth_middle": 10,
    "use_linear_in_transformer": True,
    "context_dim": 2048,
    "num_head_channels": 64,
    "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
    "use_temporal_attention": False,
    "use_temporal_resblock": False
}
