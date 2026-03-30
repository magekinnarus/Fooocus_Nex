import os
from modules import config, model_registry
from modules.model_download.runtime import download_file

vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
]


def _ensure_internal_assets(category, progress=False):
    for asset in sorted(model_registry.list_assets(category=category, internal_only=True), key=lambda item: item['id']):
        model_registry.ensure_asset(asset['id'], progress=progress)


def download_models(default_model, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads, upscale_downloads):
    from modules.util import get_file_from_folder_list

    for file_name, url in vae_approx_filenames:
        download_file(url=url, model_dir=config.path_vae_approx, file_name=file_name)

    # Check if any model exists in checkpoints
    model_found = False
    for folder in config.paths_checkpoints:
        if os.path.isdir(folder):
            if any(f.endswith(('.safetensors', '.ckpt')) for f in os.listdir(folder)):
                model_found = True
                break

    if not model_found:
        print('No checkpoint models found in your checkpoints directories.')
        print('Please add at least one model to your checkpoints folder to start generating.')

    # Embeddings, Loras, VAE downloads (optional, kept if explicitly in config)
    for file_name, url in embeddings_downloads.items():
        download_file(url=url, model_dir=config.path_embeddings, file_name=file_name)
    lora_download_roots = {
        'sdxl_lcm_lora.safetensors': config.path_loras_lcm,
        'sdxl_lightning_4step_lora.safetensors': config.path_loras_lightning,
    }
    lora_lookup_paths = [
        config.paths_loras[0],
        config.path_loras_lcm,
        config.path_loras_lightning,
        config.path_faceid_loras,
    ]

    for file_name, url in lora_downloads.items():
        preferred_root = lora_download_roots.get(file_name, config.paths_loras[0])
        existing_path = get_file_from_folder_list(file_name, [preferred_root] + [p for p in lora_lookup_paths if p != preferred_root])
        if os.path.exists(existing_path):
            continue
        download_file(url=url, model_dir=preferred_root, file_name=file_name)
    for file_name, url in vae_downloads.items():
        download_file(url=url, model_dir=config.path_vae, file_name=file_name)

    # Internal upscalers now come from the centralized manifest system.
    _ensure_internal_assets('upscale', progress=False)

    # Keep preset/config-defined entries as additive custom downloads rather than the source of truth.
    for file_name, url in upscale_downloads.items():
        download_file(url=url, model_dir=config.path_upscale_models[0], file_name=file_name)

    return default_model, checkpoint_downloads
