import os
import time
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from modules import config, model_registry
from modules.model_download.runtime import download_file

vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
]


def _ensure_assets(label, assets, progress=False):
    assets = list(assets or [])
    if not assets:
        return

    start = time.perf_counter()
    print(f'[Startup] Ensuring {label} assets ({len(assets)}) ...')
    for asset in assets:
        model_registry.ensure_asset(asset['id'], progress=progress)
    print(f'[Startup] Ensured {label} assets in {time.perf_counter() - start:.2f}s')


def _ensure_internal_assets(category, progress=False):
    _ensure_assets(
        f'internal {category}',
        sorted(model_registry.list_assets(category=category, internal_only=True), key=lambda item: item['id']),
        progress=progress,
    )


def _ensure_guidance_assets(progress=False):
    for channel in ('Structural', 'Contextual'):
        _ensure_assets(
            f'{channel.lower()} guidance',
            sorted(model_registry.list_assets(channel=channel), key=lambda item: item['id']),
            progress=progress,
        )


def _ensure_startup_support_assets(progress=False):
    _ensure_internal_assets('upscale', progress=progress)
    _ensure_internal_assets('removal', progress=progress)
    _ensure_guidance_assets(progress=progress)
    model_registry.ensure_asset('inpaint.flux_fill.empty_conditioning', progress=progress)
    model_registry.ensure_asset('inpaint.flux_fill.background_conditioning', progress=progress)


def _resolve_startup_download_url(url: str) -> str:
    parsed = urlparse(str(url or '').strip())
    host = (parsed.netloc or '').lower()
    if not host.endswith('civitai.com'):
        return url

    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if query.get('token'):
        return url

    token = os.getenv('CIVITAI_TOKEN', '').strip()
    if not token:
        return url

    query['token'] = token
    return urlunparse(parsed._replace(query=urlencode(query)))


def _download_checkpoint_targets(checkpoint_downloads):
    for file_name, url in checkpoint_downloads.items():
        download_file(
            url=_resolve_startup_download_url(url),
            model_dir=config.get_preferred_asset_root_path('checkpoints', file_name=file_name),
            file_name=file_name,
        )


def download_models(default_model, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads, upscale_downloads):
    from modules.util import get_file_from_folder_list

    overall_start = time.perf_counter()

    for file_name, url in vae_approx_filenames:
        download_file(url=url, model_dir=config.path_vae_approx, file_name=file_name)

    if checkpoint_downloads:
        checkpoint_start = time.perf_counter()
        _download_checkpoint_targets(checkpoint_downloads)
        print(f'[Startup] Checkpoint downloads completed in {time.perf_counter() - checkpoint_start:.2f}s')

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
        download_file(url=url, model_dir=config.get_preferred_asset_root_path('vae', file_name=file_name), file_name=file_name)

    # Front-load all support-model assets so the UI does not need to trigger them later.
    _ensure_startup_support_assets(progress=False)

    # Keep preset/config-defined entries as additive custom downloads rather than the source of truth.
    for file_name, url in upscale_downloads.items():
        download_file(url=url, model_dir=config.get_preferred_asset_root_path('upscale_models', file_name=file_name), file_name=file_name)

    print(f'[Startup] download_models work completed in {time.perf_counter() - overall_start:.2f}s')
    return default_model, checkpoint_downloads
