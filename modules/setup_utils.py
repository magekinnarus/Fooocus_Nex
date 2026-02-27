import os
from modules import config
from modules.model_loader import load_file_from_url

vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
]

def download_models(default_model, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads):
    from modules.util import get_file_from_folder_list
    from args_manager import args
    import os

    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=config.path_vae_approx, file_name=file_name)

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
        load_file_from_url(url=url, model_dir=config.path_embeddings, file_name=file_name)
    for file_name, url in lora_downloads.items():
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, config.paths_loras))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in vae_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_vae, file_name=file_name)

    return default_model, checkpoint_downloads
