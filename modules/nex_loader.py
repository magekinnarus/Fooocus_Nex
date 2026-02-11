import modules.config
from ldm_patched.modules.sd import load_checkpoint_guess_config
from modules.config import path_embeddings

def load_checkpoint(ckpt_filename, vae_filename=None):
    """
    Loads a checkpoint using the standard pipeline.
    Future expansion: will support GGUF loading here.
    """
    # Delegate to existing ComfyUI/Fooocus loader
    unet, clip, vae, vae_filename, clip_vision = load_checkpoint_guess_config(
        ckpt_filename, 
        embedding_directory=path_embeddings,
        vae_filename_param=vae_filename
    )
    return unet, clip, vae, vae_filename, clip_vision
