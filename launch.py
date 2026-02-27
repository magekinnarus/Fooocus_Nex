import os
import ssl
import sys

# Aliasing __main__ as launch to prevent double loading
if __name__ == "__main__":
    if os.environ.get('FOOOCUS_LAUNCHED') == '1':
        sys.exit(0)
    os.environ['FOOOCUS_LAUNCHED'] = '1'
    sys.modules['launch'] = sys.modules['__main__']

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

ssl._create_default_https_context = ssl._create_unverified_context

import platform
import fooocus_version

from build_launcher import build_launcher
from modules.launch_util import delete_folder_content # Keep this for cleanup
from modules.model_loader import load_file_from_url

def prepare_environment():
    print(f"Python {sys.version}")
    print(f"Fooocus version: {fooocus_version.version}")
    print("Dependency management is handled manually in Colab cells.")
    return

def ini_args():
    from args_manager import args
    return args


if __name__ == "__main__":
    print('[System ARGV] ' + str(sys.argv))

    prepare_environment()
    build_launcher()
    args = ini_args()

    if args.gpu_device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
        print("Set device to:", args.gpu_device_id)

    if args.hf_mirror is not None:
        os.environ['HF_MIRROR'] = str(args.hf_mirror)
        print("Set hf_mirror to:", args.hf_mirror)

    # Load .env variables (requires python-dotenv to be installed manually)
    try:
        from dotenv import load_dotenv
        if os.path.exists(os.path.join(root, '.env')):
            load_dotenv(os.path.join(root, '.env'))
            print("Loaded environment variables from .env file.")
    except ImportError:
        print("python-dotenv not installed. .env file will not be loaded automatically.")
    except Exception as e:
        print(f"Error loading .env file: {e}")

    from modules import config
    from modules.hash_cache import init_cache
    from modules.setup_utils import download_models

    os.environ["U2NET_HOME"] = config.path_inpaint
    os.environ['GRADIO_TEMP_DIR'] = config.temp_path

    if config.temp_path_cleanup_on_launch:
        print(f'[Cleanup] Attempting to delete content of temp dir {config.temp_path}')
        result = delete_folder_content(config.temp_path, '[Cleanup] ')
        if result:
            print("[Cleanup] Cleanup successful")
        else:
            print(f"[Cleanup] Failed to delete content of temp dir.")

    config.default_base_model_name, config.checkpoint_downloads = download_models(
        config.default_base_model_name, config.checkpoint_downloads,
        config.embeddings_downloads, config.lora_downloads, config.vae_downloads)

    config.update_files()
    init_cache(config.model_filenames, config.paths_checkpoints, config.lora_filenames, config.paths_loras)

    from webui import *
