import json
from abc import ABC, abstractmethod
from pathlib import Path
import os

from PIL import Image

import fooocus_version
import modules.config
from modules.flags import MetadataScheme
from modules.hash_cache import sha256_from_cache
from modules.util import get_file_from_folder_list, is_json


class MetadataParser(ABC):
    def __init__(self):
        self.raw_prompt: str = ''
        self.full_prompt: str = ''
        self.raw_negative_prompt: str = ''
        self.full_negative_prompt: str = ''
        self.steps: int = 30
        self.base_model_name: str = ''
        self.base_model_hash: str = ''
        self.loras: list = []
        self.vae_name: str = ''
        self.clip_model_name: str = ''

    @abstractmethod
    def get_scheme(self) -> MetadataScheme:
        raise NotImplementedError

    @abstractmethod
    def to_json(self, metadata: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def to_string(self, metadata: list) -> str:
        raise NotImplementedError

    def set_data(self, raw_prompt, full_prompt, raw_negative_prompt, full_negative_prompt, steps, base_model_name,
                 loras, vae_name, clip_model_name):
        self.raw_prompt = raw_prompt
        self.full_prompt = full_prompt
        self.raw_negative_prompt = raw_negative_prompt
        self.full_negative_prompt = full_negative_prompt
        self.steps = steps
        self.base_model_name = Path(base_model_name).stem

        base_model_path = get_file_from_folder_list(base_model_name, modules.config.paths_checkpoints)
        self.base_model_hash = sha256_from_cache(base_model_path)

        self.loras = []
        for (lora_name, lora_weight) in loras:
            if lora_name != 'None':
                lora_path = get_file_from_folder_list(lora_name, modules.config.paths_lora_lookup)
                lora_hash = sha256_from_cache(lora_path)
                self.loras.append((Path(lora_name).stem, lora_weight, lora_hash))
        self.vae_name = Path(vae_name).stem
        self.clip_model_name = Path(clip_model_name).stem if clip_model_name != 'None' else 'None'


class FooocusMetadataParser(MetadataParser):
    def __init__(self, scheme: MetadataScheme = MetadataScheme.FOOOCUS_NEX):
        super().__init__()
        self.scheme = scheme

    def get_scheme(self) -> MetadataScheme:
        return self.scheme

    def to_json(self, metadata: dict) -> dict:
        for key, value in metadata.items():
            if value in ['', 'None']:
                continue
            if key in ['base_model']:
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.model_filenames)
            elif key.startswith('lora_combined_'):
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.lora_filenames)
            elif key == 'vae':
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.vae_filenames)
            elif key == 'clip_model':
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.clip_filenames)
            else:
                continue

        return metadata

    def to_string(self, metadata: list) -> str:
        for li, (label, key, value) in enumerate(metadata):
            # remove model folder paths from metadata
            if key.startswith('lora_combined_'):
                name, weight = value.split(' : ')
                name = Path(name).stem
                value = f'{name} : {weight}'
                metadata[li] = (label, key, value)

        res: dict = {k: v for _, k, v in metadata}

        res['full_prompt'] = self.full_prompt
        res['full_negative_prompt'] = self.full_negative_prompt
        res['steps'] = self.steps
        res['base_model'] = self.base_model_name
        res['base_model_hash'] = self.base_model_hash

        res['vae'] = self.vae_name
        res['clip_model'] = self.clip_model_name
        res['loras'] = self.loras

        if modules.config.metadata_created_by != '':
            res['created_by'] = modules.config.metadata_created_by

        return json.dumps(dict(sorted(res.items())))

    @staticmethod
    def replace_value_with_filename(key, value, filenames):
        for filename in filenames:
            path = Path(filename)
            if key.startswith('lora_combined_'):
                name, weight = value.split(' : ')
                if name == path.stem:
                    return f'{filename} : {weight}'
            elif value == path.stem:
                return filename

        return None


def get_metadata_parser(metadata_scheme: MetadataScheme) -> MetadataParser:
    if metadata_scheme == MetadataScheme.FOOOCUS:
        return FooocusMetadataParser(MetadataScheme.FOOOCUS)
    if metadata_scheme == MetadataScheme.FOOOCUS_NEX:
        return FooocusMetadataParser(MetadataScheme.FOOOCUS_NEX)
    raise NotImplementedError


def read_info_from_image(file_or_path) -> tuple[str | None, MetadataScheme | None]:
    if not file_or_path:
        return None, None

    try:
        if isinstance(file_or_path, (str, os.PathLike)):
            file_path = os.fspath(file_or_path)
            if not file_path or not os.path.exists(file_path):
                return None, None
            with Image.open(file_path) as img:
                items = (img.info or {}).copy()
        else:
            items = (file_or_path.info or {}).copy()
    except Exception:
        return None, None

    parameters = items.pop('parameters', None)
    metadata_scheme = items.pop('fooocus_scheme', None)
    if isinstance(metadata_scheme, str):
        try:
            metadata_scheme = MetadataScheme(metadata_scheme)
        except ValueError:
            metadata_scheme = None

    if parameters is not None and is_json(parameters):
        parameters = json.loads(parameters)
        if metadata_scheme is None:
            metadata_scheme = MetadataScheme.FOOOCUS

    return parameters, metadata_scheme

def get_exif(metadata: str | None, metadata_scheme: str):
    exif = Image.Exif()
    # tags see see https://github.com/python-pillow/Pillow/blob/9.2.x/src/PIL/ExifTags.py
    # 0x9286 = UserComment
    exif[0x9286] = metadata
    # 0x0131 = Software
    exif[0x0131] = f'{fooocus_version.app_name} {fooocus_version.version}'
    # 0x927C = MakerNote
    exif[0x927C] = metadata_scheme
    return exif




