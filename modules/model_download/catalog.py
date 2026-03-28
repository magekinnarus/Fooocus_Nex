from __future__ import annotations

import json
from pathlib import Path

import modules.model_taxonomy
from typing import Iterable

from .spec import ModelCatalogEntry, ModelSource


class ModelCatalog:
    def __init__(self, entries: Iterable[ModelCatalogEntry] = ()): 
        self._entries_by_id: dict[str, ModelCatalogEntry] = {}
        self._entries_by_alias: dict[str, ModelCatalogEntry] = {}
        for entry in entries:
            self.add(entry)

    def add(self, entry: ModelCatalogEntry) -> None:
        if entry.id in self._entries_by_id:
            raise ValueError(f'Duplicate catalog id: {entry.id}')
        self._entries_by_id[entry.id] = entry
        if entry.alias:
            self._entries_by_alias[entry.alias] = entry

    def get(self, selector: str) -> ModelCatalogEntry | None:
        return self._entries_by_id.get(selector) or self._entries_by_alias.get(selector)

    def list(self) -> list[ModelCatalogEntry]:
        return list(self._entries_by_id.values())

    def filter(self, *, storage_tier: str | None = None, visibility: str | None = None) -> list[ModelCatalogEntry]:
        results = []
        for entry in self._entries_by_id.values():
            if storage_tier is not None and entry.storage_tier != storage_tier:
                continue
            if visibility is not None and entry.visibility != visibility:
                continue
            results.append(entry)
        return results

    @classmethod
    def from_dict(cls, payload: dict) -> 'ModelCatalog':
        entries = []
        for entry_data in _iter_entry_dicts(payload):
            entries.append(_entry_from_dict(entry_data))
        return cls(entries)

    @classmethod
    def from_file(cls, path: str | Path) -> 'ModelCatalog':
        payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        return cls.from_dict(payload)


def load_model_catalog(path: str | Path) -> ModelCatalog:
    return ModelCatalog.from_file(path)


def _iter_entry_dicts(node):
    if isinstance(node, list):
        for item in node:
            yield from _iter_entry_dicts(item)
    elif isinstance(node, dict):
        if 'id' in node and 'name' in node and 'root_key' in node and 'relative_path' in node:
            yield node
        else:
            for value in node.values():
                yield from _iter_entry_dicts(value)


def _entry_from_dict(data: dict) -> ModelCatalogEntry:
    source_provider = data.get('source_provider', data.get('source_kind', 'direct'))
    token_env = data.get('token_env')
    sources = tuple(
        ModelSource(
            url=source['url'],
            kind=source.get('kind', source_provider),
            token_env=source.get('token_env', token_env),
            headers=tuple(tuple(header) for header in source.get('headers', [])),
        )
        for source in data.get('sources', [])
        if source.get('url')
    )
    root_key = data['root_key']
    model_type = data.get('model_type', _default_model_type(root_key))
    display_name = data.get('display_name')
    if display_name is None:
        display_name = Path(data['name']).stem.replace('_', ' ')

    architecture = modules.model_taxonomy.normalize_architecture(data.get('architecture', 'unknown')) or 'unknown'
    sub_architecture = modules.model_taxonomy.normalize_sub_architecture(
        data.get('sub_architecture', 'general'),
        architecture=architecture,
    )
    compatibility_family = data.get('compatibility_family')
    if compatibility_family is None:
        compatibility_family = modules.model_taxonomy.get_compatibility_family(
            architecture=architecture,
            sub_architecture=sub_architecture,
            model_type=model_type,
        )

    return ModelCatalogEntry(
        id=data['id'],
        alias=data.get('alias'),
        name=data['name'],
        root_key=root_key,
        relative_path=data['relative_path'],
        display_name=display_name,
        source_file_name=data.get('source_file_name'),
        source_key=data.get('source_key'),
        model_type=model_type,
        architecture=architecture,
        sub_architecture=sub_architecture or 'general',
        compatibility_family=compatibility_family,
        asset_group_key=data.get('asset_group_key'),
        thumbnail_key=data.get('thumbnail_key'),
        thumbnail_url=data.get('thumbnail_url'),
        thumbnail_library_relative=data.get('thumbnail_library_relative'),
        source_provider=source_provider,
        source_model_id=_coerce_optional_str(data.get('source_model_id')),
        source_version_id=_coerce_optional_str(data.get('source_version_id')),
        token_env=token_env,
        catalog_source=data.get('catalog_source'),
        source_kind=data.get('source_kind', source_provider),
        sources=sources,
        storage_tier=_normalize_storage_tier(data.get('storage_tier', 'session')),
        visibility=data.get('visibility', 'generic'),
        preset_managed=bool(data.get('preset_managed', False)),
        token_required=bool(data.get('token_required', False)),
        tags=tuple(data.get('tags', [])),
    )


def _coerce_optional_str(value):
    if value is None:
        return None
    return str(value)


def _default_model_type(root_key: str) -> str:
    return {
        'checkpoints': 'checkpoint',
        'loras': 'lora',
        'unet': 'unet',
        'clip': 'clip',
        'vae': 'vae',
    }.get(root_key, root_key)


def _normalize_storage_tier(value) -> str:
    if value is None:
        return 'session'
    normalized = str(value).strip().lower()
    if normalized == 'colab':
        return 'session'
    if normalized == 'gdrive':
        return 'persistent'
    return normalized or 'session'

