from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import modules.model_taxonomy

from .spec import (
    REGISTRATION_STATES,
    REGISTRATION_STATE_LOCALLY_REGISTERED,
    REGISTRATION_STATE_SOURCED_REGISTERED,
    REGISTRATION_STATE_UNREGISTERED,
    ModelCatalogEntry,
    ModelSource,
)


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

    def filter(
        self,
        *,
        registration_state: str | None = None,
        visibility: str | None = None,
    ) -> list[ModelCatalogEntry]:
        results = []
        for entry in self._entries_by_id.values():
            if registration_state is not None and entry.registration_state != registration_state:
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
        if 'id' in node and 'name' in node and 'root_key' in node:
            yield node
        else:
            for value in node.values():
                yield from _iter_entry_dicts(value)


def _normalize_registration_state(value: str | None, source_provider: str) -> str:
    if value is None:
        if str(source_provider).strip().lower() == 'local':
            return REGISTRATION_STATE_LOCALLY_REGISTERED
        return REGISTRATION_STATE_SOURCED_REGISTERED

    normalized = str(value).strip().lower()
    if normalized not in REGISTRATION_STATES:
        raise ValueError(
            f"Catalog entry registration_state must be one of {', '.join(REGISTRATION_STATES)}."
        )
    return normalized


LOCAL_REGISTRATION_STATES = {
    REGISTRATION_STATE_UNREGISTERED,
    REGISTRATION_STATE_LOCALLY_REGISTERED,
}


def _parse_source(data: dict, *, source_provider: str, registration_state: str) -> ModelSource | None:
    source_data = data.get('source')
    if source_data is None:
        if str(source_provider).strip().lower() == 'local' or registration_state in LOCAL_REGISTRATION_STATES:
            return None
        raise ValueError(
            f"Catalog entry {data.get('id', '<unknown>')} is missing required 'source' metadata."
        )

    if not isinstance(source_data, dict) or not source_data.get('url'):
        if str(source_provider).strip().lower() == 'local' or registration_state in LOCAL_REGISTRATION_STATES:
            return None
        raise ValueError(
            f"Catalog entry {data.get('id', '<unknown>')} must define source.url."
        )

    return ModelSource(
        url=source_data['url'],
        token_env=source_data.get('token_env'),
        headers=tuple(tuple(header) for header in source_data.get('headers', [])),
    )


def _resolve_relative_path(data: dict, *, root_key: str, architecture: str, sub_architecture: str | None, source_provider: str, registration_state: str) -> str:
    explicit_relative_path = _coerce_optional_str(data.get('relative_path'))
    if explicit_relative_path:
        return explicit_relative_path

    if str(source_provider).strip().lower() != 'local' and registration_state != REGISTRATION_STATE_UNREGISTERED:
        return modules.model_taxonomy.build_canonical_relative_path(
            root_key=root_key,
            architecture=architecture,
            sub_architecture=sub_architecture,
            name=data['name'],
        )

    raise ValueError(
        f"Catalog entry {data.get('id', '<unknown>')} must define relative_path or enough metadata to derive it."
    )


def _entry_from_dict(data: dict) -> ModelCatalogEntry:
    source_provider = data.get('source_provider', 'direct')
    registration_state = _normalize_registration_state(data.get('registration_state'), source_provider)
    source = _parse_source(data, source_provider=source_provider, registration_state=registration_state)
    root_key = _normalize_root_key(data['root_key'])
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
    relative_path = _resolve_relative_path(
        data,
        root_key=root_key,
        architecture=architecture,
        sub_architecture=sub_architecture,
        source_provider=source_provider,
        registration_state=registration_state,
    )

    return ModelCatalogEntry(
        id=data['id'],
        alias=data.get('alias'),
        name=data['name'],
        root_key=root_key,
        relative_path=relative_path,
        display_name=display_name,
        model_type=model_type,
        architecture=architecture,
        sub_architecture=sub_architecture or 'general',
        compatibility_family=compatibility_family,
        asset_group_key=data.get('asset_group_key'),
        thumbnail_library_relative=data.get('thumbnail_library_relative'),
        source_provider=source_provider,
        source_version_id=_coerce_optional_str(data.get('source_version_id')),
        source=source,
        registration_state=registration_state,
        visibility=data.get('visibility', 'generic'),
        preset_managed=bool(data.get('preset_managed', False)),
        token_required=bool(data.get('token_required', False)),
        tags=tuple(data.get('tags', [])),
    )


def _coerce_optional_str(value):
    if value is None:
        return None
    return str(value)


def _normalize_root_key(value: str) -> str:
    normalized = str(value).strip().lower()
    return {
        'checkpoint': 'checkpoints',
        'lora': 'loras',
        'embedding': 'embeddings',
    }.get(normalized, normalized)


def _default_model_type(root_key: str) -> str:
    return {
        'checkpoints': 'checkpoint',
        'loras': 'lora',
        'unet': 'unet',
        'clip': 'clip',
        'vae': 'vae',
    }.get(root_key, root_key)
