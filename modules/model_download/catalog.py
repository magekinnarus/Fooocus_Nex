from __future__ import annotations

import json
from pathlib import Path
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
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
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
    sources = tuple(
        ModelSource(
            url=source['url'],
            kind=source.get('kind', data.get('source_kind', 'direct')),
            token_env=source.get('token_env'),
            headers=tuple(tuple(header) for header in source.get('headers', [])),
        )
        for source in data.get('sources', [])
        if source.get('url')
    )
    return ModelCatalogEntry(
        id=data['id'],
        alias=data.get('alias'),
        name=data['name'],
        root_key=data['root_key'],
        relative_path=data['relative_path'],
        source_kind=data.get('source_kind', 'direct'),
        sources=sources,
        storage_tier=data.get('storage_tier', 'colab'),
        visibility=data.get('visibility', 'generic'),
        preset_managed=bool(data.get('preset_managed', False)),
        token_required=bool(data.get('token_required', False)),
        tags=tuple(data.get('tags', [])),
    )
