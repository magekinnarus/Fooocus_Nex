from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from modules.model_download.catalog import ModelCatalog
from modules.model_download.spec import ModelCatalogEntry

ACTIVE_CATALOG_SUFFIX = '.catalog.json'
_RUNTIME_CATALOG_INDEX_CACHE: dict[tuple[str, ...], tuple[tuple[tuple[str, int, int], ...], 'ModelCatalogIndex']] = {}


def _normalize_selector(value: str | Path | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).replace('\\', '/').strip()
    return normalized or None


def _normalize_catalog_directories(directories: Iterable[str | Path]) -> tuple[str, ...]:
    return tuple(str(Path(directory).resolve()) for directory in directories)


def _build_catalog_snapshot(directories: Iterable[str | Path], pattern: str = f'*{ACTIVE_CATALOG_SUFFIX}') -> tuple[tuple[str, int, int], ...]:
    snapshot: list[tuple[str, int, int]] = []
    for path in iter_catalog_files(directories, pattern=pattern):
        stat = path.stat()
        snapshot.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
    return tuple(snapshot)


def _build_qualified_alias(alias: str, catalog_id: str | None) -> str | None:
    if not alias or not catalog_id:
        return None
    return f'{catalog_id}:{alias}'


@dataclass(frozen=True)
class CatalogSourceRecord:
    path: str
    catalog_id: str | None = None
    catalog_label: str | None = None
    entry_count: int = 0


@dataclass(frozen=True)
class CatalogIndexRecord:
    entry: ModelCatalogEntry
    source_path: str
    catalog_id: str | None = None
    catalog_label: str | None = None


class ModelCatalogIndex:
    def __init__(self, records: Iterable[CatalogIndexRecord] = (), sources: Iterable[CatalogSourceRecord] = ()):
        self._records: list[CatalogIndexRecord] = []
        self._records_by_id: dict[str, CatalogIndexRecord] = {}
        self._records_by_alias: dict[str, CatalogIndexRecord] = {}
        self._ambiguous_aliases: dict[str, list[CatalogIndexRecord]] = {}
        self._records_by_qualified_alias: dict[str, CatalogIndexRecord] = {}
        self._records_by_relative_path: dict[str, list[CatalogIndexRecord]] = {}
        self._records_by_name: dict[str, list[CatalogIndexRecord]] = {}
        self._sources: list[CatalogSourceRecord] = list(sources)
        for record in records:
            self.add(record)

    def add(self, record: CatalogIndexRecord) -> None:
        entry = record.entry
        if entry.id in self._records_by_id:
            existing = self._records_by_id[entry.id]
            raise ValueError(
                f'Duplicate catalog id: {entry.id} ({existing.source_path} vs {record.source_path})'
            )
        self._records.append(record)
        self._records_by_id[entry.id] = record

        qualified_alias = _build_qualified_alias(entry.alias, record.catalog_id)
        if qualified_alias is not None:
            existing = self._records_by_qualified_alias.get(qualified_alias)
            if existing is not None:
                raise ValueError(
                    f'Duplicate qualified catalog alias: {qualified_alias} ({existing.source_path} vs {record.source_path})'
                )
            self._records_by_qualified_alias[qualified_alias] = record

        if entry.alias:
            existing_alias = self._records_by_alias.get(entry.alias)
            if existing_alias is None and entry.alias not in self._ambiguous_aliases:
                self._records_by_alias[entry.alias] = record
            else:
                collisions = self._ambiguous_aliases.setdefault(entry.alias, [])
                if existing_alias is not None:
                    collisions.append(existing_alias)
                    del self._records_by_alias[entry.alias]
                collisions.append(record)

        normalized_relative_path = _normalize_selector(entry.relative_path)
        if normalized_relative_path:
            self._records_by_relative_path.setdefault(normalized_relative_path, []).append(record)

        normalized_name = _normalize_selector(entry.name)
        if normalized_name:
            self._records_by_name.setdefault(normalized_name, []).append(record)

    def get(self, selector: str) -> ModelCatalogEntry | None:
        record = self.get_record(selector)
        return None if record is None else record.entry

    def get_record(self, selector: str) -> CatalogIndexRecord | None:
        record = self._records_by_id.get(selector) or self._records_by_qualified_alias.get(selector)
        if record is not None:
            return record
        if selector in self._ambiguous_aliases:
            collisions = ', '.join(sorted(record.source_path for record in self._ambiguous_aliases[selector]))
            raise ValueError(
                f'Ambiguous catalog alias: {selector} ({collisions}). Use a qualified alias like <catalog_id>:{selector}.'
            )
        return self._records_by_alias.get(selector)

    def find_by_relative_path(self, relative_path: str | Path, root_keys: Iterable[str] | None = None) -> CatalogIndexRecord | None:
        normalized_path = _normalize_selector(relative_path)
        if normalized_path is None:
            return None
        return _first_filtered_record(self._records_by_relative_path.get(normalized_path, []), root_keys=root_keys)

    def find_by_name(self, name: str | Path, root_keys: Iterable[str] | None = None) -> CatalogIndexRecord | None:
        normalized_name = _normalize_selector(name)
        if normalized_name is None:
            return None
        return _first_filtered_record(self._records_by_name.get(normalized_name, []), root_keys=root_keys)

    def list(self) -> list[ModelCatalogEntry]:
        return [record.entry for record in self._records]

    def list_records(self) -> list[CatalogIndexRecord]:
        return list(self._records)

    def list_sources(self) -> list[CatalogSourceRecord]:
        return list(self._sources)

    def filter(
        self,
        *,
        architecture: str | None = None,
        sub_architecture: str | None = None,
        compatibility_family: str | None = None,
        model_type: str | None = None,
        root_key: str | None = None,
        storage_tier: str | None = None,
        visibility: str | None = None,
    ) -> list[ModelCatalogEntry]:
        results = []
        for record in self._records:
            entry = record.entry
            if architecture is not None and entry.architecture != architecture:
                continue
            if sub_architecture is not None and entry.sub_architecture != sub_architecture:
                continue
            if compatibility_family is not None and entry.compatibility_family != compatibility_family:
                continue
            if model_type is not None and entry.model_type != model_type:
                continue
            if root_key is not None and entry.root_key != root_key:
                continue
            if storage_tier is not None and entry.storage_tier != storage_tier:
                continue
            if visibility is not None and entry.visibility != visibility:
                continue
            results.append(entry)
        return results

    @classmethod
    def from_files(cls, paths: Iterable[str | Path]) -> 'ModelCatalogIndex':
        records: list[CatalogIndexRecord] = []
        sources: list[CatalogSourceRecord] = []
        for path in paths:
            payload_path = Path(path)
            payload = json.loads(payload_path.read_text(encoding='utf-8-sig'))
            catalog = ModelCatalog.from_dict(payload)
            catalog_id = payload.get('catalog_id') if isinstance(payload, dict) else None
            catalog_label = payload.get('catalog_label') if isinstance(payload, dict) else None
            source_path = str(payload_path)
            sources.append(
                CatalogSourceRecord(
                    path=source_path,
                    catalog_id=catalog_id,
                    catalog_label=catalog_label,
                    entry_count=len(catalog.list()),
                )
            )
            for entry in catalog.list():
                records.append(
                    CatalogIndexRecord(
                        entry=entry,
                        source_path=source_path,
                        catalog_id=catalog_id,
                        catalog_label=catalog_label,
                    )
                )
        return cls(records=records, sources=sources)

    @classmethod
    def from_directories(
        cls,
        directories: Iterable[str | Path],
        pattern: str = f'*{ACTIVE_CATALOG_SUFFIX}',
    ) -> 'ModelCatalogIndex':
        return cls.from_files(iter_catalog_files(directories, pattern=pattern))



def iter_catalog_files(directories: Iterable[str | Path], pattern: str = f'*{ACTIVE_CATALOG_SUFFIX}'):
    seen: set[str] = set()
    for directory in directories:
        root = Path(directory)
        if not root.is_dir():
            continue
        for path in sorted(root.glob(pattern)):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path



def is_runtime_catalog_file(path: str | Path) -> bool:
    return Path(path).name.endswith(ACTIVE_CATALOG_SUFFIX)



def clear_runtime_model_catalog_index_cache() -> None:
    _RUNTIME_CATALOG_INDEX_CACHE.clear()



def load_runtime_model_catalog_index(catalog_dirs: Iterable[str | Path] | None = None, *, force_refresh: bool = False) -> ModelCatalogIndex:
    if catalog_dirs is None:
        import modules.config as config

        catalog_dirs = config.get_model_catalog_directories()
    normalized_dirs = _normalize_catalog_directories(catalog_dirs)
    snapshot = _build_catalog_snapshot(normalized_dirs)
    cached = None if force_refresh else _RUNTIME_CATALOG_INDEX_CACHE.get(normalized_dirs)
    if cached is not None:
        cached_snapshot, cached_index = cached
        if cached_snapshot == snapshot:
            return cached_index
    index = ModelCatalogIndex.from_directories(normalized_dirs)
    _RUNTIME_CATALOG_INDEX_CACHE[normalized_dirs] = (snapshot, index)
    return index



def _first_filtered_record(records: Iterable[CatalogIndexRecord], root_keys: Iterable[str] | None = None) -> CatalogIndexRecord | None:
    if root_keys is None:
        for record in records:
            return record
        return None

    allowed_root_keys = {str(root_key) for root_key in root_keys}
    for record in records:
        if record.entry.root_key in allowed_root_keys:
            return record
    return None
