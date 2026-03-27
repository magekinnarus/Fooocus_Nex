from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ModelSource:
    url: str
    kind: str = 'direct'
    token_env: str | None = None
    headers: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ModelCatalogEntry:
    id: str
    name: str
    root_key: str
    relative_path: str
    alias: str | None = None
    source_kind: str = 'direct'
    sources: Tuple[ModelSource, ...] = ()
    storage_tier: str = 'colab'
    visibility: str = 'generic'
    preset_managed: bool = False
    token_required: bool = False
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DownloadPlan:
    entry: ModelCatalogEntry
    destination_root: str
    destination_path: str
    resolved_url: str
    headers: Tuple[Tuple[str, str], ...] = ()
    transport: str = 'aria2'


@dataclass(frozen=True)
class DownloadResult:
    success: bool
    destination_path: str
    transport: str
    message: str = ''
    skipped: bool = False
