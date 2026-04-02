from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


REGISTRATION_STATE_UNREGISTERED = 'unregistered'
REGISTRATION_STATE_LOCALLY_REGISTERED = 'locally_registered'
REGISTRATION_STATE_SOURCED_REGISTERED = 'sourced_registered'
REGISTRATION_STATES = (
    REGISTRATION_STATE_UNREGISTERED,
    REGISTRATION_STATE_LOCALLY_REGISTERED,
    REGISTRATION_STATE_SOURCED_REGISTERED,
)


@dataclass(frozen=True)
class ModelSource:
    url: str
    token_env: str | None = None
    headers: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ModelCatalogEntry:
    id: str
    name: str
    root_key: str
    relative_path: str
    alias: str | None = None
    display_name: str | None = None
    model_type: str = 'unknown'
    architecture: str = 'unknown'
    sub_architecture: str = 'general'
    compatibility_family: str | None = None
    asset_group_key: str | None = None
    thumbnail_library_relative: str | None = None
    source_provider: str = 'direct'
    source_version_id: str | None = None
    source: ModelSource | None = None
    registration_state: str = REGISTRATION_STATE_SOURCED_REGISTERED
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

