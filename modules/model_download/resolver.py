from __future__ import annotations

import os
from abc import ABC, abstractmethod

from .policy import ModelDownloadPolicy
from .spec import DownloadPlan, ModelCatalogEntry, ModelSource


class DownloadResolver(ABC):
    @abstractmethod
    def resolve(self, entry: ModelCatalogEntry, policy: ModelDownloadPolicy) -> DownloadPlan:
        raise NotImplementedError


class DirectResolver(DownloadResolver):
    def resolve(self, entry: ModelCatalogEntry, policy: ModelDownloadPolicy) -> DownloadPlan:
        source = _first_source(entry)
        destination_root = policy.resolve_root_path(entry)
        destination_path = os.path.join(destination_root, entry.relative_path)
        return DownloadPlan(
            entry=entry,
            destination_root=destination_root,
            destination_path=destination_path,
            resolved_url=source.url,
            headers=source.headers,
            transport='aria2',
        )


class CivitAIResolver(DownloadResolver):
    def __init__(self, token_env: str = 'CIVITAI_TOKEN'):
        self.token_env = token_env

    def resolve(self, entry: ModelCatalogEntry, policy: ModelDownloadPolicy) -> DownloadPlan:
        source = _first_source(entry)
        token = os.getenv(self.token_env, '')
        resolved_url = source.url
        if token and 'civitai.com/api/download/models/' not in resolved_url:
            resolved_url = f'{resolved_url}{"&" if "?" in resolved_url else "?"}token={token}'

        destination_root = policy.resolve_root_path(entry)
        destination_path = os.path.join(destination_root, entry.relative_path)
        return DownloadPlan(
            entry=entry,
            destination_root=destination_root,
            destination_path=destination_path,
            resolved_url=resolved_url,
            headers=source.headers,
            transport='aria2',
        )


class HuggingFaceResolver(DownloadResolver):
    def __init__(self, token_env: str = 'HUGGINGFACE_TOKEN'):
        self.token_env = token_env

    def resolve(self, entry: ModelCatalogEntry, policy: ModelDownloadPolicy) -> DownloadPlan:
        source = _first_source(entry)
        headers = list(source.headers)
        token = os.getenv(self.token_env, '')
        if entry.token_required and token:
            headers.append(('Authorization', f'Bearer {token}'))

        destination_root = policy.resolve_root_path(entry)
        destination_path = os.path.join(destination_root, entry.relative_path)
        return DownloadPlan(
            entry=entry,
            destination_root=destination_root,
            destination_path=destination_path,
            resolved_url=source.url,
            headers=tuple(headers),
            transport='aria2',
        )


def _first_source(entry: ModelCatalogEntry) -> ModelSource:
    if not entry.sources:
        raise ValueError(f'Catalog entry {entry.id} does not define any sources')
    return entry.sources[0]
