from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from .spec import ModelCatalogEntry


@dataclass
class ModelDownloadPolicy:
    root_map: dict[str, str | Sequence[str]]
    default_root_key: str = 'loras'

    def resolve_root_key(self, entry: ModelCatalogEntry) -> str:
        return entry.root_key or self.default_root_key

    def resolve_root_path(self, entry: ModelCatalogEntry) -> str:
        root_key = self.resolve_root_key(entry)
        if root_key not in self.root_map:
            raise KeyError(f'Unknown model root key: {root_key}')
        root_value = self.root_map[root_key]
        if isinstance(root_value, str):
            return root_value
        if isinstance(root_value, Sequence):
            for candidate in root_value:
                candidate_text = str(candidate or '').strip()
                if candidate_text:
                    return candidate_text
        raise KeyError(f'No configured filesystem path for model root key: {root_key}')

    def should_expose_in_generic_lora_list(self, entry: ModelCatalogEntry) -> bool:
        return (
            entry.visibility == 'generic'
            and entry.registration_state != 'unregistered'
            and not entry.preset_managed
        )
