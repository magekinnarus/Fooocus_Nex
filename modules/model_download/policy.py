from __future__ import annotations

from dataclasses import dataclass

from .spec import ModelCatalogEntry


@dataclass
class ModelDownloadPolicy:
    root_map: dict[str, str]
    default_root_key: str = 'loras'

    def resolve_root_key(self, entry: ModelCatalogEntry) -> str:
        return entry.root_key or self.default_root_key

    def resolve_root_path(self, entry: ModelCatalogEntry) -> str:
        root_key = self.resolve_root_key(entry)
        if root_key not in self.root_map:
            raise KeyError(f'Unknown model root key: {root_key}')
        return self.root_map[root_key]

    def should_expose_in_generic_lora_list(self, entry: ModelCatalogEntry) -> bool:
        return entry.visibility == 'generic' and not entry.preset_managed
