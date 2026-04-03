from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import modules.config as config
from modules.model_download.spec import REGISTRATION_STATE_UNREGISTERED, ModelCatalogEntry
from modules.model_manager_helpers import (
    AUTO_GENERATED_UNREGISTERED_TAGS,
    INSTALLED_MODEL_LINKS_FILENAME,
    INSTALLED_MODEL_LINKS_ID,
    INSTALLED_MODEL_LINKS_LABEL,
    UNREGISTERED_INSTALL_CATALOG_FILENAME,
    UNREGISTERED_INSTALL_CATALOG_ID,
    UNREGISTERED_INSTALL_CATALOG_LABEL,
    USER_FACING_DISCOVERY_ROOT_KEYS,
    _build_unregistered_entry_id,
    _default_model_type_for_root,
    _entry_to_payload,
    _normalize_absolute_path,
    _normalize_generated_sub_architecture,
    _normalize_lookup_key,
    _normalize_path,
)
from modules.model_manager_runtime import InstalledModelLink, ModelInventoryRecord


class ModelManagerLinks:
    def __init__(self, manager: Any):
        self._manager = manager

    def _installed_links_path(self) -> Path:
        writable_dir = Path(self._manager._get_writable_catalog_directory())
        writable_dir.mkdir(parents=True, exist_ok=True)
        return writable_dir / INSTALLED_MODEL_LINKS_FILENAME

    def _load_installed_links_payload(self) -> dict[str, Any]:
        path = self._installed_links_path()
        if not path.exists():
            return {
                'catalog_id': INSTALLED_MODEL_LINKS_ID,
                'catalog_label': INSTALLED_MODEL_LINKS_LABEL,
                'links': [],
            }
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault('catalog_id', INSTALLED_MODEL_LINKS_ID)
        payload.setdefault('catalog_label', INSTALLED_MODEL_LINKS_LABEL)
        if not isinstance(payload.get('links'), list):
            payload['links'] = []
        return payload

    def _write_installed_links_payload(self, payload: dict[str, Any]) -> None:
        path = self._installed_links_path()
        path.write_text(json.dumps(payload, indent=4, ensure_ascii=False) + '\n', encoding='utf-8')

    def refresh_installed_links(self) -> list[InstalledModelLink]:
        payload = self._load_installed_links_payload()
        valid_entry_ids = {
            str(record.entry.id)
            for record in self._manager._ensure_catalog_index().list_records()
        }
        links: list[InstalledModelLink] = []
        links_by_entry_id: dict[str, list[InstalledModelLink]] = {}
        links_by_relative_path: dict[str, list[InstalledModelLink]] = {}
        links_by_name: dict[str, list[InstalledModelLink]] = {}
        links_by_path: dict[str, InstalledModelLink] = {}
        normalized_links_payload: list[dict[str, Any]] = []
        payload_changed = False

        for item in payload.get('links', []):
            if not isinstance(item, dict):
                payload_changed = True
                continue
            entry_id = str(item.get('entry_id') or '').strip()
            root_key = str(item.get('root_key') or '').strip()
            installed_path = _normalize_absolute_path(item.get('installed_path'))
            if not entry_id or not root_key or not installed_path:
                payload_changed = True
                continue
            if entry_id not in valid_entry_ids:
                payload_changed = True
                continue
            if not os.path.exists(installed_path):
                payload_changed = True
                continue
            installed_root_path = _normalize_absolute_path(item.get('installed_root_path'))
            installed_relative_path = _normalize_path(item.get('installed_relative_path'))
            install_id = str(item.get('install_id') or hashlib.sha1(f'{entry_id}:{installed_path}'.encode('utf-8')).hexdigest()[:12])
            link = InstalledModelLink(
                install_id=install_id,
                entry_id=entry_id,
                root_key=root_key,
                installed_path=installed_path,
                installed_root_path=installed_root_path,
                installed_relative_path=installed_relative_path,
            )
            links.append(link)
            links_by_entry_id.setdefault(entry_id, []).append(link)
            if installed_relative_path:
                links_by_relative_path.setdefault(installed_relative_path.lower(), []).append(link)
            links_by_name.setdefault(os.path.basename(installed_path).lower(), []).append(link)
            links_by_path[installed_path] = link
            normalized_entry = link.to_dict()
            normalized_links_payload.append(normalized_entry)
            if item != normalized_entry:
                payload_changed = True

        with self._manager._index_lock:
            self._manager._installed_links = links
            self._manager._installed_links_by_entry_id = links_by_entry_id
            self._manager._installed_links_by_relative_path = links_by_relative_path
            self._manager._installed_links_by_name = links_by_name
            self._manager._installed_links_by_path = links_by_path
        if payload_changed:
            payload['links'] = normalized_links_payload
            self._write_installed_links_payload(payload)
        return links

    def _ensure_installed_links(self):
        with self._manager._index_lock:
            if not self._manager._installed_links_by_entry_id and not self._manager._installed_links:
                self.refresh_installed_links()
            return list(self._manager._installed_links)

    def _find_installed_link_for_entry(self, entry_id: str) -> InstalledModelLink | None:
        self._ensure_installed_links()
        links = self._manager._installed_links_by_entry_id.get(str(entry_id), [])
        for link in links:
            if os.path.exists(link.installed_path):
                return link
        return links[0] if links else None

    def _ensure_persisted_installed_link(self, entry: ModelCatalogEntry, inventory_record: ModelInventoryRecord | None = None) -> InstalledModelLink | None:
        inventory_record = inventory_record or self._manager.inventory_record(entry)
        if not inventory_record.installed or not inventory_record.installed_path:
            return None

        existing_link = self._find_installed_link_for_entry(entry.id)
        if existing_link is not None and os.path.exists(existing_link.installed_path):
            return existing_link

        return self._upsert_installed_link(
            entry_id=entry.id,
            root_key=entry.root_key,
            installed_path=inventory_record.installed_path,
            installed_root_path=inventory_record.installed_root_path,
            installed_relative_path=inventory_record.installed_relative_path or entry.relative_path,
        )

    def _find_installed_link_by_selector(self, selector: str, root_keys: Iterable[str] | None = None) -> InstalledModelLink | None:
        self._ensure_installed_links()
        normalized_selector = _normalize_lookup_key(selector)
        if normalized_selector is None:
            return None
        allowed_root_keys = None if root_keys is None else {str(root_key) for root_key in root_keys}

        candidates: list[InstalledModelLink] = []
        candidates.extend(self._manager._installed_links_by_relative_path.get(normalized_selector, []))
        candidates.extend(self._manager._installed_links_by_name.get(os.path.basename(normalized_selector).lower(), []))

        seen: set[str] = set()
        for link in candidates:
            if link.install_id in seen:
                continue
            seen.add(link.install_id)
            if allowed_root_keys is not None and link.root_key not in allowed_root_keys:
                continue
            return link
        return None

    def _upsert_installed_link(
        self,
        *,
        entry_id: str,
        root_key: str,
        installed_path: str,
        installed_root_path: str | None = None,
        installed_relative_path: str | None = None,
    ) -> InstalledModelLink:
        payload = self._load_installed_links_payload()
        normalized_installed_path = _normalize_absolute_path(installed_path)
        normalized_root_path = _normalize_absolute_path(installed_root_path)
        normalized_relative_path = _normalize_path(installed_relative_path)
        install_id = hashlib.sha1(f'{entry_id}:{normalized_installed_path}'.encode('utf-8')).hexdigest()[:12]
        links = [item for item in payload.get('links', []) if isinstance(item, dict)]
        filtered = []
        for item in links:
            if str(item.get('entry_id') or '') == str(entry_id):
                continue
            if _normalize_absolute_path(item.get('installed_path')) == normalized_installed_path:
                continue
            filtered.append(item)
        filtered.append({
            'install_id': install_id,
            'entry_id': str(entry_id),
            'root_key': str(root_key),
            'installed_path': normalized_installed_path,
            'installed_root_path': normalized_root_path,
            'installed_relative_path': normalized_relative_path,
        })
        payload['links'] = filtered
        self._write_installed_links_payload(payload)
        self.refresh_installed_links()
        link = self._find_installed_link_for_entry(str(entry_id))
        if link is None:
            raise RuntimeError(f'Installed link for {entry_id} disappeared after write')
        return link

    def _discovery_roots_for_key(self, root_key: str) -> list[str]:
        roots = list(self._manager._root_map.get(root_key, []))
        if root_key != 'checkpoints':
            return roots

        overlapping_roots = set()
        for other_root_key in USER_FACING_DISCOVERY_ROOT_KEYS:
            if other_root_key == root_key:
                continue
            overlapping_roots.update(self._manager._root_map.get(other_root_key, []))
        return [root for root in roots if root not in overlapping_roots]

    def _unregistered_catalog_path(self) -> Path:
        writable_dir = Path(self._manager._get_writable_catalog_directory())
        writable_dir.mkdir(parents=True, exist_ok=True)
        return writable_dir / UNREGISTERED_INSTALL_CATALOG_FILENAME

    def _build_unregistered_catalog_payload(self, entries: list[ModelCatalogEntry]) -> dict[str, Any]:
        path = self._unregistered_catalog_path()
        if path.exists():
            payload = self._manager._load_catalog_payload(str(path))
            if not isinstance(payload, dict):
                payload = {}
        else:
            payload = {}

        existing_entries = payload.get('entries', [])
        if not isinstance(existing_entries, list):
            existing_entries = []

        preserved_entries = []
        for entry_data in existing_entries:
            if not isinstance(entry_data, dict):
                continue
            tags = {str(tag) for tag in entry_data.get('tags', [])}
            if entry_data.get('registration_state') == REGISTRATION_STATE_UNREGISTERED and 'auto_generated' in tags:
                continue
            preserved_entries.append(entry_data)

        payload['catalog_id'] = payload.get('catalog_id') or UNREGISTERED_INSTALL_CATALOG_ID
        payload['catalog_label'] = payload.get('catalog_label') or UNREGISTERED_INSTALL_CATALOG_LABEL
        payload['entries'] = preserved_entries + [_entry_to_payload(entry) for entry in entries]
        return payload

    def _build_unregistered_entry(self, root_key: str, relative_path: str) -> ModelCatalogEntry:
        normalized_relative_path = _normalize_path(relative_path)
        if normalized_relative_path is None:
            raise ValueError('relative_path is required for unregistered discovery')

        taxonomy = config.resolve_model_taxonomy(
            normalized_relative_path,
            root_keys=(root_key,),
            folder_paths=self._manager._root_map.get(root_key, []),
        )
        architecture = taxonomy.architecture
        sub_architecture = _normalize_generated_sub_architecture(
            root_key,
            taxonomy.architecture,
            taxonomy.sub_architecture,
        )

        return ModelCatalogEntry(
            id=_build_unregistered_entry_id(root_key, normalized_relative_path),
            name=os.path.basename(normalized_relative_path),
            root_key=root_key,
            relative_path=normalized_relative_path,
            display_name=Path(os.path.basename(normalized_relative_path)).stem.replace('_', ' '),
            model_type=_default_model_type_for_root(root_key),
            architecture=architecture,
            sub_architecture=sub_architecture or 'general',
            compatibility_family=taxonomy.compatibility_family,
            source_provider='local',
            registration_state=REGISTRATION_STATE_UNREGISTERED,
            visibility='generic',
            preset_managed=False,
            token_required=False,
            tags=AUTO_GENERATED_UNREGISTERED_TAGS,
        )

    def discover_unregistered_installed_entries(self) -> list[ModelCatalogEntry]:
        installed_index = self._manager._ensure_installed_index()
        linked_paths = {
            _normalize_absolute_path(link.installed_path)
            for link in self._ensure_installed_links()
            if _normalize_absolute_path(link.installed_path) is not None
        }
        discovered: dict[str, ModelCatalogEntry] = {}
        for root_key in USER_FACING_DISCOVERY_ROOT_KEYS:
            roots = self._discovery_roots_for_key(root_key)
            if not roots:
                continue
            for root in roots:
                root_index = installed_index.get(root)
                if root_index is None:
                    continue
                for relative_path, absolute_path in sorted(root_index.relative_paths.items()):
                    normalized_relative_path = _normalize_path(relative_path)
                    normalized_absolute_path = _normalize_absolute_path(absolute_path)
                    if normalized_relative_path is None:
                        continue
                    if normalized_absolute_path in linked_paths:
                        continue
                    existing_record = self._manager._find_catalog_record(normalized_relative_path, root_keys=[root_key])
                    if existing_record is not None and not self._manager._is_auto_generated_unregistered_record(existing_record):
                        continue
                    entry = self._build_unregistered_entry(root_key, normalized_relative_path)
                    discovered.setdefault(entry.id, entry)
        return [discovered[key] for key in sorted(discovered)]

    def sync_unregistered_install_catalog(self) -> dict[str, Any]:
        entries = self.discover_unregistered_installed_entries()
        path = self._unregistered_catalog_path()
        if not entries and not path.exists():
            return {
                'path': str(path),
                'entry_count': 0,
                'changed': False,
            }

        payload = self._build_unregistered_catalog_payload(entries)
        new_text = json.dumps(payload, indent=4, ensure_ascii=False) + '\n'
        old_text = path.read_text(encoding='utf-8-sig') if path.exists() else None
        changed = old_text != new_text
        if changed:
            path.write_text(new_text, encoding='utf-8')
            self._manager.refresh_catalog_index(force_refresh=True)
        return {
            'path': str(path),
            'entry_count': len(entries),
            'changed': changed,
        }
