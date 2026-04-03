from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Iterable

from modules.model_download.spec import REGISTRATION_STATE_UNREGISTERED, ModelCatalogEntry
from modules.model_manager_helpers import (
    _derive_clip_display_name_from_unet_payload,
    _derive_clip_name_from_unet_name,
    _entry_to_payload,
    _normalize_match_name,
    _score_match_candidate,
)


class ModelManagerCompanions:
    def __init__(self, manager: Any):
        self._manager = manager

    def _find_companion_clip_catalog_entry(self, entry_like: ModelCatalogEntry | dict[str, Any] | None) -> ModelCatalogEntry | None:
        if entry_like is None:
            return None
        if isinstance(entry_like, ModelCatalogEntry):
            asset_group_key = entry_like.asset_group_key
            architecture = entry_like.architecture
            sub_architecture = entry_like.sub_architecture
            source_provider = entry_like.source_provider
        else:
            asset_group_key = entry_like.get('asset_group_key')
            architecture = entry_like.get('architecture')
            sub_architecture = entry_like.get('sub_architecture')
            source_provider = entry_like.get('source_provider')

        if not asset_group_key:
            return None

        candidates: list[tuple[int, ModelCatalogEntry]] = []
        for candidate_record in self._manager._ensure_catalog_index().list_records():
            candidate = candidate_record.entry
            if candidate.root_key != 'clip':
                continue
            if candidate.registration_state == REGISTRATION_STATE_UNREGISTERED:
                continue
            if candidate.asset_group_key != asset_group_key:
                continue
            score = 0
            if architecture and candidate.architecture == architecture:
                score += 4
            if sub_architecture and candidate.sub_architecture == sub_architecture:
                score += 3
            if source_provider and candidate.source_provider == source_provider:
                score += 2
            if candidate.source is not None:
                score += 1
            candidates.append((score, candidate))

        candidates.sort(key=lambda item: (-item[0], item[1].display_name or item[1].name))
        return candidates[0][1] if candidates else None

    def _suggest_installed_companion_clips(
        self,
        *,
        target_clip_entry: ModelCatalogEntry | None,
        query_unet_entry: ModelCatalogEntry,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for record in self._manager.iter_inventory(root_key='clip', installed=True):
            candidate_entry = record.entry
            score = 0.0
            reasons: list[str] = []

            if target_clip_entry is not None:
                if (
                    target_clip_entry.asset_group_key
                    and candidate_entry.asset_group_key
                    and candidate_entry.asset_group_key == target_clip_entry.asset_group_key
                ):
                    score += 120.0
                    reasons.append('asset_group_key_exact')
                base_score, base_reasons = _score_match_candidate(
                    target_clip_entry,
                    candidate_entry,
                    source_provider=target_clip_entry.source_provider,
                    source_version_id=target_clip_entry.source_version_id,
                )
                score += base_score
                reasons.extend(base_reasons)

            unet_similarity = SequenceMatcher(
                None,
                _normalize_match_name(query_unet_entry.name),
                _normalize_match_name(candidate_entry.name),
            ).ratio()
            if unet_similarity > 0:
                score += unet_similarity * 18.0
                reasons.append(f'unet_name_similarity:{unet_similarity:.2f}')

            if score < 8.0 and 'asset_group_key_exact' not in reasons:
                continue

            candidates.append({
                'score': round(score, 2),
                'reasons': reasons,
                'entry': record.to_dict(),
            })

        candidates.sort(key=lambda item: (-item['score'], item['entry']['display_name'] or item['entry']['name']))
        return candidates[: max(1, int(limit))]

    def resolve_companion_clip(self, selector_or_entry: str | ModelCatalogEntry, *, installed_only: bool = False) -> ModelCatalogEntry | None:
        entry = selector_or_entry if isinstance(selector_or_entry, ModelCatalogEntry) else self._manager.get_entry(str(selector_or_entry))
        if entry is None or entry.root_key != 'unet':
            return None

        target_clip_entry = self._find_companion_clip_catalog_entry(entry)
        if not installed_only:
            return target_clip_entry

        if target_clip_entry is not None:
            record = self._manager.inventory_record(target_clip_entry)
            if record.installed:
                return target_clip_entry

        installed_candidates = self._suggest_installed_companion_clips(
            target_clip_entry=target_clip_entry,
            query_unet_entry=entry,
            limit=1,
        )
        if not installed_candidates:
            return None
        candidate_id = installed_candidates[0].get('entry', {}).get('id')
        return self._manager.get_entry(str(candidate_id)) if candidate_id else None

    def _build_unet_companion_clip_context(
        self,
        entry: ModelCatalogEntry,
        *,
        matched_selector: str | None = None,
        suggestions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        matched_entry = self._manager.get_entry(matched_selector) if matched_selector else None
        if matched_entry is None and suggestions:
            first_id = suggestions[0].get('entry', {}).get('id')
            if first_id:
                matched_entry = self._manager.get_entry(str(first_id))

        target_clip_entry = self._find_companion_clip_catalog_entry(matched_entry or entry)
        installed_candidates = self._suggest_installed_companion_clips(
            target_clip_entry=target_clip_entry,
            query_unet_entry=entry,
        )
        expected_name = target_clip_entry.name if target_clip_entry is not None else _derive_clip_name_from_unet_name((matched_entry or entry).name)
        recommended_selector = installed_candidates[0]['entry']['id'] if installed_candidates else None
        return {
            'required': True,
            'expected_name': expected_name,
            'matched_catalog_entry': self._manager.inventory_record(target_clip_entry).to_dict() if target_clip_entry is not None else None,
            'installed_candidates': installed_candidates,
            'recommended_selector': recommended_selector,
            'needs_user_path': not bool(installed_candidates),
        }

    def _register_unet_companion_clip(
        self,
        entry: ModelCatalogEntry,
        *,
        matched_selector: str | None = None,
        companion_selector: str | None = None,
        companion_relative_path: str | None = None,
    ) -> dict[str, Any] | None:
        if entry.root_key != 'unet':
            return None

        target_clip_entry = self._find_companion_clip_catalog_entry(self._manager.get_entry(matched_selector) if matched_selector else entry)
        companion_record = None
        if companion_selector:
            companion_record = self._manager._find_catalog_record(companion_selector, root_keys=['clip'])
            if companion_record is None:
                raise KeyError(f'Unknown companion clip selector: {companion_selector}')
        elif companion_relative_path:
            companion_record = self._manager._find_catalog_record(companion_relative_path, root_keys=['clip'])
        else:
            suggestions = self._suggest_installed_companion_clips(
                target_clip_entry=target_clip_entry,
                query_unet_entry=entry,
                limit=1,
            )
            if suggestions:
                companion_record = self._manager._find_catalog_record(suggestions[0]['entry']['id'], root_keys=['clip'])

        if companion_record is None:
            return {
                'status': 'missing',
                'expected_name': target_clip_entry.name if target_clip_entry is not None else _derive_clip_name_from_unet_name(entry.name),
                'matched_catalog_entry': self._manager.inventory_record(target_clip_entry).to_dict() if target_clip_entry is not None else None,
                'needs_user_path': True,
            }

        companion_updates: dict[str, Any] = {}
        companion_matched_selector = target_clip_entry.id if target_clip_entry is not None else None
        if companion_matched_selector is None:
            companion_updates['name'] = _derive_clip_name_from_unet_name(entry.name)
            companion_updates['display_name'] = _derive_clip_display_name_from_unet_payload(_entry_to_payload(entry))
            companion_updates['architecture'] = entry.architecture
            companion_updates['sub_architecture'] = entry.sub_architecture
            companion_updates['compatibility_family'] = entry.compatibility_family
            companion_updates['source_provider'] = entry.source_provider
            if entry.asset_group_key:
                companion_updates['asset_group_key'] = entry.asset_group_key

        companion_entry = self._manager._register_single_model_entry(
            companion_record.entry.id,
            matched_selector=companion_matched_selector,
            updates=companion_updates,
        )
        return {
            'status': 'registered',
            'entry': self._manager.inventory_record(companion_entry).to_dict(),
            'matched_catalog_entry': self._manager.inventory_record(target_clip_entry).to_dict() if target_clip_entry is not None else None,
            'needs_user_path': False,
        }
