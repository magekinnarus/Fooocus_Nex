from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import modules.config as config
import modules.model_catalog_index as catalog_index
import modules.model_taxonomy as model_taxonomy
from modules.extra_utils import get_files_from_folder
from modules.model_download.spec import ModelCatalogEntry


MODEL_FILE_EXTENSIONS = [
    '.pth',
    '.ckpt',
    '.bin',
    '.safetensors',
    '.fooocus.patch',
    '.gguf',
    '.sft',
]


def _normalize_path(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).replace('\\', '/').strip()
    return normalized or None


def _normalize_paths(values: Iterable[str | os.PathLike[str]] | str | os.PathLike[str] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, os.PathLike)):
        values = [values]
    results: list[str] = []
    for value in values:
        normalized = os.path.abspath(os.path.realpath(str(value)))
        if normalized not in results:
            results.append(normalized)
    return results


def _build_default_root_map() -> dict[str, list[str]]:
    root_map: dict[str, list[str]] = {
        'checkpoints': _normalize_paths(config.paths_checkpoints),
        'loras': _normalize_paths(config.paths_loras),
        'loras_lcm': _normalize_paths(config.path_loras_lcm),
        'loras_lightning': _normalize_paths(config.path_loras_lightning),
        'faceid_loras': _normalize_paths(config.path_faceid_loras),
        'embeddings': _normalize_paths(config.path_embeddings),
        'vae_approx': _normalize_paths(config.path_vae_approx),
        'vae': _normalize_paths(config.path_vae),
        'unet': _normalize_paths(config.path_unet),
        'clip': _normalize_paths(config.paths_clips),
        'upscale_models': _normalize_paths(config.path_upscale_models),
        'inpaint': _normalize_paths(config.path_inpaint),
        'controlnet_models': _normalize_paths(config.path_controlnet),
        'clip_vision': _normalize_paths(config.path_clip_vision),
        'vision_support': _normalize_paths(config.path_vision_support),
        'preprocessors': _normalize_paths(config.path_preprocessors),
        'insightface': _normalize_paths(config.path_insightface),
        'removals': _normalize_paths(config.path_removals),
    }
    return {key: paths for key, paths in root_map.items() if paths}

def _coerce_result_path(result: Any) -> str | None:
    if result is None:
        return None
    if isinstance(result, str):
        return result
    for attr in ('destination_path', 'path', 'result_path'):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value:
            return value
    if isinstance(result, dict):
        for key in ('destination_path', 'path', 'result_path'):
            value = result.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _coerce_result_message(result: Any) -> str:
    if result is None:
        return ''
    if isinstance(result, str):
        return result
    message = getattr(result, 'message', None)
    if isinstance(message, str) and message:
        return message
    if isinstance(result, dict):
        value = result.get('message')
        if isinstance(value, str) and value:
            return value
    return ''

def _coerce_result_success(result: Any) -> bool:
    if result is None:
        return False
    if isinstance(result, dict) and 'success' in result:
        return bool(result.get('success'))
    success = getattr(result, 'success', None)
    if success is None:
        return True
    return bool(success)


def _default_filter_sub_architecture(
    architecture: str | None,
    sub_architecture: str | None,
    *,
    root_key: str | None = None,
    model_type: str | None = None,
) -> str | None:
    architecture = model_taxonomy.normalize_architecture(architecture)
    sub_architecture = model_taxonomy.normalize_sub_architecture(sub_architecture, architecture=architecture)

    if architecture != model_taxonomy.ARCHITECTURE_SDXL:
        return None

    if root_key == 'loras' and model_type == 'lora' and sub_architecture == model_taxonomy.SUB_ARCHITECTURE_NOOB:
        return model_taxonomy.SUB_ARCHITECTURE_ILLUSTRIOUS

    return sub_architecture


def _matches_filter_scope(
    candidate_architecture: str | None,
    candidate_sub_architecture: str | None,
    *,
    target_architecture: str | None = None,
    target_sub_architecture: str | None = None,
    root_key: str | None = None,
    model_type: str | None = None,
) -> bool:
    normalized_target_architecture = model_taxonomy.normalize_architecture(target_architecture)
    normalized_target_sub_architecture = _default_filter_sub_architecture(
        normalized_target_architecture,
        target_sub_architecture,
        root_key=root_key,
        model_type=model_type,
    )
    normalized_candidate_architecture = model_taxonomy.normalize_architecture(candidate_architecture)
    normalized_candidate_sub_architecture = _default_filter_sub_architecture(
        normalized_candidate_architecture,
        candidate_sub_architecture,
        root_key=root_key,
        model_type=model_type,
    )

    if normalized_target_architecture is not None and normalized_candidate_architecture != normalized_target_architecture:
        return False
    if normalized_target_sub_architecture is not None and normalized_candidate_sub_architecture != normalized_target_sub_architecture:
        return False
    return True


@dataclass(frozen=True)
class InstalledPathIndex:
    root_path: str
    relative_paths: dict[str, str] = field(default_factory=dict)
    basenames: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelInventoryRecord:
    entry: ModelCatalogEntry
    installed: bool
    installed_path: str | None = None
    installed_root_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.entry.id,
            'name': self.entry.name,
            'display_name': self.entry.display_name,
            'root_key': self.entry.root_key,
            'relative_path': self.entry.relative_path,
            'alias': self.entry.alias,
            'model_type': self.entry.model_type,
            'architecture': self.entry.architecture,
            'sub_architecture': self.entry.sub_architecture,
            'compatibility_family': self.entry.compatibility_family,
            'source_provider': self.entry.source_provider,
            'source_model_id': self.entry.source_model_id,
            'source_version_id': self.entry.source_version_id,
            'catalog_source': self.entry.catalog_source,
            'storage_tier': self.entry.storage_tier,
            'visibility': self.entry.visibility,
            'preset_managed': self.entry.preset_managed,
            'token_required': self.entry.token_required,
            'thumbnail_key': self.entry.thumbnail_key,
            'thumbnail_url': self.entry.thumbnail_url,
            'thumbnail_library_relative': self.entry.thumbnail_library_relative,
            'installed': self.installed,
            'installed_path': self.installed_path,
            'installed_root_path': self.installed_root_path,
        }


@dataclass
class DownloadJobState:
    job_id: str
    selector: str
    entry_id: str
    status: str = 'queued'
    progress: float | None = None
    message: str = ''
    result_path: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            'job_id': self.job_id,
            'selector': self.selector,
            'entry_id': self.entry_id,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'result_path': self.result_path,
            'error': self.error,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'updated_at': self.updated_at,
        }


class DownloadJobRegistry:
    def __init__(self):
        self._jobs: dict[str, DownloadJobState] = {}
        self._lock = threading.RLock()

    def create_job(self, selector: str, entry: ModelCatalogEntry) -> DownloadJobState:
        job = DownloadJobState(job_id=uuid.uuid4().hex, selector=str(selector), entry_id=entry.id)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> DownloadJobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[DownloadJobState]:
        with self._lock:
            return list(self._jobs.values())

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        result_path: str | None = None,
        error: str | None = None,
    ) -> DownloadJobState:
        with self._lock:
            job = self._jobs[job_id]
            if status is not None:
                if status not in {'queued', 'running', 'succeeded', 'failed'}:
                    raise ValueError(f'Unknown job status: {status}')
                job.status = status
                if status == 'running' and job.started_at is None:
                    job.started_at = time.time()
                if status in {'succeeded', 'failed'}:
                    job.finished_at = time.time()
            if progress is not None:
                job.progress = max(0.0, min(1.0, float(progress)))
            if message is not None:
                job.message = message
            if result_path is not None:
                job.result_path = result_path
            if error is not None:
                job.error = error
            job.updated_at = time.time()
            return job

    def submit(
        self,
        selector: str,
        entry: ModelCatalogEntry,
        worker: Callable[[ModelCatalogEntry, Callable[[float | None, str | None], None]], Any],
    ) -> DownloadJobState:
        job = self.create_job(selector, entry)

        def report_progress(progress: float | None = None, message: str | None = None) -> None:
            self.update(job.job_id, status='running', progress=progress, message=message)

        def runner() -> None:
            self.update(job.job_id, status='running', progress=0.0, message='Download started')
            try:
                result = worker(entry, report_progress)
                success = _coerce_result_success(result)
                self.update(
                    job.job_id,
                    status='succeeded' if success else 'failed',
                    progress=1.0 if success else 0.0,
                    message=_coerce_result_message(result) or ('Download completed' if success else 'Download failed'),
                    result_path=_coerce_result_path(result),
                    error=None if success else (_coerce_result_message(result) or 'Download failed'),
                )
            except Exception as exc:
                self.update(job.job_id, status='failed', message=str(exc), error=str(exc))

        threading.Thread(target=runner, name=f'model-download-{job.job_id}', daemon=True).start()
        return job

    def wait_for(self, job_id: str, timeout: float | None = None) -> DownloadJobState | None:
        deadline = None if timeout is None else time.time() + timeout
        while True:
            job = self.get(job_id)
            if job is None:
                return None
            if job.status in {'succeeded', 'failed'}:
                return job
            if deadline is not None and time.time() >= deadline:
                return job
            time.sleep(0.01)


class ModelManager:
    def __init__(
        self,
        catalog_dirs: Iterable[str | os.PathLike[str]] | None = None,
        root_map: dict[str, Iterable[str | os.PathLike[str]]] | None = None,
        download_worker: Callable[[ModelCatalogEntry, Callable[[float | None, str | None], None]], Any] | None = None,
    ):
        self._catalog_dirs = [str(path) for path in catalog_dirs] if catalog_dirs is not None else None
        self._root_map = {
            key: _normalize_paths(paths)
            for key, paths in (root_map.items() if root_map is not None else _build_default_root_map().items())
        }
        self._catalog_index = None
        self._installed_index: dict[str, InstalledPathIndex] = {}
        self._index_lock = threading.RLock()
        self._download_worker = download_worker
        self.download_jobs = DownloadJobRegistry()

    @property
    def catalog_index(self):
        if self._catalog_index is None:
            self.refresh_catalog_index()
        return self._catalog_index

    @property
    def root_map(self) -> dict[str, list[str]]:
        return {key: list(paths) for key, paths in self._root_map.items()}

    def refresh_catalog_index(self):
        catalog_dirs = self._catalog_dirs
        if catalog_dirs is None:
            catalog_dirs = config.get_model_catalog_directories()
        self._catalog_index = catalog_index.load_runtime_model_catalog_index(catalog_dirs)
        return self._catalog_index

    def refresh_installed_index(self):
        installed_index: dict[str, InstalledPathIndex] = {}
        for roots in self._root_map.values():
            for root in roots:
                if not os.path.isdir(root):
                    continue
                rel_to_abs: dict[str, str] = {}
                basenames: dict[str, str] = {}
                for relative_path in get_files_from_folder(root, extensions=MODEL_FILE_EXTENSIONS):
                    normalized_relative = _normalize_path(relative_path)
                    if normalized_relative is None:
                        continue
                    absolute_path = os.path.abspath(os.path.join(root, relative_path))
                    rel_to_abs.setdefault(normalized_relative, absolute_path)
                    basenames.setdefault(os.path.basename(relative_path).lower(), absolute_path)
                installed_index[root] = InstalledPathIndex(root_path=root, relative_paths=rel_to_abs, basenames=basenames)
        with self._index_lock:
            self._installed_index = installed_index
        return installed_index

    def refresh(self):
        self.refresh_catalog_index()
        self.refresh_installed_index()
        return self

    def _ensure_catalog_index(self):
        if self._catalog_index is None:
            self.refresh_catalog_index()
        return self._catalog_index

    def _ensure_installed_index(self):
        with self._index_lock:
            if not self._installed_index:
                self._installed_index = self.refresh_installed_index()
            return dict(self._installed_index)

    def _find_catalog_record(self, selector: str, root_keys: Iterable[str] | None = None):
        index = self._ensure_catalog_index()
        normalized = _normalize_path(selector)
        if normalized is None:
            return None

        record = index.find_by_relative_path(normalized, root_keys=root_keys)
        if record is not None:
            return record

        record = index.find_by_name(os.path.basename(normalized), root_keys=root_keys)
        if record is not None:
            return record

        record = index.get_record(normalized)
        if record is not None and (root_keys is None or record.entry.root_key in set(root_keys)):
            return record

        return None

    def get_entry(self, selector: str, root_keys: Iterable[str] | None = None) -> ModelCatalogEntry | None:
        record = self._find_catalog_record(selector, root_keys=root_keys)
        return None if record is None else record.entry

    def _entry_installation(self, entry: ModelCatalogEntry) -> tuple[str | None, str | None]:
        roots = self._root_map.get(entry.root_key, [])
        if not roots:
            return None, None

        canonical_relative = _normalize_path(entry.relative_path)
        candidate_names: list[str] = []
        for candidate in (entry.name, entry.source_file_name, os.path.basename(entry.relative_path)):
            normalized = _normalize_path(candidate)
            if normalized and normalized not in candidate_names:
                candidate_names.append(normalized)

        installed_index = self._ensure_installed_index()
        for root in roots:
            root_index = installed_index.get(root)
            if root_index is None:
                continue
            if canonical_relative and canonical_relative in root_index.relative_paths:
                return root_index.relative_paths[canonical_relative], root
            for candidate_name in candidate_names:
                matched = root_index.basenames.get(os.path.basename(candidate_name).lower())
                if matched is not None:
                    return matched, root

        return None, None

    def inventory_record(self, entry: ModelCatalogEntry) -> ModelInventoryRecord:
        installed_path, installed_root = self._entry_installation(entry)
        return ModelInventoryRecord(
            entry=entry,
            installed=installed_path is not None,
            installed_path=installed_path,
            installed_root_path=installed_root,
        )

    def iter_inventory(
        self,
        *,
        architecture: str | None = None,
        sub_architecture: str | None = None,
        compatibility_family: str | None = None,
        model_type: str | None = None,
        root_key: str | None = None,
        storage_tier: str | None = None,
        visibility: str | None = None,
        preset_managed: bool | None = None,
        installed: bool | None = None,
    ) -> list[ModelInventoryRecord]:
        entries = self._ensure_catalog_index().filter(
            architecture=architecture,
            sub_architecture=sub_architecture,
            compatibility_family=compatibility_family,
            model_type=model_type,
            root_key=root_key,
            storage_tier=storage_tier,
            visibility=visibility,
        )
        records = [self.inventory_record(entry) for entry in entries]
        if preset_managed is not None:
            records = [record for record in records if record.entry.preset_managed is preset_managed]
        if installed is not None:
            records = [record for record in records if record.installed is installed]
        return records

    def list_installed(self, **filters) -> list[ModelInventoryRecord]:
        return self.iter_inventory(installed=True, **filters)

    def list_available(self, **filters) -> list[ModelInventoryRecord]:
        return self.iter_inventory(installed=False, **filters)

    def list_dropdown_entries(
        self,
        *,
        base_model_name: str | None = None,
        root_key: str | None = None,
        installed_only: bool = True,
        generic_only: bool = True,
        include_preset_managed: bool = False,
        architecture: str | None = None,
        sub_architecture: str | None = None,
    ) -> list[ModelInventoryRecord]:
        if base_model_name is not None and architecture is None:
            scope = self.get_filter_scope(base_model_name, root_key=root_key, model_type='lora' if root_key == 'loras' else None)
            architecture = scope['architecture']
            sub_architecture = scope['sub_architecture']

        records = self.iter_inventory(
            architecture=architecture,
            sub_architecture=sub_architecture,
            root_key=root_key,
            installed=installed_only if installed_only else None,
        )
        if generic_only:
            records = [
                record
                for record in records
                if record.entry.visibility == 'generic' and (include_preset_managed or not record.entry.preset_managed)
            ]
        elif not include_preset_managed:
            records = [record for record in records if not record.entry.preset_managed]
        return records

    def list_installed_lora_dropdown_choices(
        self,
        *,
        base_model_name: str | None = None,
        include_preset_managed: bool = False,
    ) -> list[str]:
        scope = self.get_filter_scope(base_model_name, root_key='loras', model_type='lora')
        installed_index = self._ensure_installed_index()
        choices: list[str] = []
        seen: set[str] = set()

        for root_key in ('loras', 'loras_lcm', 'loras_lightning'):
            roots = self._root_map.get(root_key, [])
            for root in roots:
                root_index = installed_index.get(root)
                if root_index is None:
                    continue
                for relative_path in sorted(root_index.relative_paths):
                    normalized_relative_path = _normalize_path(relative_path)
                    if normalized_relative_path is None or normalized_relative_path in seen:
                        continue

                    entry = self.get_entry(normalized_relative_path, root_keys=[root_key])
                    if entry is not None:
                        if entry.model_type != 'lora':
                            continue
                        if entry.visibility != 'generic' and not include_preset_managed:
                            continue
                        if entry.preset_managed and not include_preset_managed:
                            continue
                        candidate_architecture = entry.architecture
                        candidate_sub_architecture = entry.sub_architecture
                    else:
                        if root_key in {'loras_lcm', 'loras_lightning'} and not include_preset_managed:
                            continue
                        taxonomy = config.resolve_model_taxonomy(
                            normalized_relative_path,
                            root_keys=(root_key,),
                            folder_paths=self._root_map.get(root_key, []),
                        )
                        candidate_architecture = taxonomy.architecture
                        candidate_sub_architecture = taxonomy.sub_architecture

                    if not _matches_filter_scope(
                        candidate_architecture,
                        candidate_sub_architecture,
                        target_architecture=scope['architecture'],
                        target_sub_architecture=scope['sub_architecture'],
                        root_key='loras',
                        model_type='lora',
                    ):
                        continue

                    seen.add(normalized_relative_path)
                    choices.append(normalized_relative_path)

        return sorted(choices)

    def build_architecture_groups(self, **filters) -> list[dict[str, Any]]:
        entries = self.iter_inventory(**filters)
        buckets: dict[str, dict[str, Any]] = {}
        for record in entries:
            entry = record.entry
            bucket = buckets.setdefault(
                entry.architecture,
                {
                    'architecture': entry.architecture,
                    'compatibility_family': entry.compatibility_family,
                    'records': [],
                    'sub_architectures': {},
                    'installed_count': 0,
                    'available_count': 0,
                    'total_count': 0,
                },
            )
            bucket['records'].append(record.to_dict())
            bucket['total_count'] += 1
            if record.installed:
                bucket['installed_count'] += 1
            else:
                bucket['available_count'] += 1

            subgroup = bucket['sub_architectures'].setdefault(
                entry.sub_architecture,
                {
                    'sub_architecture': entry.sub_architecture,
                    'records': [],
                    'installed_count': 0,
                    'available_count': 0,
                    'total_count': 0,
                },
            )
            subgroup['records'].append(record.to_dict())
            subgroup['total_count'] += 1
            if record.installed:
                subgroup['installed_count'] += 1
            else:
                subgroup['available_count'] += 1
        return [buckets[key] for key in sorted(buckets)]

    def build_inventory_payload(self, **filters) -> dict[str, Any]:
        records = self.iter_inventory(**filters)
        return {
            'installed': [record.to_dict() for record in records if record.installed],
            'available': [record.to_dict() for record in records if not record.installed],
            'groups': self.build_architecture_groups(**filters),
        }

    def _resolve_taxonomy(self, selector: str):
        entry = self.get_entry(selector)
        if entry is not None:
            return model_taxonomy.build_resolved_model_taxonomy(
                architecture=entry.architecture,
                sub_architecture=entry.sub_architecture,
                compatibility_family=entry.compatibility_family,
                source='catalog',
                catalog_entry_id=entry.id,
            )
        try:
            return config.resolve_model_taxonomy(selector)
        except Exception:
            return model_taxonomy.build_resolved_model_taxonomy(source='default')

    def get_filter_scope(
        self,
        base_model_name: str | None,
        *,
        root_key: str | None = None,
        model_type: str | None = None,
    ) -> dict[str, Any]:
        if base_model_name is None:
            return {
                'architecture': None,
                'sub_architecture': None,
                'compatibility_family': None,
                'source': 'default',
                'catalog_entry_id': None,
            }

        taxonomy = self._resolve_taxonomy(base_model_name)
        return {
            'architecture': taxonomy.architecture,
            'sub_architecture': _default_filter_sub_architecture(
                taxonomy.architecture,
                taxonomy.sub_architecture,
                root_key=root_key,
                model_type=model_type,
            ),
            'compatibility_family': taxonomy.compatibility_family,
            'source': taxonomy.source,
            'catalog_entry_id': taxonomy.catalog_entry_id,
        }

    def start_download_job(
        self,
        selector: str,
        *,
        worker: Callable[[ModelCatalogEntry, Callable[[float | None, str | None], None]], Any] | None = None,
    ) -> DownloadJobState:
        entry = self.get_entry(selector)
        if entry is None:
            raise KeyError(f'Unknown model selector: {selector}')
        worker = worker or self._download_worker
        if worker is None:
            raise RuntimeError('No download worker has been configured')
        return self.download_jobs.submit(selector, entry, worker)

    def get_job(self, job_id: str) -> DownloadJobState | None:
        return self.download_jobs.get(job_id)

    def list_jobs(self) -> list[DownloadJobState]:
        return self.download_jobs.list_jobs()


default_model_manager = ModelManager()






