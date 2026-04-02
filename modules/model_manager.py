from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import modules.config as config
import modules.model_catalog_index as catalog_index
import modules.model_taxonomy as model_taxonomy
import modules.model_thumbnails as model_thumbnails
from modules.extra_utils import get_files_from_folder
from modules.model_download.spec import (
    REGISTRATION_STATE_LOCALLY_REGISTERED,
    REGISTRATION_STATE_SOURCED_REGISTERED,
    REGISTRATION_STATE_UNREGISTERED,
    ModelCatalogEntry,
    ModelSource,
)


MODEL_FILE_EXTENSIONS = [
    '.pth',
    '.ckpt',
    '.bin',
    '.safetensors',
    '.fooocus.patch',
    '.gguf',
    '.sft',
]

USER_FACING_DISCOVERY_ROOT_KEYS = (
    'checkpoints',
    'loras',
    'unet',
    'clip',
    'vae',
    'embeddings',
)
UNREGISTERED_INSTALL_CATALOG_FILENAME = 'unregistered_install_catalog.catalog.json'
UNREGISTERED_INSTALL_CATALOG_ID = 'user.unregistered.install'
UNREGISTERED_INSTALL_CATALOG_LABEL = 'Unregistered Installed Models'
INSTALLED_MODEL_LINKS_FILENAME = 'installed_model_links.json'
INSTALLED_MODEL_LINKS_ID = 'user.installed.links'
INSTALLED_MODEL_LINKS_LABEL = 'Installed Model Links'
LOCAL_REGISTERED_CATALOG_FILENAME = 'user_local_models.catalog.json'
LOCAL_REGISTERED_CATALOG_ID = 'user.local.models'
LOCAL_REGISTERED_CATALOG_LABEL = 'User Local Models'
AUTO_GENERATED_UNREGISTERED_TAGS = ('auto_generated', 'unregistered')


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


def _normalize_absolute_path(value: str | os.PathLike[str] | None) -> str | None:
    if value is None:
        return None
    normalized = os.path.abspath(os.path.realpath(str(value))).replace('\\', '/').strip()
    return normalized or None


def _normalize_lookup_key(value: str | os.PathLike[str] | None) -> str | None:
    normalized = _normalize_path(value)
    return None if normalized is None else normalized.lower()



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


def _default_model_type_for_root(root_key: str) -> str:
    return {
        'checkpoints': 'checkpoint',
        'loras': 'lora',
        'unet': 'unet',
        'clip': 'clip',
        'vae': 'vae',
        'embeddings': 'embedding',
    }.get(root_key, root_key)


def _normalize_generated_sub_architecture(root_key: str, architecture: str | None, sub_architecture: str | None) -> str | None:
    if root_key in {'vae', 'embeddings'}:
        return model_taxonomy.SUB_ARCHITECTURE_NONE

    filtered = _default_filter_sub_architecture(
        architecture,
        sub_architecture,
        root_key=root_key,
        model_type=_default_model_type_for_root(root_key),
    )
    if filtered is not None:
        return filtered

    if model_taxonomy.normalize_architecture(architecture) in {model_taxonomy.ARCHITECTURE_SD15, model_taxonomy.ARCHITECTURE_SDXL}:
        return model_taxonomy.SUB_ARCHITECTURE_BASE
    return None


def _build_unregistered_entry_id(root_key: str, relative_path: str) -> str:
    digest = hashlib.sha1(f'{root_key}:{relative_path}'.encode('utf-8')).hexdigest()[:12]
    return f'unregistered.{root_key}.{digest}'


def _normalize_match_name(value: str | None) -> str:
    if value is None:
        return ''
    normalized = Path(str(value)).stem.lower()
    normalized = re.sub(r'[\W_]+', ' ', normalized)
    normalized = re.sub(r'\b(sd15|sd 15|sdxl|xl|checkpoint|model|lora|vae|clip|embedding|embeddings|gguf)\b', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def _tokenize_match_name(value: str | None) -> set[str]:
    normalized = _normalize_match_name(value)
    if not normalized:
        return set()
    return {token for token in normalized.split(' ') if token}


def _score_match_candidate(
    query_entry: ModelCatalogEntry,
    candidate: ModelCatalogEntry,
    *,
    source_provider: str | None = None,
    source_version_id: str | None = None,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    query_tokens = _tokenize_match_name(query_entry.name)
    candidate_labels = [candidate.name, candidate.display_name, candidate.alias]
    candidate_norms = [_normalize_match_name(label) for label in candidate_labels if label]
    candidate_tokens = set()
    for label in candidate_labels:
        candidate_tokens.update(_tokenize_match_name(label))

    if source_provider and str(candidate.source_provider).lower() == str(source_provider).lower():
        score += 8.0
        reasons.append(f'provider:{candidate.source_provider}')

    if source_version_id and candidate.source_version_id and str(candidate.source_version_id) == str(source_version_id):
        score += 100.0
        reasons.append('version_id_exact')

    query_norm = _normalize_match_name(query_entry.name)
    best_ratio = 0.0
    exact_name = False
    for candidate_norm in candidate_norms:
        if not candidate_norm:
            continue
        if candidate_norm == query_norm and query_norm:
            exact_name = True
        best_ratio = max(best_ratio, SequenceMatcher(None, query_norm, candidate_norm).ratio())
    if exact_name:
        score += 35.0
        reasons.append('name_exact')
    if best_ratio > 0.0:
        score += best_ratio * 40.0
        reasons.append(f'name_similarity:{best_ratio:.2f}')

    if query_tokens and candidate_tokens:
        overlap = len(query_tokens & candidate_tokens) / max(len(query_tokens), len(candidate_tokens), 1)
        if overlap > 0:
            score += overlap * 25.0
            reasons.append(f'token_overlap:{overlap:.2f}')

    if query_entry.architecture and query_entry.architecture == candidate.architecture:
        score += 6.0
        reasons.append(f'architecture:{candidate.architecture}')
    if query_entry.sub_architecture and query_entry.sub_architecture == candidate.sub_architecture:
        score += 3.0
        reasons.append(f'sub_architecture:{candidate.sub_architecture}')

    return score, reasons




_UNET_COMPANION_QUANT_SUFFIX_RE = re.compile(r'_(q\d(?:_[a-z0-9]+)*)$', re.IGNORECASE)


def _derive_clip_name_from_unet_name(value: str | None) -> str:
    filename = os.path.basename(str(value or '').strip())
    if not filename:
        return 'paired_clips.safetensors'
    stem, _ = os.path.splitext(filename)
    cleaned = _UNET_COMPANION_QUANT_SUFFIX_RE.sub('', stem).strip(' _-')
    if not cleaned:
        cleaned = stem or 'paired'
    return f'{cleaned}_clips.safetensors'


def _derive_clip_display_name_from_unet_payload(payload: dict[str, Any]) -> str:
    source = str(payload.get('display_name') or payload.get('name') or '').strip()
    if not source:
        return 'paired clips'
    stem = Path(source).stem if os.path.splitext(source)[1] else source
    stem = re.sub(r'\bq\d(?:\s+[a-z0-9]+)*\b$', '', stem, flags=re.IGNORECASE).strip(' _-')
    stem = re.sub(r'[_\-]+', ' ', stem).strip()
    return f'{stem or "paired"} clips'

def _entry_to_payload(entry: ModelCatalogEntry) -> dict[str, Any]:
    payload = {
        'id': entry.id,
        'name': entry.name,
        'root_key': entry.root_key,
        'relative_path': entry.relative_path,
        'display_name': entry.display_name,
        'model_type': entry.model_type,
        'architecture': entry.architecture,
        'sub_architecture': entry.sub_architecture,
        'compatibility_family': entry.compatibility_family,
        'source_provider': entry.source_provider,
        'registration_state': entry.registration_state,
        'visibility': entry.visibility,
        'preset_managed': entry.preset_managed,
        'token_required': entry.token_required,
        'tags': list(entry.tags),
    }
    if entry.alias is not None:
        payload['alias'] = entry.alias
    if entry.asset_group_key is not None:
        payload['asset_group_key'] = entry.asset_group_key
    if entry.thumbnail_library_relative is not None:
        payload['thumbnail_library_relative'] = entry.thumbnail_library_relative
    if entry.source_version_id is not None:
        payload['source_version_id'] = entry.source_version_id
    if entry.source is not None:
        payload['source'] = {
            'url': entry.source.url,
            'token_env': entry.source.token_env,
            'headers': [list(header) for header in entry.source.headers],
        }
    return payload


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
class InstalledModelLink:
    install_id: str
    entry_id: str
    root_key: str
    installed_path: str
    installed_root_path: str | None = None
    installed_relative_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'install_id': self.install_id,
            'entry_id': self.entry_id,
            'root_key': self.root_key,
            'installed_path': self.installed_path,
            'installed_root_path': self.installed_root_path,
            'installed_relative_path': self.installed_relative_path,
        }


@dataclass(frozen=True)
class ModelInventoryRecord:
    entry: ModelCatalogEntry
    installed: bool
    installed_path: str | None = None
    installed_root_path: str | None = None
    installed_relative_path: str | None = None

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
            'source_version_id': self.entry.source_version_id,
            'registration_state': self.entry.registration_state,
            'visibility': self.entry.visibility,
            'preset_managed': self.entry.preset_managed,
            'token_required': self.entry.token_required,
            'thumbnail_library_relative': self.entry.thumbnail_library_relative,
            'installed': self.installed,
            'installed_path': self.installed_path,
            'installed_root_path': self.installed_root_path,
            'installed_relative_path': self.installed_relative_path,
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
        writable_catalog_dir: str | os.PathLike[str] | None = None,
    ):
        self._catalog_dirs = [str(path) for path in catalog_dirs] if catalog_dirs is not None else None
        self._root_map = {
            key: _normalize_paths(paths)
            for key, paths in (root_map.items() if root_map is not None else _build_default_root_map().items())
        }
        self._catalog_index = None
        self._writable_catalog_dir = str(writable_catalog_dir) if writable_catalog_dir is not None else (
            self._catalog_dirs[-1] if self._catalog_dirs else None
        )
        self._installed_index: dict[str, InstalledPathIndex] = {}
        self._installed_links: list[InstalledModelLink] = []
        self._installed_links_by_entry_id: dict[str, list[InstalledModelLink]] = {}
        self._installed_links_by_relative_path: dict[str, list[InstalledModelLink]] = {}
        self._installed_links_by_name: dict[str, list[InstalledModelLink]] = {}
        self._installed_links_by_path: dict[str, InstalledModelLink] = {}
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

    def refresh_catalog_index(self, *, force_refresh: bool = False):
        catalog_dirs = self._catalog_dirs
        if catalog_dirs is None:
            catalog_dirs = config.get_model_catalog_directories()
        self._catalog_index = catalog_index.load_runtime_model_catalog_index(catalog_dirs, force_refresh=force_refresh)
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
        self.refresh_installed_links()
        self.sync_unregistered_install_catalog()
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

    def _installed_links_path(self) -> Path:
        writable_dir = Path(self._get_writable_catalog_directory())
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
            for record in self._ensure_catalog_index().list_records()
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

        with self._index_lock:
            self._installed_links = links
            self._installed_links_by_entry_id = links_by_entry_id
            self._installed_links_by_relative_path = links_by_relative_path
            self._installed_links_by_name = links_by_name
            self._installed_links_by_path = links_by_path
        if payload_changed:
            payload['links'] = normalized_links_payload
            self._write_installed_links_payload(payload)
        return links

    def _ensure_installed_links(self):
        with self._index_lock:
            if not self._installed_links_by_entry_id and not self._installed_links:
                self.refresh_installed_links()
            return list(self._installed_links)

    def _find_installed_link_for_entry(self, entry_id: str) -> InstalledModelLink | None:
        self._ensure_installed_links()
        links = self._installed_links_by_entry_id.get(str(entry_id), [])
        for link in links:
            if os.path.exists(link.installed_path):
                return link
        return links[0] if links else None

    def _ensure_persisted_installed_link(self, entry: ModelCatalogEntry, inventory_record: ModelInventoryRecord | None = None) -> InstalledModelLink | None:
        inventory_record = inventory_record or self.inventory_record(entry)
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
        candidates.extend(self._installed_links_by_relative_path.get(normalized_selector, []))
        candidates.extend(self._installed_links_by_name.get(os.path.basename(normalized_selector).lower(), []))

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
        roots = list(self._root_map.get(root_key, []))
        if root_key != 'checkpoints':
            return roots

        overlapping_roots = set()
        for other_root_key in USER_FACING_DISCOVERY_ROOT_KEYS:
            if other_root_key == root_key:
                continue
            overlapping_roots.update(self._root_map.get(other_root_key, []))
        return [root for root in roots if root not in overlapping_roots]

    @staticmethod
    def _is_auto_generated_unregistered_record(record) -> bool:
        if record is None:
            return False
        tags = {str(tag) for tag in record.entry.tags}
        return (
            record.entry.registration_state == REGISTRATION_STATE_UNREGISTERED
            and 'auto_generated' in tags
        )

    def _preferred_record(self, records):
        records = list(records)
        if not records:
            return None
        for record in records:
            if not self._is_auto_generated_unregistered_record(record):
                return record
        return records[0]

    def _find_catalog_record(self, selector: str, root_keys: Iterable[str] | None = None):
        index = self._ensure_catalog_index()
        normalized = _normalize_path(selector)
        if normalized is None:
            return None

        relative_record = self._preferred_record(index.list_by_relative_path(normalized, root_keys=root_keys))
        if relative_record is not None and not self._is_auto_generated_unregistered_record(relative_record):
            return relative_record

        name_record = self._preferred_record(index.list_by_name(os.path.basename(normalized), root_keys=root_keys))
        if name_record is not None and not self._is_auto_generated_unregistered_record(name_record):
            return name_record

        if relative_record is not None:
            return relative_record
        if name_record is not None:
            return name_record

        record = index.get_record(normalized)
        if record is not None and (root_keys is None or record.entry.root_key in set(root_keys)):
            return record

        installed_link = self._find_installed_link_by_selector(normalized, root_keys=root_keys)
        if installed_link is not None:
            linked_record = index.get_record(installed_link.entry_id)
            if linked_record is not None:
                return linked_record

        return None

    def get_entry(self, selector: str, root_keys: Iterable[str] | None = None) -> ModelCatalogEntry | None:
        record = self._find_catalog_record(selector, root_keys=root_keys)
        return None if record is None else record.entry

    def _catalog_path(self, source_path: str) -> Path:
        return Path(source_path).resolve()

    def _load_catalog_payload(self, source_path: str) -> dict[str, Any]:
        path = self._catalog_path(source_path)
        return json.loads(path.read_text(encoding='utf-8-sig'))

    def _write_catalog_payload(self, source_path: str, payload: dict[str, Any]) -> None:
        path = self._catalog_path(source_path)
        path.write_text(json.dumps(payload, indent=4, ensure_ascii=False) + '\n', encoding='utf-8')

    def _iter_payload_entries(self, node: Any):
        if isinstance(node, list):
            for item in node:
                yield from self._iter_payload_entries(item)
        elif isinstance(node, dict):
            if 'id' in node and 'name' in node and 'root_key' in node:
                yield node
            else:
                for value in node.values():
                    yield from self._iter_payload_entries(value)


    def _get_writable_catalog_directory(self) -> str:
        return self._writable_catalog_dir or config.get_writable_model_catalog_directory()

    def _unregistered_catalog_path(self) -> Path:
        writable_dir = Path(self._get_writable_catalog_directory())
        writable_dir.mkdir(parents=True, exist_ok=True)
        return writable_dir / UNREGISTERED_INSTALL_CATALOG_FILENAME

    def _build_unregistered_catalog_payload(self, entries: list[ModelCatalogEntry]) -> dict[str, Any]:
        path = self._unregistered_catalog_path()
        if path.exists():
            payload = self._load_catalog_payload(str(path))
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

    def _replace_catalog_entry_payload(self, source_path: str, entry_id: str, entry_payload: dict[str, Any]) -> None:
        payload = self._load_catalog_payload(source_path)
        updated = False
        for existing_entry in self._iter_payload_entries(payload):
            if existing_entry.get('id') != entry_id:
                continue
            existing_entry.clear()
            existing_entry.update(entry_payload)
            updated = True
            break

        if not updated:
            payload.setdefault('entries', []).append(entry_payload)

        self._write_catalog_payload(source_path, payload)

    def _remove_catalog_entry_payload(self, source_path: str, entry_id: str) -> bool:
        payload = self._load_catalog_payload(source_path)
        entries = payload.get('entries', [])
        if not isinstance(entries, list):
            return False
        filtered_entries = [entry for entry in entries if not (isinstance(entry, dict) and entry.get('id') == entry_id)]
        if len(filtered_entries) == len(entries):
            return False
        payload['entries'] = filtered_entries
        self._write_catalog_payload(source_path, payload)
        return True

    def suggest_catalog_matches(
        self,
        selector: str,
        *,
        limit: int = 3,
        source_provider: str | None = None,
        source_version_id: str | None = None,
    ) -> list[dict[str, Any]]:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')

        return self._suggest_catalog_matches_for_entry(
            record.entry,
            limit=limit,
            source_provider=source_provider,
            source_version_id=source_version_id,
            exclude_entry_ids=[record.entry.id],
        )

    def _suggest_catalog_matches_for_entry(
        self,
        query_entry: ModelCatalogEntry,
        *,
        limit: int = 3,
        source_provider: str | None = None,
        source_version_id: str | None = None,
        exclude_entry_ids: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        excluded = {str(value) for value in (exclude_entry_ids or []) if value}
        candidates: list[dict[str, Any]] = []
        for candidate_record in self._ensure_catalog_index().list_records():
            candidate_entry = candidate_record.entry
            if candidate_entry.id in excluded:
                continue
            if candidate_entry.root_key != query_entry.root_key:
                continue
            if self._is_auto_generated_unregistered_record(candidate_record):
                continue
            if candidate_entry.registration_state == REGISTRATION_STATE_UNREGISTERED:
                continue

            score, reasons = _score_match_candidate(
                query_entry,
                candidate_entry,
                source_provider=source_provider,
                source_version_id=source_version_id,
            )
            if score < 8.0 and 'version_id_exact' not in reasons:
                continue

            candidates.append({
                'score': round(score, 2),
                'reasons': reasons,
                'entry': self.inventory_record(candidate_entry).to_dict(),
            })

        candidates.sort(key=lambda item: (-item['score'], item['entry']['display_name'] or item['entry']['name']))
        return candidates[: max(1, int(limit))]

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
        for candidate_record in self._ensure_catalog_index().list_records():
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
        for record in self.iter_inventory(root_key='clip', installed=True):
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
        entry = selector_or_entry if isinstance(selector_or_entry, ModelCatalogEntry) else self.get_entry(str(selector_or_entry))
        if entry is None or entry.root_key != 'unet':
            return None

        target_clip_entry = self._find_companion_clip_catalog_entry(entry)
        if not installed_only:
            return target_clip_entry

        if target_clip_entry is not None:
            record = self.inventory_record(target_clip_entry)
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
        return self.get_entry(str(candidate_id)) if candidate_id else None

    def _build_unet_companion_clip_context(
        self,
        entry: ModelCatalogEntry,
        *,
        matched_selector: str | None = None,
        suggestions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        matched_entry = self.get_entry(matched_selector) if matched_selector else None
        if matched_entry is None and suggestions:
            first_id = suggestions[0].get('entry', {}).get('id')
            if first_id:
                matched_entry = self.get_entry(str(first_id))

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
            'matched_catalog_entry': self.inventory_record(target_clip_entry).to_dict() if target_clip_entry is not None else None,
            'installed_candidates': installed_candidates,
            'recommended_selector': recommended_selector,
            'needs_user_path': not bool(installed_candidates),
        }

    def build_registration_context(
        self,
        selector: str,
        *,
        suggest_limit: int = 3,
        source_provider: str | None = None,
        source_version_id: str | None = None,
        matched_selector: str | None = None,
    ) -> dict[str, Any]:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')

        entry_record = self.inventory_record(record.entry)
        suggestions = self.suggest_catalog_matches(
            selector,
            limit=suggest_limit,
            source_provider=source_provider,
            source_version_id=source_version_id,
        )
        thumbnail = model_thumbnails.resolve_thumbnail(record.entry)
        context = {
            'entry': entry_record.to_dict(),
            'source_path': record.source_path,
            'suggestions': suggestions,
            'editable': True,
            'mode': 'registration',
            'thumbnail': {
                'relative_path': thumbnail.relative_path,
                'absolute_path': thumbnail.absolute_path,
                'exists': thumbnail.exists,
                'source': thumbnail.source,
            },
        }
        if record.entry.root_key == 'unet':
            context['companion_clip'] = self._build_unet_companion_clip_context(
                record.entry,
                matched_selector=matched_selector,
                suggestions=suggestions,
            )
        return context

    def build_installed_link_context(
        self,
        selector: str,
        *,
        suggest_limit: int = 3,
    ) -> dict[str, Any]:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')

        inventory_record = self.inventory_record(record.entry)
        if not inventory_record.installed:
            raise ValueError('The selected model is not currently installed.')

        installed_link = self._find_installed_link_by_selector(selector, root_keys=[record.entry.root_key])
        if installed_link is None:
            installed_link = self._find_installed_link_for_entry(record.entry.id)
        if installed_link is None:
            installed_link = self._ensure_persisted_installed_link(record.entry, inventory_record)
        if installed_link is None:
            raise ValueError('The selected model does not have an editable installed link yet.')

        installed_relative_path = installed_link.installed_relative_path or inventory_record.installed_relative_path or record.entry.relative_path
        query_entry = ModelCatalogEntry(
            id=f'query.{record.entry.root_key}.{hashlib.sha1(str(installed_relative_path).encode("utf-8")).hexdigest()[:12]}',
            name=os.path.basename(installed_relative_path) or record.entry.name,
            root_key=record.entry.root_key,
            relative_path=installed_relative_path,
            display_name=inventory_record.entry.display_name,
            model_type=record.entry.model_type,
            architecture=record.entry.architecture,
            sub_architecture=record.entry.sub_architecture,
            compatibility_family=record.entry.compatibility_family,
            source_provider=record.entry.source_provider,
            source_version_id=record.entry.source_version_id,
            registration_state=record.entry.registration_state,
            visibility=record.entry.visibility,
            preset_managed=record.entry.preset_managed,
            token_required=record.entry.token_required,
            tags=record.entry.tags,
            asset_group_key=record.entry.asset_group_key,
            thumbnail_library_relative=record.entry.thumbnail_library_relative,
            source=record.entry.source,
        )
        suggestions = self._suggest_catalog_matches_for_entry(
            query_entry,
            limit=suggest_limit,
            source_provider=record.entry.source_provider,
            source_version_id=record.entry.source_version_id,
            exclude_entry_ids=[record.entry.id],
        )
        thumbnail = model_thumbnails.resolve_thumbnail(record.entry)
        return {
            'entry': inventory_record.to_dict(),
            'installed_link': installed_link.to_dict(),
            'source_path': record.source_path,
            'suggestions': suggestions,
            'editable': True,
            'mode': 'installed_link',
            'can_edit_catalog_fields': str(record.entry.id).startswith('user.local.'),
            'thumbnail': {
                'relative_path': thumbnail.relative_path,
                'absolute_path': thumbnail.absolute_path,
                'exists': thumbnail.exists,
                'source': thumbnail.source,
            },
        }

    def _prepare_registered_payload(
        self,
        source_entry: ModelCatalogEntry,
        *,
        matched_selector: str | None = None,
        updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = _entry_to_payload(source_entry)

        if matched_selector:
            matched_entry = self.get_entry(matched_selector)
            if matched_entry is None:
                raise KeyError(f'Unknown matched selector: {matched_selector}')
            if matched_entry.root_key != payload['root_key']:
                raise ValueError('Matched catalog entry must share the same root_key.')
            matched_payload = _entry_to_payload(matched_entry)
            for key in (
                'name',
                'display_name',
                'model_type',
                'architecture',
                'sub_architecture',
                'compatibility_family',
                'asset_group_key',
                'thumbnail_library_relative',
                'source_provider',
                'source_version_id',
                'visibility',
                'token_required',
            ):
                if matched_payload.get(key) is not None:
                    payload[key] = matched_payload[key]
            if matched_payload.get('source') is not None:
                payload['source'] = matched_payload['source']

        updates = dict(updates or {})
        if 'source_url' in updates:
            source_url = str(updates.pop('source_url') or '').strip()
            if source_url:
                payload['source'] = {'url': source_url}
            else:
                payload.pop('source', None)

        for key in (
            'alias',
            'display_name',
            'name',
            'relative_path',
            'model_type',
            'architecture',
            'sub_architecture',
            'compatibility_family',
            'asset_group_key',
            'thumbnail_library_relative',
            'source_provider',
            'source_version_id',
            'visibility',
        ):
            if key in updates and updates[key] is not None:
                payload[key] = updates[key]

        if 'token_env' in updates:
            token_env = updates['token_env']
            source_payload = dict(payload.get('source') or {})
            if source_payload or token_env:
                if token_env:
                    source_payload['token_env'] = str(token_env)
                payload['source'] = source_payload

        payload['architecture'] = model_taxonomy.normalize_architecture(payload.get('architecture')) or 'unknown'
        if payload.get('root_key') in {'vae', 'embeddings'} or payload.get('model_type') in {'vae', 'embedding'}:
            payload['sub_architecture'] = model_taxonomy.SUB_ARCHITECTURE_NONE
        else:
            normalized_sub_architecture = model_taxonomy.normalize_sub_architecture(
                payload.get('sub_architecture', 'general'),
                architecture=payload['architecture'],
            )
            payload['sub_architecture'] = normalized_sub_architecture or 'general'

        payload['compatibility_family'] = updates.get('compatibility_family') or model_taxonomy.get_compatibility_family(
            architecture=payload['architecture'],
            sub_architecture=payload['sub_architecture'],
            model_type=payload.get('model_type'),
        )

        provider = str(payload.get('source_provider') or 'local').strip().lower()
        source_payload = payload.get('source') if isinstance(payload.get('source'), dict) else None
        has_source_url = bool(source_payload and str(source_payload.get('url') or '').strip())
        payload['registration_state'] = (
            REGISTRATION_STATE_SOURCED_REGISTERED if has_source_url else REGISTRATION_STATE_LOCALLY_REGISTERED
        )
        if not has_source_url:
            payload.pop('source', None)
        if not provider:
            payload['source_provider'] = 'local'

        tags = [str(tag) for tag in payload.get('tags', []) if str(tag) not in {'auto_generated', 'unregistered'}]
        payload['tags'] = tags
        payload['preset_managed'] = False
        payload['token_required'] = bool(payload.get('token_required', False))
        return payload

    def _local_registered_catalog_path(self) -> Path:
        writable_dir = Path(self._get_writable_catalog_directory())
        writable_dir.mkdir(parents=True, exist_ok=True)
        return writable_dir / LOCAL_REGISTERED_CATALOG_FILENAME

    def _load_local_registered_catalog_payload(self) -> dict[str, Any]:
        path = self._local_registered_catalog_path()
        if not path.exists():
            return {
                'catalog_id': LOCAL_REGISTERED_CATALOG_ID,
                'catalog_label': LOCAL_REGISTERED_CATALOG_LABEL,
                'entries': [],
            }
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault('catalog_id', LOCAL_REGISTERED_CATALOG_ID)
        payload.setdefault('catalog_label', LOCAL_REGISTERED_CATALOG_LABEL)
        if not isinstance(payload.get('entries'), list):
            payload['entries'] = []
        return payload

    def _upsert_local_registered_entry(
        self,
        source_entry: ModelCatalogEntry,
        *,
        updates: dict[str, Any] | None = None,
        existing_entry_id: str | None = None,
    ) -> ModelCatalogEntry:
        payload = self._prepare_registered_payload(source_entry, updates=updates)
        stable_relative = _normalize_path(payload.get('relative_path') or source_entry.relative_path) or source_entry.relative_path
        digest = hashlib.sha1(f"{source_entry.root_key}:{stable_relative}".encode('utf-8')).hexdigest()[:12]
        payload['id'] = existing_entry_id or f'user.local.{source_entry.root_key}.{digest}'

        catalog_payload = self._load_local_registered_catalog_payload()
        entries = [entry for entry in catalog_payload.get('entries', []) if isinstance(entry, dict)]
        updated = False
        for index, entry_data in enumerate(entries):
            if entry_data.get('id') != payload['id']:
                continue
            entries[index] = payload
            updated = True
            break
        if not updated:
            entries.append(payload)
        catalog_payload['entries'] = entries
        self._local_registered_catalog_path().write_text(json.dumps(catalog_payload, indent=4, ensure_ascii=False) + '\n', encoding='utf-8')
        self.refresh_catalog_index(force_refresh=True)
        entry = self.get_entry(payload['id'])
        if entry is None:
            raise RuntimeError(f"Local registered entry {payload['id']} disappeared after write")
        return entry

    def _register_single_model_entry(
        self,
        selector: str,
        *,
        matched_selector: str | None = None,
        updates: dict[str, Any] | None = None,
    ) -> ModelCatalogEntry:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')
        payload = self._prepare_registered_payload(record.entry, matched_selector=matched_selector, updates=updates)
        self._replace_catalog_entry_payload(record.source_path, record.entry.id, payload)
        self.refresh_catalog_index(force_refresh=True)
        refreshed = self.get_entry(record.entry.id)
        if refreshed is None:
            raise RuntimeError(f'Catalog entry {record.entry.id} disappeared after registration update')
        return refreshed

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

        target_clip_entry = self._find_companion_clip_catalog_entry(self.get_entry(matched_selector) if matched_selector else entry)
        companion_record = None
        if companion_selector:
            companion_record = self._find_catalog_record(companion_selector, root_keys=['clip'])
            if companion_record is None:
                raise KeyError(f'Unknown companion clip selector: {companion_selector}')
        elif companion_relative_path:
            companion_record = self._find_catalog_record(companion_relative_path, root_keys=['clip'])
        else:
            suggestions = self._suggest_installed_companion_clips(
                target_clip_entry=target_clip_entry,
                query_unet_entry=entry,
                limit=1,
            )
            if suggestions:
                companion_record = self._find_catalog_record(suggestions[0]['entry']['id'], root_keys=['clip'])

        if companion_record is None:
            return {
                'status': 'missing',
                'expected_name': target_clip_entry.name if target_clip_entry is not None else _derive_clip_name_from_unet_name(entry.name),
                'matched_catalog_entry': self.inventory_record(target_clip_entry).to_dict() if target_clip_entry is not None else None,
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

        companion_entry = self._register_single_model_entry(
            companion_record.entry.id,
            matched_selector=companion_matched_selector,
            updates=companion_updates,
        )
        return {
            'status': 'registered',
            'entry': self.inventory_record(companion_entry).to_dict(),
            'matched_catalog_entry': self.inventory_record(target_clip_entry).to_dict() if target_clip_entry is not None else None,
            'needs_user_path': False,
        }

    def register_model_entry_bundle(
        self,
        selector: str,
        *,
        matched_selector: str | None = None,
        updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')

        inventory_record = self.inventory_record(record.entry)
        if not inventory_record.installed or not inventory_record.installed_path:
            raise ValueError('The selected model is not currently installed.')

        update_payload = dict(updates or {})
        existing_entry_id = record.entry.id if str(record.entry.id).startswith('user.local.') else None

        if matched_selector:
            matched_entry = self.get_entry(matched_selector)
            if matched_entry is None:
                raise KeyError(f'Unknown matched selector: {matched_selector}')
            if matched_entry.root_key != record.entry.root_key:
                raise ValueError('Matched catalog entry must share the same root_key.')
            entry = matched_entry
        else:
            entry = self._upsert_local_registered_entry(
                record.entry,
                updates=update_payload,
                existing_entry_id=existing_entry_id,
            )

        installed_link = self._upsert_installed_link(
            entry_id=entry.id,
            root_key=record.entry.root_key,
            installed_path=inventory_record.installed_path,
            installed_root_path=inventory_record.installed_root_path,
            installed_relative_path=inventory_record.installed_relative_path or record.entry.relative_path,
        )

        if record.entry.registration_state == REGISTRATION_STATE_UNREGISTERED or self._is_auto_generated_unregistered_record(record):
            self._remove_catalog_entry_payload(record.source_path, record.entry.id)
            self.refresh_catalog_index(force_refresh=True)

        return {
            'entry': entry,
            'installed_link': installed_link.to_dict(),
        }

    def update_installed_model_link_bundle(
        self,
        selector: str,
        *,
        matched_selector: str | None = None,
        updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')

        inventory_record = self.inventory_record(record.entry)
        if not inventory_record.installed:
            raise ValueError('The selected model is not currently installed.')

        current_link = self._find_installed_link_by_selector(selector, root_keys=[record.entry.root_key])
        if current_link is None:
            current_link = self._find_installed_link_for_entry(record.entry.id)
        if current_link is None and not inventory_record.installed_path:
            raise ValueError('The selected model does not have an editable installed link yet.')

        update_payload = dict(updates or {})
        installed_relative_path = _normalize_path(
            update_payload.pop('installed_relative_path', None)
            or (current_link.installed_relative_path if current_link is not None else None)
            or inventory_record.installed_relative_path
        )
        installed_root_path = inventory_record.installed_root_path or (current_link.installed_root_path if current_link is not None else None)
        installed_path = inventory_record.installed_path or (current_link.installed_path if current_link is not None else None)

        if installed_root_path and installed_relative_path:
            candidate_installed_path = os.path.abspath(os.path.join(installed_root_path, installed_relative_path))
            if not os.path.exists(candidate_installed_path):
                raise ValueError(f'Installed path not found: {installed_relative_path}')
            installed_path = candidate_installed_path

        if not installed_path:
            raise ValueError('Unable to resolve the installed path for this model.')

        catalog_updates = {
            key: value
            for key, value in update_payload.items()
            if key in {
                'alias',
                'display_name',
                'name',
                'relative_path',
                'model_type',
                'architecture',
                'sub_architecture',
                'compatibility_family',
                'asset_group_key',
                'thumbnail_library_relative',
                'source_provider',
                'source_version_id',
                'visibility',
                'source_url',
                'token_env',
            }
        }

        if matched_selector:
            target_entry = self.get_entry(matched_selector)
            if target_entry is None:
                raise KeyError(f'Unknown matched selector: {matched_selector}')
            if target_entry.root_key != record.entry.root_key:
                raise ValueError('Matched catalog entry must share the same root_key.')
        elif str(record.entry.id).startswith('user.local.'):
            target_entry = self._upsert_local_registered_entry(
                record.entry,
                updates=catalog_updates,
                existing_entry_id=record.entry.id,
            ) if catalog_updates else record.entry
        elif catalog_updates:
            target_entry = self._upsert_local_registered_entry(
                record.entry,
                updates=catalog_updates,
            )
        else:
            target_entry = record.entry

        installed_link = self._upsert_installed_link(
            entry_id=target_entry.id,
            root_key=record.entry.root_key,
            installed_path=installed_path,
            installed_root_path=installed_root_path,
            installed_relative_path=installed_relative_path,
        )

        return {
            'entry': target_entry,
            'installed_link': installed_link.to_dict(),
        }

    def register_model_entry(
        self,
        selector: str,
        *,
        matched_selector: str | None = None,
        updates: dict[str, Any] | None = None,
    ) -> ModelCatalogEntry:
        return self.register_model_entry_bundle(
            selector,
            matched_selector=matched_selector,
            updates=updates,
        )['entry']

    def _build_unregistered_entry(self, root_key: str, relative_path: str) -> ModelCatalogEntry:
        normalized_relative_path = _normalize_path(relative_path)
        if normalized_relative_path is None:
            raise ValueError('relative_path is required for unregistered discovery')

        taxonomy = config.resolve_model_taxonomy(
            normalized_relative_path,
            root_keys=(root_key,),
            folder_paths=self._root_map.get(root_key, []),
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
        installed_index = self._ensure_installed_index()
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
                    existing_record = self._find_catalog_record(normalized_relative_path, root_keys=[root_key])
                    if existing_record is not None and not self._is_auto_generated_unregistered_record(existing_record):
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
            self.refresh_catalog_index(force_refresh=True)
        return {
            'path': str(path),
            'entry_count': len(entries),
            'changed': changed,
        }

    def update_catalog_entry_thumbnail_path(self, selector: str, thumbnail_library_relative: str) -> ModelCatalogEntry:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')

        normalized_relative = str(thumbnail_library_relative).replace('\\', '/').strip().lstrip('/')
        payload = self._load_catalog_payload(record.source_path)
        updated = False
        for entry_data in self._iter_payload_entries(payload):
            if entry_data.get('id') != record.entry.id:
                continue
            entry_data['thumbnail_library_relative'] = normalized_relative
            updated = True
            break

        if not updated:
            raise KeyError(f'Catalog entry {record.entry.id} not found in {record.source_path}')

        self._write_catalog_payload(record.source_path, payload)
        self.refresh_catalog_index(force_refresh=True)
        refreshed_entry = self.get_entry(record.entry.id)
        if refreshed_entry is None:
            raise RuntimeError(f'Catalog entry {record.entry.id} disappeared after catalog refresh')
        return refreshed_entry

    def persist_entry_thumbnail(
        self,
        selector: str,
        source: str | os.PathLike[str] | Any,
        *,
        slug: str | None = None,
        size: int | None = None,
    ) -> tuple[ModelCatalogEntry, model_thumbnails.ThumbnailResolution]:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')

        resolution = model_thumbnails.persist_thumbnail_image(
            source,
            entry=record.entry,
            slug=slug,
            size=size,
        )
        refreshed_entry = self.update_catalog_entry_thumbnail_path(record.entry.id, resolution.relative_path)
        return refreshed_entry, resolution

    def resolve_entry_thumbnail(self, selector: str, *, slug: str | None = None) -> tuple[ModelCatalogEntry, model_thumbnails.ThumbnailResolution]:
        record = self._find_catalog_record(selector)
        if record is None:
            raise KeyError(f'Unknown model selector: {selector}')
        return record.entry, model_thumbnails.resolve_thumbnail(record.entry, slug=slug)

    def _entry_installation(self, entry: ModelCatalogEntry) -> tuple[str | None, str | None, str | None]:
        installed_link = self._find_installed_link_for_entry(entry.id)
        if installed_link is not None and os.path.exists(installed_link.installed_path):
            return installed_link.installed_path, installed_link.installed_root_path, installed_link.installed_relative_path

        roots = self._root_map.get(entry.root_key, [])
        if not roots:
            return None, None, None

        canonical_relative = _normalize_path(entry.relative_path)
        candidate_names: list[str] = []
        for candidate in (entry.name, os.path.basename(entry.relative_path)):
            normalized = _normalize_path(candidate)
            if normalized and normalized not in candidate_names:
                candidate_names.append(normalized)

        installed_index = self._ensure_installed_index()
        for root in roots:
            root_index = installed_index.get(root)
            if root_index is None:
                continue
            if canonical_relative and canonical_relative in root_index.relative_paths:
                return root_index.relative_paths[canonical_relative], root, canonical_relative
            for candidate_name in candidate_names:
                matched = root_index.basenames.get(os.path.basename(candidate_name).lower())
                if matched is not None:
                    matched_relative = _normalize_path(os.path.relpath(matched, root).replace('\\', '/'))
                    return matched, root, matched_relative

        return None, None, None

    def inventory_record(self, entry: ModelCatalogEntry) -> ModelInventoryRecord:
        installed_path, installed_root, installed_relative = self._entry_installation(entry)
        return ModelInventoryRecord(
            entry=entry,
            installed=installed_path is not None,
            installed_path=installed_path,
            installed_root_path=installed_root,
            installed_relative_path=installed_relative,
        )

    def iter_inventory(
        self,
        *,
        architecture: str | None = None,
        sub_architecture: str | None = None,
        compatibility_family: str | None = None,
        model_type: str | None = None,
        root_key: str | None = None,
        registration_state: str | None = None,
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
            registration_state=registration_state,
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













