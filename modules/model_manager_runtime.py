from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from modules.model_download.spec import ModelCatalogEntry
from modules.model_manager_helpers import (
    _coerce_result_message,
    _coerce_result_path,
    _coerce_result_success,
)

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



