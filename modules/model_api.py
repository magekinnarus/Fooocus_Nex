from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import JSONResponse

from modules.model_download.policy import ModelDownloadPolicy
from modules.model_download.resolver import CivitAIResolver, DirectResolver, HuggingFaceResolver
from modules.model_download.transport import Aria2Transport
from modules.model_manager import ModelManager, default_model_manager


DEFAULT_DOWNLOAD_MESSAGE = 'Download queued'


def _select_resolver(entry):
    provider = (entry.source_provider or entry.source_kind or 'direct').lower()
    if provider == 'civitai':
        return CivitAIResolver(token_env=entry.token_env or 'CIVITAI_TOKEN')
    if provider in {'huggingface', 'hf'}:
        return HuggingFaceResolver(token_env=entry.token_env or 'HUGGINGFACE_TOKEN')
    return DirectResolver()


def _execute_download(entry, manager: ModelManager, report_progress):
    policy = ModelDownloadPolicy(root_map=manager.root_map)
    resolver = _select_resolver(entry)
    report_progress(0.1, 'Resolving download plan')
    plan = resolver.resolve(entry, policy)
    report_progress(0.3, f'Downloading {entry.name}')
    result = Aria2Transport().download(plan)
    if result.success:
        manager.refresh_installed_index()
    return result


def _build_job_worker(manager: ModelManager, download_worker=None):
    if download_worker is None:
        def download_worker(entry, report_progress):
            return _execute_download(entry, manager, report_progress)

    def worker(entry, report_progress):
        result = download_worker(entry, report_progress)
        if getattr(result, 'success', False):
            manager.refresh_installed_index()
        return result

    return worker


def _serialize_records(records):
    return [record.to_dict() for record in records]


def _catalog_sources_payload(manager: ModelManager) -> list[dict[str, Any]]:
    sources = []
    for source in manager.catalog_index.list_sources():
        sources.append({
            'path': source.path,
            'catalog_id': source.catalog_id,
            'catalog_label': source.catalog_label,
            'entry_count': source.entry_count,
        })
    return sources


def create_model_router(manager: ModelManager | None = None, download_worker=None) -> APIRouter:
    manager = manager or default_model_manager
    router = APIRouter()

    @router.get('/api/models/catalog')
    def list_catalog(
        architecture: str | None = Query(default=None),
        sub_architecture: str | None = Query(default=None),
        compatibility_family: str | None = Query(default=None),
        model_type: str | None = Query(default=None),
        root_key: str | None = Query(default=None),
        storage_tier: str | None = Query(default=None),
        visibility: str | None = Query(default=None),
        preset_managed: bool | None = Query(default=None),
        installed: bool | None = Query(default=None),
    ):
        records = manager.iter_inventory(
            architecture=architecture,
            sub_architecture=sub_architecture,
            compatibility_family=compatibility_family,
            model_type=model_type,
            root_key=root_key,
            storage_tier=storage_tier,
            visibility=visibility,
            preset_managed=preset_managed,
            installed=installed,
        )
        return JSONResponse(content={
            'entries': _serialize_records(records),
            'groups': manager.build_architecture_groups(
                architecture=architecture,
                sub_architecture=sub_architecture,
                compatibility_family=compatibility_family,
                model_type=model_type,
                root_key=root_key,
                storage_tier=storage_tier,
                visibility=visibility,
                preset_managed=preset_managed,
                installed=installed,
            ),
            'sources': _catalog_sources_payload(manager),
            'count': len(records),
        })

    @router.get('/api/models/installed')
    def list_installed(
        architecture: str | None = Query(default=None),
        sub_architecture: str | None = Query(default=None),
        compatibility_family: str | None = Query(default=None),
        model_type: str | None = Query(default=None),
        root_key: str | None = Query(default=None),
        preset_managed: bool | None = Query(default=None),
    ):
        records = manager.list_installed(
            architecture=architecture,
            sub_architecture=sub_architecture,
            compatibility_family=compatibility_family,
            model_type=model_type,
            root_key=root_key,
            preset_managed=preset_managed,
        )
        return JSONResponse(content={
            'entries': _serialize_records(records),
            'groups': manager.build_architecture_groups(
                architecture=architecture,
                sub_architecture=sub_architecture,
                compatibility_family=compatibility_family,
                model_type=model_type,
                root_key=root_key,
                preset_managed=preset_managed,
                installed=True,
            ),
            'count': len(records),
        })

    @router.get('/api/models/browser')
    def browser_payload(
        base_model_name: str | None = Query(default=None),
        root_key: str | None = Query(default=None),
        installed_only: bool = Query(default=False),
        generic_only: bool = Query(default=False),
        include_preset_managed: bool = Query(default=False),
        architecture: str | None = Query(default=None),
        sub_architecture: str | None = Query(default=None),
    ):
        scope = manager.get_filter_scope(base_model_name, root_key=root_key, model_type='lora' if root_key == 'loras' else None)
        if architecture is None:
            architecture = scope['architecture']
        if sub_architecture is None:
            sub_architecture = scope['sub_architecture']

        records = manager.iter_inventory(
            architecture=architecture,
            sub_architecture=sub_architecture,
            root_key=root_key,
            installed=installed_only if installed_only else None,
            preset_managed=False if not include_preset_managed else None,
        )
        if generic_only:
            records = [record for record in records if record.entry.visibility == 'generic']
        if not include_preset_managed:
            records = [record for record in records if not record.entry.preset_managed]

        installed_records = [record for record in records if record.installed]
        available_records = [record for record in records if not record.installed]
        return JSONResponse(content={
            'scope': scope,
            'installed': _serialize_records(installed_records),
            'available': _serialize_records(available_records),
            'groups': manager.build_architecture_groups(
                architecture=architecture,
                sub_architecture=sub_architecture,
                root_key=root_key,
                installed=installed_only if installed_only else None,
                preset_managed=False if not include_preset_managed else None,
            ),
            'count': len(records),
        })

    @router.post('/api/models/download')
    def start_download(payload: dict = Body(...)):
        selector = payload.get('selector') if isinstance(payload, dict) else None
        if not selector and isinstance(payload, dict):
            selector = payload.get('model_id')
        if not selector:
            raise HTTPException(status_code=400, detail='Missing selector')

        worker = _build_job_worker(manager, download_worker=download_worker)
        try:
            job = manager.start_download_job(str(selector), worker=worker)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return JSONResponse(content={
            'status': 'queued',
            'message': DEFAULT_DOWNLOAD_MESSAGE,
            'job': job.to_dict(),
        }, status_code=202)

    @router.get('/api/models/downloads/{job_id}')
    def get_download_status(job_id: str):
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Download job not found')
        return JSONResponse(content=job.to_dict())

    @router.post('/api/models/refresh')
    def refresh_models():
        manager.refresh_installed_index()
        return JSONResponse(content={
            'status': 'success',
            'installed': len(manager.list_installed()),
            'available': len(manager.list_available()),
            'groups': manager.build_architecture_groups(),
        })

    return router


model_router = create_model_router(default_model_manager)
