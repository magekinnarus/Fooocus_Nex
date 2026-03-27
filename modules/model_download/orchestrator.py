from __future__ import annotations

from dataclasses import dataclass

from .catalog import ModelCatalog
from .policy import ModelDownloadPolicy
from .resolver import DownloadResolver, DirectResolver
from .spec import DownloadPlan, DownloadResult
from .transport import DownloadTransport, Aria2Transport


@dataclass
class ModelDownloadOrchestrator:
    catalog: ModelCatalog
    policy: ModelDownloadPolicy
    resolver: DownloadResolver | None = None
    transport: DownloadTransport | None = None

    def __post_init__(self):
        if self.resolver is None:
            self.resolver = DirectResolver()
        if self.transport is None:
            self.transport = Aria2Transport()

    def plan(self, selector: str) -> DownloadPlan:
        entry = self.catalog.get(selector)
        if entry is None:
            raise KeyError(f'Unknown model selector: {selector}')
        return self.resolver.resolve(entry, self.policy)

    def download(self, selector: str) -> DownloadResult:
        plan = self.plan(selector)
        return self.transport.download(plan)
