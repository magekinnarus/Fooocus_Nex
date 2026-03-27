from __future__ import annotations

from abc import ABC, abstractmethod

from .spec import DownloadPlan, DownloadResult


class DownloadTransport(ABC):
    @abstractmethod
    def download(self, plan: DownloadPlan) -> DownloadResult:
        raise NotImplementedError


class Aria2Transport(DownloadTransport):
    def download(self, plan: DownloadPlan) -> DownloadResult:
        raise NotImplementedError('Aria2Transport scaffold is not wired yet')


class FallbackTransport(DownloadTransport):
    def download(self, plan: DownloadPlan) -> DownloadResult:
        raise NotImplementedError('FallbackTransport scaffold is not wired yet')
