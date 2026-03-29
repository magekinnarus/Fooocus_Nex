from __future__ import annotations

import os
import shutil
import subprocess
from typing import Iterable
from urllib.parse import urlparse

from modules.model_loader import load_file_from_url


def download_file(
    url: str,
    *,
    model_dir: str,
    file_name: str | None = None,
    progress: bool = True,
    headers: Iterable[tuple[str, str]] = (),
    prefer_aria2: bool = True,
) -> str:
    os.makedirs(model_dir, exist_ok=True)

    if not file_name:
        file_name = os.path.basename(urlparse(url).path)

    destination = os.path.abspath(os.path.join(model_dir, file_name))
    if os.path.exists(destination):
        return destination

    if prefer_aria2 and shutil.which('aria2c'):
        try:
            return _download_with_aria2(
                url=url,
                model_dir=model_dir,
                file_name=file_name,
                headers=headers,
            )
        except Exception as exc:
            print(f"Aria2 download failed for {url}: {exc}. Falling back to the Python downloader.")
            _cleanup_partial_download(destination)

    return load_file_from_url(
        url=url,
        model_dir=model_dir,
        file_name=file_name,
        progress=progress,
        headers=headers,
    )


def _download_with_aria2(
    *,
    url: str,
    model_dir: str,
    file_name: str,
    headers: Iterable[tuple[str, str]] = (),
) -> str:
    command = [
        'aria2c',
        '--console-log-level=warn',
        '-c',
        '-x', '16',
        '-s', '16',
        '-k', '1M',
        '--dir', model_dir,
        '--out', file_name,
    ]

    for key, value in headers:
        command.extend(['--header', f'{key}: {value}'])

    command.append(url)
    subprocess.check_call(command)
    return os.path.abspath(os.path.join(model_dir, file_name))



def _cleanup_partial_download(destination: str) -> None:
    for candidate in (destination, f'{destination}.aria2'):
        try:
            if os.path.exists(candidate):
                os.remove(candidate)
        except OSError:
            pass

