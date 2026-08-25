"""Compatibility shim for the retired inherited Fooocus launcher generator.

The Windows one-click launcher is now a checked-in ``bootstrap`` module and
is invoked by the release batch file. Keeping this import-safe no-op avoids
breaking third-party scripts that imported ``build_launcher`` while ensuring
the old code can never write stale ``run_*.bat`` files into the repository.
"""


is_win32_standalone_build = False


def build_launcher() -> None:
    """Retained for source compatibility; release launchers are prebuilt."""

    return None


__all__ = ["build_launcher", "is_win32_standalone_build"]
