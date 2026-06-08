from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCOPES = ("workspaces", "outputs-temp", "staging")


@dataclass(frozen=True)
class CleanupTarget:
    label: str
    path: Path
    exclude_paths: tuple[Path, ...] = ()


@dataclass
class CleanupSummary:
    scanned: int = 0
    deleted: int = 0
    reclaimed_bytes: int = 0


def _resolve_path(path: Path) -> Path:
    try:
        return path.resolve()
    except FileNotFoundError:
        return path.absolute()


def _load_repo_cleanup_roots() -> dict[str, Path]:
    original_argv = list(sys.argv)
    original_sys_path = list(sys.path)
    try:
        sys.argv = [original_argv[0] if original_argv else "cleanup_temp_artifacts.py"]
        repo_root_text = str(REPO_ROOT)
        if repo_root_text not in sys.path:
            sys.path.insert(0, repo_root_text)
        import modules.config as config

        temp_root = _resolve_path(Path(config.temp_path))
        outputs_root = _resolve_path(Path(config.path_outputs))
        return {
            "temp-root": temp_root,
            "workspaces": temp_root / "workspaces",
            "outputs-temp": outputs_root / "temp",
            "staging": outputs_root / "staging",
        }
    finally:
        sys.argv = original_argv
        sys.path[:] = original_sys_path


def build_cleanup_targets(
    roots: dict[str, Path],
    *,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
    include_temp_root: bool = False,
) -> list[CleanupTarget]:
    requested_scopes = list(scopes or DEFAULT_SCOPES)
    requested_set = set(requested_scopes)
    targets: list[CleanupTarget] = []

    if include_temp_root:
        excluded_children = []
        workspaces_root = roots.get("workspaces")
        if workspaces_root is not None and "workspaces" in requested_set:
            excluded_children.append(_resolve_path(workspaces_root))
        targets.append(
            CleanupTarget(
                label="temp-root",
                path=_resolve_path(roots["temp-root"]),
                exclude_paths=tuple(excluded_children),
            )
        )

    for scope in requested_scopes:
        path = roots.get(scope)
        if path is None:
            continue
        targets.append(CleanupTarget(label=scope, path=_resolve_path(path)))
    return targets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean temporary Fooocus Nex artifacts. "
            "Default scope: temp/workspaces, outputs/temp, and outputs/staging."
        )
    )
    parser.add_argument(
        "--scope",
        action="append",
        choices=sorted(DEFAULT_SCOPES),
        help="Limit cleanup to one or more scopes. Repeat the flag to add scopes.",
    )
    parser.add_argument(
        "--include-temp-root",
        action="store_true",
        help="Also clean direct children of the configured temp root.",
    )
    parser.add_argument(
        "--older-than-hours",
        type=float,
        default=24.0,
        help="Delete only items older than this many hours. Ignored when --all-ages is used.",
    )
    parser.add_argument(
        "--all-ages",
        action="store_true",
        help="Delete matching items regardless of age.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without removing anything.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    return parser.parse_args()


def _age_cutoff_timestamp(*, older_than_hours: float | None, all_ages: bool) -> float | None:
    if all_ages:
        return None
    hours = float(older_than_hours if older_than_hours is not None else 24.0)
    return time.time() - max(0.0, hours) * 3600.0


def _item_size_bytes(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0

    total = 0
    try:
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += int(child.stat().st_size)
                except OSError:
                    continue
    except OSError:
        return total
    return total


def _latest_mtime(path: Path) -> float:
    try:
        latest = float(path.stat().st_mtime)
    except OSError:
        return 0.0

    if path.is_dir() and not path.is_symlink():
        try:
            for child in path.rglob("*"):
                try:
                    latest = max(latest, float(child.stat().st_mtime))
                except OSError:
                    continue
        except OSError:
            return latest
    return latest


def _should_delete(path: Path, cutoff_timestamp: float | None) -> bool:
    if cutoff_timestamp is None:
        return True
    return _latest_mtime(path) <= cutoff_timestamp


def cleanup_targets(
    targets: list[CleanupTarget],
    *,
    cutoff_timestamp: float | None,
    dry_run: bool = False,
) -> CleanupSummary:
    summary = CleanupSummary()

    for target in targets:
        if not target.path.exists() or not target.path.is_dir():
            continue

        excluded_paths = {_resolve_path(path) for path in target.exclude_paths}
        for child in target.path.iterdir():
            resolved_child = _resolve_path(child)
            if resolved_child in excluded_paths:
                continue
            summary.scanned += 1
            if not _should_delete(child, cutoff_timestamp):
                continue

            reclaimed_bytes = _item_size_bytes(child)
            if not dry_run:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            summary.deleted += 1
            summary.reclaimed_bytes += reclaimed_bytes

    return summary


def _format_bytes(size_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def _print_plan(targets: list[CleanupTarget], *, cutoff_timestamp: float | None, dry_run: bool) -> None:
    mode = "dry-run" if dry_run else "delete"
    print(f"[Cleanup] Mode: {mode}")
    if cutoff_timestamp is None:
        print("[Cleanup] Age filter: all ages")
    else:
        age_hours = max(0.0, (time.time() - cutoff_timestamp) / 3600.0)
        print(f"[Cleanup] Age filter: older than {age_hours:.1f} hours")
    for target in targets:
        print(f"[Cleanup] Target {target.label}: {target.path}")


def _confirm() -> bool:
    reply = input("Proceed with cleanup? [y/N]: ").strip().lower()
    return reply in {"y", "yes"}


def main() -> int:
    args = _parse_args()
    roots = _load_repo_cleanup_roots()
    scopes = tuple(args.scope or DEFAULT_SCOPES)
    targets = build_cleanup_targets(
        roots,
        scopes=scopes,
        include_temp_root=bool(args.include_temp_root),
    )
    cutoff_timestamp = _age_cutoff_timestamp(
        older_than_hours=args.older_than_hours,
        all_ages=bool(args.all_ages),
    )

    _print_plan(targets, cutoff_timestamp=cutoff_timestamp, dry_run=bool(args.dry_run))
    if not args.dry_run and not args.yes and not _confirm():
        print("[Cleanup] Cancelled.")
        return 0

    summary = cleanup_targets(
        targets,
        cutoff_timestamp=cutoff_timestamp,
        dry_run=bool(args.dry_run),
    )
    action = "Would delete" if args.dry_run else "Deleted"
    print(
        f"[Cleanup] {action} {summary.deleted} item(s) from {summary.scanned} scanned item(s); "
        f"reclaimed {_format_bytes(summary.reclaimed_bytes)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
