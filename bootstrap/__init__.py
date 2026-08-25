"""Nexfocus Windows bootstrap primitives.

The bootstrap package intentionally has no application or third-party imports.
It is copied into the release archive and is also safe to exercise from the
repository's system Python during tests.
"""

from .profiles import APPROVED_PROFILES, LEGACY_PROFILE, MODERN_PROFILE

__all__ = [
    "APPROVED_PROFILES",
    "LEGACY_PROFILE",
    "MODERN_PROFILE",
]
