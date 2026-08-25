"""Errors surfaced by the Nexfocus Windows bootstrap."""


class BootstrapError(RuntimeError):
    """Base class for expected, user-actionable bootstrap failures."""


class HardwareProbeError(BootstrapError):
    """NVIDIA hardware or driver information could not be obtained."""


class UnsupportedHardwareError(BootstrapError):
    """The detected GPU is outside the supported compute capability range."""


class ProfileError(BootstrapError):
    """A profile or dependency lock is not an approved release profile."""


class InstallError(BootstrapError):
    """A private runtime installation failed."""


class TransactionError(InstallError):
    """A runtime transaction could not be committed safely."""
