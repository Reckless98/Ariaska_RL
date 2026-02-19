"""core/execution/ssh_exceptions.py — Phase 41: SSH pool exception hierarchy."""
from __future__ import annotations


class SSHPoolError(Exception):
    """Base exception for SSH pool errors."""
    pass


class SSHConnectionError(SSHPoolError):
    """Connection failure after retries."""
    pass


class SSHAuthError(SSHPoolError):
    """Authentication failure."""
    pass


class SSHTimeoutError(SSHPoolError):
    """Timeout during SSH operation."""
    pass
