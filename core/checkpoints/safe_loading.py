"""Runtime preflight for restricted checkpoint deserialization."""


def require_safe_checkpoint_runtime():
    """Reject unsupported runtimes before any checkpoint bytes are decoded.

    PyTorch 2.10 fixes GHSA-63cw-57p8-fm3p in the weights-only loader.
    Never substitute unrestricted pickle loading for a missing safe API.
    """
    import torch
    from packaging.version import InvalidVersion, Version

    try:
        supported = Version(str(torch.__version__)) >= Version('2.10.0')
    except InvalidVersion:
        supported = False
    if not supported or not callable(getattr(torch.serialization, 'safe_globals', None)):
        raise RuntimeError(
            'Restricted checkpoint loading requires PyTorch >=2.10.0 with '
            'torch.serialization.safe_globals; install requirements-recovery.lock '
            'in the supported CPU environment or upgrade PyTorch.'
        )
