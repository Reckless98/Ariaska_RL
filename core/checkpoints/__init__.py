"""Unified checkpoint format for Ariaska RL.

All training products — GPU distillation, local training, enhanced runs —
save and load through ``UnifiedCheckpoint``.  This ensures GPU-trained
weights are loaded identically on local machines.
"""

from core.checkpoints.unified_checkpoint import UnifiedCheckpoint

__all__ = ["UnifiedCheckpoint"]
