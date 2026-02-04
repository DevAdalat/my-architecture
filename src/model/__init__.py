"""DPSN-R Model Package.

Exports the main model class and configuration.
"""

from .act import AdaptiveComputeTime, compute_ponder_loss
from .config import DPSNRConfig
from .controller import TinyController
from .dpsn_r import DPSNR
from .pool import MassivePool

__all__ = [
    "DPSNR",
    "DPSNRConfig",
    "TinyController",
    "MassivePool",
    "AdaptiveComputeTime",
    "compute_ponder_loss",
]
