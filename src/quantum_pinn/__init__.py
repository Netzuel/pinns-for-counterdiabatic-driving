"""Physics-informed neural networks for optimal counterdiabatic quantum computation."""

from .model import QuantumPINN
from .physics import PAPER_WEIGHTS, RETAINED_WEIGHTS, LossWeights, loss_terms
from .training import (
    RunConfig,
    evaluate,
    load_checkpoint,
    load_hamiltonians,
    load_pretrained,
    train,
)

__version__ = "1.0.0"
__all__ = [
    "PAPER_WEIGHTS",
    "RETAINED_WEIGHTS",
    "LossWeights",
    "QuantumPINN",
    "RunConfig",
    "evaluate",
    "load_checkpoint",
    "load_hamiltonians",
    "load_pretrained",
    "loss_terms",
    "train",
]
