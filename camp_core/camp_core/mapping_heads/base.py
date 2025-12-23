from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


@dataclass
class MappingHeadOutput:
    """Container for mapping embeddings to atom weights."""

    weights: torch.Tensor              # shape [B, R]
    extras: Optional[Dict[str, torch.Tensor]] = None


class MappingHeadBase(nn.Module, ABC):
    """Abstract base class for scene-to-weights mapping heads.

    Given scene embeddings h in R^D, a mapping head produces weights
    w in the simplex Delta^{R-1}, where R is the number of atoms.
    """

    def __init__(self, embedding_dim: int, num_atoms: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if num_atoms <= 0:
            raise ValueError("num_atoms must be positive.")
        self.embedding_dim = int(embedding_dim)
        self.num_atoms = int(num_atoms)

    @abstractmethod
    def forward(self, embeddings: torch.Tensor) -> MappingHeadOutput:
        """Map embeddings of shape [B, D] to weights of shape [B, R]."""
        raise NotImplementedError
