from typing import Tuple

import torch
from torch import nn


class LinearMappingHead(nn.Module):
    """Linear mapping from embeddings to weights (raw logits).

    Unlike Softmax-based heads, this head returns w = Theta * phi
    where Theta is the learnable parameter matrix. 
    
    The constraints w >= 0 and sum(w) = 1 are NOT enforced here,
    but must be enforced by the Master optimization problem (CVXPY) 
    or by the downstream logic.

    Args:
        embedding_dim: Dimension of input intermediate features phi(xi).
        num_atoms: Dimension of output weight vector w.
        use_bias: Whether to include a bias term (affine map).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_atoms: int,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_atoms = num_atoms
        # Theta is represented by the weight matrix of a Linear layer
        self.linear = nn.Linear(embedding_dim, num_atoms, bias=use_bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map embeddings of shape (B, D) to raw weights of shape (B, K)."""
        if embeddings.dim() != 2:
            raise ValueError(
                f"Expected embeddings to have shape (B, D), got {embeddings.shape}"
            )
        if embeddings.size(1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, got {embeddings.size(1)}"
            )

        # w = Theta * phi (+ bias)
        weights = self.linear(embeddings)
        return weights

    def extra_repr(self) -> str:
        return f"embedding_dim={self.embedding_dim}, num_atoms={self.num_atoms}, active=Linear(NoSoftmax)"
