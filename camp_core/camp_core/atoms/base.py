from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import cvxpy as cp
import numpy as np


Context = Dict[str, Any]


@dataclass
class AtomEvalResult:
    """Container for evaluating a single atom at a given solution."""

    name: str
    value: float


class Atom(ABC):
    """Abstract base class for convex atoms a_r(xi, y).

    Subclasses must implement `build_objective`, which returns a convex
    cvxpy expression in the decision variable `y` and the scenario context.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def build_objective(self, y: cp.Variable, context: Context) -> cp.Expression:
        """Build the convex objective expression for this atom.

        Parameters
        ----------
        y:
            cvxpy decision variable (usually a vectorized trajectory).
        context:
            Scenario-specific information (features, geometry, etc.).

        Returns
        -------
        expr:
            A convex cvxpy expression in `y`.
        """
        raise NotImplementedError

    def evaluate(self, y_value: np.ndarray, context: Context) -> AtomEvalResult:
        """Optional convenience method to evaluate the atom at a numpy point.

        This can be overridden with a closed-form implementation if available.
        The default implementation rebuilds the cvxpy expression and plugs
        in the numpy value, which is convenient but not the fastest.
        """
        dim = y_value.size
        y = cp.Variable(dim)
        expr = self.build_objective(y, context)
        prob = cp.Problem(cp.Minimize(expr))
        prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
        # We ignore the optimization result and only use expr.value.
        # In most cases, subclasses should override this method anyway.
        return AtomEvalResult(name=self.name, value=float(expr.value))
