from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cvxpy as cp
import numpy as np


@dataclass
class BendersCut:
    """Single Benders cut for a given scenario.

    Represents a linear under-estimator of a convex function f_i(w):

        theta_i >= value + gradient^T (w - w_anchor)

    where:
        - value     = f_i(w_anchor)
        - gradient  = subgradient of f_i at w_anchor
    """

    scenario_id: Any
    w_anchor: np.ndarray        # shape [R]
    value: float                # scalar f_i(w_anchor)
    gradient: np.ndarray        # shape [R]


@dataclass
class BendersMasterConfig:
    """Configuration for the Benders master problem.

    Attributes
    ----------
    num_atoms:
        Dimension of the weight vector w (number of atoms).
    scenario_ids:
        Identifiers for scenarios (e.g., scene indices). These determine
        how many theta_i variables we maintain and in which order.
    risk_type:
        Either "mean" (empirical risk) or "cvar".
    alpha:
        Confidence level for CVaR in (0, 1). Only used if risk_type == "cvar".
    solver:
        Name of the cvxpy solver.
    solver_kwargs:
        Extra keyword arguments passed to Problem.solve(...).
    """

    num_atoms: int
    scenario_ids: Sequence[Any]
    risk_type: str = "mean"  # "mean" or "cvar"
    alpha: float = 0.9
    solver: str = "OSQP"
    solver_kwargs: Dict[str, Any] = field(default_factory=dict)
    min_weights: Dict[int, float] = field(default_factory=dict)
    fixed_weights: Dict[int, float] = field(default_factory=dict)


@dataclass
class BendersMasterSolution:
    """Solution returned by the Benders master."""

    status: str
    success: bool
    w_opt: Optional[np.ndarray]          # shape [R]
    theta: Optional[np.ndarray]          # shape [N]
    objective_value: Optional[float]


class BendersMaster:
    """Outer master problem for weights w over atoms with Benders cuts.

    The master maintains a collection of Benders cuts for each scenario i
    and solves an LP over:
        - w in simplex
        - theta_i approximating f_i(w)
        - (optionally) CVaR auxiliary variables

    The objective is either:
        - mean risk: (1/N) sum_i theta_i
        - CVaR_alpha: t + (1 / ((1 - alpha) N)) sum_i s_i
                      with s_i >= theta_i - t, s_i >= 0
    """

    def __init__(self, config: BendersMasterConfig) -> None:
        self.config = config
        self.num_atoms = config.num_atoms

        if self.num_atoms <= 0:
            raise ValueError("num_atoms must be positive.")

        if len(config.scenario_ids) == 0:
            raise ValueError("scenario_ids must be non-empty.")

        self.scenario_ids: List[Any] = list(config.scenario_ids)
        self.scenario_index: Dict[Any, int] = {
            sid: idx for idx, sid in enumerate(self.scenario_ids)
        }

        self.cuts: List[List[BendersCut]] = [[] for _ in self.scenario_ids]

    # ------------------------------------------------------------------
    # Cut management
    # ------------------------------------------------------------------
    def add_cut(self, cut: BendersCut) -> None:
        """Add a Benders cut for the specified scenario."""
        if cut.w_anchor.shape != (self.num_atoms,):
            raise ValueError(
                f"w_anchor for cut must have shape ({self.num_atoms},), "
                f"got {cut.w_anchor.shape}."
            )
        if cut.gradient.shape != (self.num_atoms,):
            raise ValueError(
                f"gradient for cut must have shape ({self.num_atoms},), "
                f"got {cut.gradient.shape}."
            )

        if cut.scenario_id not in self.scenario_index:
            raise KeyError(
                f"Unknown scenario_id {cut.scenario_id!r} for BendersMaster."
            )

        idx = self.scenario_index[cut.scenario_id]
        self.cuts[idx].append(cut)

    def num_cuts_per_scenario(self) -> Mapping[Any, int]:
        """Return number of cuts for each scenario_id."""
        return {
            sid: len(self.cuts[idx]) for idx, sid in enumerate(self.scenario_ids)
        }

    # ------------------------------------------------------------------
    # Master optimization
    # ------------------------------------------------------------------
    def solve(self) -> BendersMasterSolution:
        """Solve the Benders master LP and return the optimal weights w.

        The method assumes that at least one cut has been added for each
        scenario. If some scenario has no cuts, the problem would be
        ill-posed (theta_i unconstrained), so we currently treat that as
        an error.
        """
        num_scenarios = len(self.scenario_ids)
        if num_scenarios == 0:
            raise RuntimeError("No scenarios available in BendersMaster.")

        for idx, sid in enumerate(self.scenario_ids):
            if len(self.cuts[idx]) == 0:
                raise RuntimeError(
                    f"No Benders cuts found for scenario_id={sid!r}; "
                    "cannot build a meaningful master problem."
                )

        # Decision variables
        w = cp.Variable(self.num_atoms)
        theta = cp.Variable(num_scenarios)

        constraints = []

        # Simplex constraints on w
        constraints.append(w >= 0)
        constraints.append(cp.sum(w) == 1.0)

        # Minimum weight constraints
        if self.config.min_weights:
            for atom_idx, min_val in self.config.min_weights.items():
                if 0 <= atom_idx < self.num_atoms:
                    constraints.append(w[atom_idx] >= min_val)

        # Fixed weight constraints
        if self.config.fixed_weights:
            for atom_idx, fixed_val in self.config.fixed_weights.items():
                if 0 <= atom_idx < self.num_atoms:
                    constraints.append(w[atom_idx] == fixed_val)

        # Benders cuts: theta_i >= value + grad^T (w - w_anchor)
        for i, cuts_i in enumerate(self.cuts):
            for cut in cuts_i:
                expr = cut.value + cut.gradient @ (w - cut.w_anchor)
                constraints.append(theta[i] >= expr)

        # Risk objective
        risk_type = self.config.risk_type.lower()
        if risk_type == "mean":
            objective_expr = cp.sum(theta) / num_scenarios
        elif risk_type == "cvar":
            alpha = self.config.alpha
            if not (0.0 < alpha < 1.0):
                raise ValueError("alpha must be in (0, 1) for CVaR.")
            t = cp.Variable()
            s = cp.Variable(num_scenarios)
            constraints.append(s >= 0)
            constraints.append(s >= theta - t)
            objective_expr = t + (1.0 / ((1.0 - alpha) * num_scenarios)) * cp.sum(s)
        else:
            raise ValueError(f"Unsupported risk_type: {self.config.risk_type!r}")

        problem = cp.Problem(cp.Minimize(objective_expr), constraints)

        try:
            problem.solve(
                solver=self.config.solver,
                **self.config.solver_kwargs,
            )
        except Exception as ex:
            return BendersMasterSolution(
                status=f"exception: {type(ex).__name__}: {ex}",
                success=False,
                w_opt=None,
                theta=None,
                objective_value=None,
            )

        status = problem.status
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return BendersMasterSolution(
                status=status,
                success=False,
                w_opt=None,
                theta=None,
                objective_value=None,
            )

        if w.value is None or theta.value is None:
            return BendersMasterSolution(
                status=status,
                success=False,
                w_opt=None,
                theta=None,
                objective_value=None,
            )

        w_opt = np.asarray(w.value).reshape(-1)
        theta_val = np.asarray(theta.value).reshape(-1)
        obj_val = float(objective_expr.value)

        return BendersMasterSolution(
            status=status,
            success=True,
            w_opt=w_opt,
            theta=theta_val,
            objective_value=obj_val,
        )
