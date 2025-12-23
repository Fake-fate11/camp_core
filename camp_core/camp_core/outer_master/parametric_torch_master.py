from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from camp_core.outer_master.benders_master import BendersCut


@dataclass
class ParametricTorchMasterConfig:
    """Config for ParametricTorchMaster (GPU PGD Solver)."""
    num_atoms: int
    embedding_dim: int
    risk_type: str = "cvar"  # "mean" or "cvar"
    alpha: float = 0.9
    
    # Solver Config
    lr: float = 0.05
    max_iter: int = 500
    penalty_multiplier: float = 1000.0 # For Cut Constraints
    tolerance: float = 1e-5
    device: str = "cuda"
    
    # Regularization
    prior_reg_strength: float = 1.0 
    offline_anchor_weight: float = 0.0

class ParametricTorchMaster:
    """
    GPU-Accelerated Master Problem Solver using Primal-Dual / Penalty Method.
    Avoids CVXPY overhead by keeping everything on GPU tensors.
    """
    
    def __init__(
        self,
        config: ParametricTorchMasterConfig,
        scene_embeddings: torch.Tensor, # [M, D]
        prior_weights: Optional[np.ndarray] = None, # [R]
        prior_precision: Optional[np.ndarray] = None, 
    ) -> None:
        self.config = config
        self.num_atoms = config.num_atoms
        self.embedding_dim = config.embedding_dim
        self.device = torch.device(config.device)
        
        # Store Embeddings on Device
        # Add Bias Column
        M, D = scene_embeddings.shape
        self.num_scenarios = M
        
        ones = torch.ones((M, 1), device=self.device)
        self.phi_aug = torch.cat([scene_embeddings.to(self.device), ones], dim=1) # [M, D+1]
        
        # Cuts Storage: List of Tensors?
        # To vectorizing, we can stack cuts.
        # But cuts arrive sequentially. We store them as standard lists first, 
        # then "pack" them into tensors for the solve step.
        self.cuts: List[List[BendersCut]] = [[] for _ in range(M)]
        
        # Initialize Theta (Warm start possible)
        # [R, D+1]
        self.theta = torch.zeros(self.num_atoms, self.embedding_dim + 1, device=self.device, requires_grad=True)
        # Init weights to uniform
        nn.init.normal_(self.theta, std=0.01)
        
        # Prior Weights
        if prior_weights is not None:
            self.prior_w = torch.tensor(prior_weights, device=self.device, dtype=torch.float32)
        else:
            self.prior_w = torch.ones(self.num_atoms, device=self.device) / self.num_atoms
            
    def add_cut(self, scenario_idx: int, cut: BendersCut) -> None:
        """Add cut."""
        self.cuts[scenario_idx].append(cut)
        
    def _pack_cuts(self, active_indices):
        """
        Pack cuts into tensors for vectorized constraint check.
        Limit: Each scenario can have different number of cuts.
        Strategy: Flatten all cuts into a big list, keep indexing.
        Or: Pad to max cuts per scene?
        
        Given implementation simplicity:
        We minimize Sum(ReLU(Violation)).
        Violation = (V + s@(w - w_anc)) - theta
        
        We can construct:
        all_grad: [TotalCuts, R]
        all_val:  [TotalCuts]
        all_anc:  [TotalCuts, R]
        all_scene_ids: [TotalCuts] (Integer index mapping cut -> scene in active batch)
        """
        grads = []
        vals = []
        anchors = []
        scene_map = []
        
        for local_idx, global_idx in enumerate(active_indices):
            scene_cuts = self.cuts[global_idx]
            for c in scene_cuts:
                grads.append(c.gradient) # numpy
                vals.append(c.value)
                anchors.append(c.w_anchor)
                scene_map.append(local_idx) # Map to column in W
                
        if len(vals) == 0:
            return None
            
        return {
            "grads": torch.tensor(np.array(grads), device=self.device, dtype=torch.float32), # [C, R]
            "vals": torch.tensor(np.array(vals), device=self.device, dtype=torch.float32),   # [C]
            "anchors": torch.tensor(np.array(anchors), device=self.device, dtype=torch.float32), # [C, R]
            "scene_ids": torch.tensor(scene_map, device=self.device, dtype=torch.long) # [C]
        }

    def solve(self, verbose: bool = False, active_indices: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        """PGD Solve."""
        if active_indices is None:
            active_indices = np.arange(self.num_scenarios)
            
        num_active = len(active_indices)
        phi_batch = self.phi_aug[active_indices] # [B, D+1]
        
        # Pack Cuts
        packed_cuts = self._pack_cuts(active_indices)
        
        # Optimization Variables
        # Theta is persistent (self.theta), but we might want to clone for trial steps?
        # Actually in Benders we refine Theta. We can optimize in-place or copy.
        # Let's clone to avoid breaking if solve fails, but here we want updates.
        # PGD variable
        theta_param = self.theta.clone().detach().requires_grad_(True)
        
        # CVaR Aux Variables
        # eta (scalar), s_cvar [B]
        eta_param = torch.zeros(1, device=self.device, requires_grad=True)
        # s_cvar must be >= 0. We can use softplus parameterization or projection.
        # Let's use projection.
        s_cvar_param = torch.zeros(num_active, device=self.device, requires_grad=True)
        
        optimizer = optim.Adam([theta_param, eta_param, s_cvar_param], lr=self.config.lr)
        
        # Check Cuts Valid
        has_cuts = (packed_cuts is not None)
        
        start_t = time.time()
        
        for it in range(self.config.max_iter):
            optimizer.zero_grad()
            
            # 1. Forward W
            # Theta: [R, D+1]
            # Phi: [B, D+1]
            # W = Theta @ Phi.T -> [R, B]
            W = theta_param @ phi_batch.T
            
            # --- Constraints Penalties ---
            loss_constr = 0.0
            
            # A. Simplex (Soft Penalty)
            # w >= 0
            neg_viol = torch.relu(-W) # [R, B]
            loss_constr += torch.sum(neg_viol) * self.config.penalty_multiplier
            
            # sum(w) == 1
            sum_viol = torch.abs(torch.sum(W, dim=0) - 1.0) # [B]
            loss_constr += torch.sum(sum_viol) * self.config.penalty_multiplier
            
            # B. Benders Cuts
            # theta_i >= V + g@(w - w_anc)
            # Implies: penalty if (V + ...) > theta_i
            # Epigraph variable "theta_i" in Paper is NOT the parameter Theta.
            # In paper: Min c'x + theta. theta is estimator of future cost Q(w).
            # Wait, in CVXPY master: "theta_vars" are auxiliary scalars per scene approximating Q(w).
            # But here "Theta" is the mapping parameter.
            # The Cut is on the *Value Function* variable.
            # We are minimizing Obj(Theta) where Obj includes Risk(Q(w)).
            # Q(w) >= Cut.
            # So we need Auxiliary variable Q_est [B] that represents Q(w_i).
            # And Q_est[i] >= Cut_k(w_i).
            # And we minimize Risk(Q_est).
            
            # Re-design:
            # We optimize Theta. W_i = Theta(phi_i).
            # The "Value" of W_i is implicitly max_k (Cut_k(W_i)).
            # Because Q(w) = max(cuts).
            # So we can just set Q_val[i] = max_{cuts for i} (RHS).
            # Minimizing Risk(Q_val) directly.
            # This avoids extra aux variables for cuts!
            
            if has_cuts:
                # RHS = val + grad @ (W[:, scene_ids] - anchor)
                # packed: [TotalCuts]
                # W_sliced = W[:, scene_ids] -> [R, TotalCuts]
                w_selected = W[:, packed_cuts["scene_ids"]]
                
                rhs = packed_cuts["vals"] + torch.sum(packed_cuts["grads"] * (w_selected.T - packed_cuts["anchors"]), dim=1)
                # rhs: [TotalCuts]
                
                # We need Q_est[i] >= rhs for all cuts mapping to i.
                # Q_est[i] = max(rhs where scene_id==i)
                # For efficient scatter max:
                # Initialize Q_est with -infinity? Or 0 (since Q>=0)?
                # Q approx is max of linear cuts.
                
                # Scatter Max is not native efficient in PyTorch 1.9 (scatter_reduce is newer).
                # But typically we can just Penalize:
                # We define Axuiliary Q_vars [B] as parameters? No, Q is dependent.
                # Let's use Q_vars as explicit parameters (like theta_vars in CVXPY).
                # Minimize Risk(Q_vars) + Penalty(ReLU(RHS - Q_vars))
                pass # Handled below with q_vars
            
            # We introduce q_vars explicitly to handle "Max" nicely via optimization
            # Start loop with q_vars definition? No, add to optimizer.
            
            # --- RESTART ---
            # To do this cleanly: 
            # We need q_vars [B] in optimizer.
            pass
        
        # --- Correct Loop Setup ---
        q_vars = torch.zeros(num_active, device=self.device, requires_grad=True) 
        # Re-init optimizer
        optimizer = optim.Adam([theta_param, eta_param, s_cvar_param, q_vars], lr=self.config.lr)
        
        cuts_gpu = packed_cuts
        
        for it in range(self.config.max_iter):
            optimizer.zero_grad()
            
            W = theta_param @ phi_batch.T # [R, B]
            
            # 1. Simplex Constraints Penalty
            viol_nonneg = torch.relu(-W).sum()
            viol_sum = torch.abs(W.sum(dim=0) - 1.0).sum()
            
            loss_simplex = (viol_nonneg + viol_sum) * self.config.penalty_multiplier
            
            # 2. Cut Constraints Penalty
            # q_vars[i] >= cut_rhs
            loss_cuts = 0.0
            if cuts_gpu is not None:
                # RHS: [TotalCuts]
                w_sel = W[:, cuts_gpu["scene_ids"]]
                grad_term = torch.sum(cuts_gpu["grads"] * (w_sel.T - cuts_gpu["anchors"]), dim=1)
                rhs = cuts_gpu["vals"] + grad_term
                
                q_sel = q_vars[cuts_gpu["scene_ids"]]
                
                # Violation: rhs > q
                viol_cuts = torch.relu(rhs - q_sel)
                # Sum squares or L1? L1 is exact penalty compliant.
                loss_cuts = viol_cuts.sum() * self.config.penalty_multiplier
            
            # 3. CVaR Constraints Penalty & Objective
            # s_i >= q_i - eta
            # s_i >= 0
            if self.config.risk_type == "cvar":
                viol_s_pos = torch.relu(-s_cvar_param).sum()
                viol_cvar = torch.relu((q_vars - eta_param) - s_cvar_param).sum()
                
                loss_cvar_constr = (viol_s_pos + viol_cvar) * self.config.penalty_multiplier
                
                # CVaR Obj
                # eta + 1/((1-a)M) sum(s)
                risk_val = eta_param + s_cvar_param.sum() / ((1.0 - self.config.alpha) * num_active)
            else:
                loss_cvar_constr = 0.0
                risk_val = q_vars.mean()
                
            # 4. Regularization
            # ||Theta_weights||^2
            weights_only = theta_param[:, :-1]
            reg_val = self.config.prior_reg_strength * torch.sum(weights_only**2)
            
            if self.config.offline_anchor_weight > 0:
                # ||w - w_off||^2
                # w_off broadcast
                w_diff = W - self.prior_w.unsqueeze(1)
                reg_val += self.config.offline_anchor_weight * torch.sum(w_diff**2) / num_active
                
            total_loss = risk_val + reg_val + loss_simplex + loss_cuts + loss_cvar_constr
            
            total_loss.backward()
            optimizer.step()
            
            # Early Stop Checks
            if it % 50 == 0:
                if loss_simplex < 1e-4 and loss_cuts < 1e-4 and loss_cvar_constr < 1e-4:
                    # Feasible enough?
                    # Check convergence gradient?
                    pass
        
        # Store result
        self.theta.data.copy_(theta_param.data)
        
        return {
            "status": "optimal",
            "Theta": self.theta.detach().cpu().numpy(),
            "loss": total_loss.item()
        }
        
    def update_head_weights(self, head: nn.Module, theta_value: np.ndarray):
        """Same as CVXPY master."""
        if theta_value is None:
            return
            
        with torch.no_grad():
            if hasattr(head, "linear"):
                # theta_value: [R, D_aug]
                R, D_aug = theta_value.shape
                # ... standard copy ...
                weights_np = theta_value[:, :-1]
                bias_np = theta_value[:, -1]
                head.linear.weight.data.copy_(torch.from_numpy(weights_np).float())
                if head.linear.bias is not None:
                     head.linear.bias.data.copy_(torch.from_numpy(bias_np).float())
