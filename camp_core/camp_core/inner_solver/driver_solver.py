from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cvxpy as cp
import numpy as np
import torch

from camp_core.atoms.base import Context

class DriverAwareInnerSolver:
    """
    Solves the inner optimization problem using driver atoms.
    Optimized uses CVXPY Parameters (DPP) to avoid re-canonicalization overhead.
    STRICT DPP COMPLIANCE: Use "Slack Variables" to move geometric parameters into constraints.
    Minimize w * t^2 subject to t == A(p)x + b(p).
    """
    MAX_OBSTACLES = 32 # Maximum number of obstacles to handle via padding

    def __init__(self, horizon: int, dt: float) -> None:
        self.horizon = horizon
        self.dt = dt
        
        self.atoms = [
            "jerk",
            "smoothness",
            "lane_deviation",
            "speed_limit",
            "progress",
            "clearance",
        ]
        self.atom_names = self.atoms

        # --- Build DPP Problem ---
        self._build_dpp_problem()

    def _build_dpp_problem(self):
        # Variables
        self.y = cp.Variable((self.horizon, 2))
        
        # Slack Variables (for DPP compliance)
        # Objectives will be sum_squares(slack)
        self.slack_jerk = cp.Variable((self.horizon - 3, 2)) # Jerk vector
        self.slack_smooth = cp.Variable((self.horizon - 2, 2)) # Accel vector
        self.slack_lane = cp.Variable(self.horizon)
        self.slack_speed_proxy = cp.Variable(self.horizon - 1) # Norm of velocity
        self.slack_speed_viol = cp.Variable(self.horizon - 1) # Violation
        self.slack_progress = cp.Variable()
        self.slack_clear = cp.Variable((self.MAX_OBSTACLES, self.horizon))
        
        # Parameters
        self.p_y0 = cp.Parameter(2, name="p_y0")
        
        # Weights (Standard Parameters)
        self.p_w_jerk = cp.Parameter(nonneg=True, name="p_w_jerk")
        self.p_w_smooth = cp.Parameter(nonneg=True, name="p_w_smooth")
        self.p_w_lane = cp.Parameter(nonneg=True, name="p_w_lane")
        self.p_w_speed = cp.Parameter(nonneg=True, name="p_w_speed")
        self.p_w_progress = cp.Parameter(nonneg=True, name="p_w_progress")
        
        # Per-obstacle weights (vector)
        self.p_w_obs = cp.Parameter(self.MAX_OBSTACLES, nonneg=True, name="p_w_obs")
        
        # Geometric Parameters
        # 1. Lane Deviation
        self.p_ref_normals = cp.Parameter((self.horizon, 2), name="p_ref_normals")
        self.p_lane_ref_proj = cp.Parameter(self.horizon, name="p_lane_ref_proj") 
        
        # 3. Speed Limit
        self.p_speed_limit = cp.Parameter(nonneg=True, name="p_speed_limit")
        
        # 3. Progress
        self.p_progress_target = cp.Parameter(name="p_progress_target")
        self.p_ref_tangents = cp.Parameter((self.horizon - 1, 2), name="p_ref_tangents") # Tangents for path integral
        
        # 4. Clearance
        self.p_safety_thresh = cp.Parameter(nonneg=True, name="p_safety_thresh")
        self.p_obs_dists = cp.Parameter((self.MAX_OBSTACLES, self.horizon), name="p_obs_dists") 
        self.p_obs_normals = cp.Parameter((self.MAX_OBSTACLES * self.horizon, 2), name="p_obs_normals")
        self.p_obs_ref_proj = cp.Parameter((self.MAX_OBSTACLES, self.horizon), name="p_obs_ref_proj")
        
        # --- Constraints & Objectives ---
        constraints = [self.y[0] == self.p_y0]
        
        # 0. Jerk
        # slack == jerk
        vel = (self.y[1:] - self.y[:-1]) / self.dt
        acc = (vel[1:] - vel[:-1]) / self.dt
        jerk = (acc[1:] - acc[:-1]) / self.dt
        constraints.append(self.slack_jerk == jerk)
        # Cost: w * sum_sq(slack) / H
        cost_jerk = cp.sum_squares(self.slack_jerk) * (self.p_w_jerk / (self.horizon - 3))

        # 1. Smoothness
        # slack == acc
        constraints.append(self.slack_smooth == acc)
        # Cost: w * sum_sq(slack) / H
        cost_smooth = cp.sum_squares(self.slack_smooth) * (self.p_w_smooth / (self.horizon - 2))
        
        # 2. Lane Deviation
        # slack == dot(y, n) - proj
        y_proj = cp.sum(cp.multiply(self.y, self.p_ref_normals), axis=1)
        constraints.append(self.slack_lane == (y_proj - self.p_lane_ref_proj))
        cost_lane = cp.sum_squares(self.slack_lane) * (self.p_w_lane / self.horizon)
        
        # 3. Speed Limit
        # slack_proxy >= norm(v)
        # slack_viol >= slack_proxy - limit, >= 0
        constraints.append(cp.norm(vel, axis=1) <= self.slack_speed_proxy)
        constraints.append(self.slack_speed_viol >= self.slack_speed_proxy - self.p_speed_limit)
        constraints.append(self.slack_speed_viol >= 0)
        cost_speed = cp.sum_squares(self.slack_speed_viol) * (self.p_w_speed / (self.horizon - 1))
        
        # 4. Progress
        # Path Integral Progress: Measure displacement along reference tangents
        # dist = sum( (y[t+1] - y[t]) . tangent[t] )
        traj_diffs = self.y[1:] - self.y[:-1]
        step_progs = cp.sum(cp.multiply(traj_diffs, self.p_ref_tangents), axis=1)
        dist_traveled = cp.sum(step_progs)
        
        constraints.append(self.slack_progress >= self.p_progress_target - dist_traveled)
        constraints.append(self.slack_progress >= 0)
        cost_progress = cp.square(self.slack_progress) * self.p_w_progress
        
        # 5. Clearance
        # slack[m] >= thresh - (d0 + proj), >= 0
        # lin_dist = d0 + (y.n - ref_proj)
        # y.n
        # Reshape y for broadcasting? No, manual loop for clarity
        for m in range(self.MAX_OBSTACLES):
             start_idx = m * self.horizon
             end_idx = (m + 1) * self.horizon
             normals_m = self.p_obs_normals[start_idx:end_idx]
             
             y_proj_m = cp.sum(cp.multiply(self.y, normals_m), axis=1)
             ref_proj_m = self.p_obs_ref_proj[m]
             proj_m = y_proj_m - ref_proj_m
             
             lin_dist_m = self.p_obs_dists[m] + proj_m
             
             # Constraint
             constraints.append(self.slack_clear[m] >= self.p_safety_thresh - lin_dist_m)
             
        constraints.append(self.slack_clear >= 0)
        
        # Weighted sum of slacks
        # cost = sum_m ( w_m * sum_t(slack[m,t]^2) ) / H
        #      = sum_m ( w_m * norm(slack[m])^2 ) / H
        # Use simple multiplication per obstacle
        # cp.sum_squares(slack_clear) calculates sum over all elements
        # We need row-wise weighting
        
        clear_costs = []
        for m in range(self.MAX_OBSTACLES):
             clear_costs.append(self.p_w_obs[m] * cp.sum_squares(self.slack_clear[m]))
             
        term_clearance = cp.sum(clear_costs) / self.horizon
        
        # --- Total Objective ---
        obj_expr = cost_jerk + cost_smooth + cost_lane + cost_speed + cost_progress + term_clearance
        
        self.objective = cp.Minimize(obj_expr)
        self.constraints = constraints
        self.problem = cp.Problem(self.objective, self.constraints)
        
        self.raw_costs_list = [cost_jerk, cost_smooth, cost_lane, cost_speed, cost_progress, term_clearance] # These are weighted now, but good enough

        # Check DPP (Strict)
        if not self.problem.is_dpp():
             print(f"WARNING: Solver problem is NOT DPP compliant! Speedup will be limited. DPP={self.problem.is_dpp()}, DCP={self.problem.is_dcp()}")

    def solve(
        self,
        y0: np.ndarray,
        w: np.ndarray,
        context: Context,
        warm_start_traj: np.ndarray = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        DPP Optimized Solve.
        """
        # SAFEGUARD: Ensure all inputs are numpy arrays
        if isinstance(y0, torch.Tensor): y0 = y0.detach().cpu().numpy()
        if isinstance(w, torch.Tensor): w = w.detach().cpu().numpy()
        if warm_start_traj is not None and isinstance(warm_start_traj, torch.Tensor): 
            warm_start_traj = warm_start_traj.detach().cpu().numpy()
            
        if hasattr(context, "lane_centerline"):
            lc = getattr(context, "lane_centerline")
            if isinstance(lc, torch.Tensor):
                try:
                    setattr(context, "lane_centerline", lc.detach().cpu().numpy())
                except AttributeError:
                    pass 
        if hasattr(context, "__dataclass_fields__"):
            from dataclasses import asdict
            context = asdict(context)
            
        if hasattr(context, "__dataclass_fields__"):
            from dataclasses import asdict
            context = asdict(context)
            

            
        # Data Preparation
        self.p_y0.value = y0
        
        # Scales & Coeffs
        scale_keys = ["jerk_scale", "smooth_scale", "corridor_scale", "speed_limit_scale", "progress_scale", "clearance_scale"]
        scales = []
        for k in scale_keys:
            s = context.get(k, 1.0)
            if s is None: s = 1.0
            scales.append(max(float(s), 1e-4))
        scales_arr = np.array(scales)
        if verbose:
            print(f"  Scales: {scales_arr}")
            print(f"  Progress Scale: {context.get('progress_scale')}")
        
        # Calculate coefficients: w_i / scale_i
        w_abs = np.abs(w)
        coeffs = w_abs / (scales_arr + 1e-6)
        
        # 2. Reference & Lane Deviation
        ref_traj = warm_start_traj
        if ref_traj is None:
             ref_traj = context.get("reference_trajectory", None)
        if ref_traj is None:
            ref_traj = np.tile(y0, (self.horizon, 1))
        
        if ref_traj.shape[0] != self.horizon:
             if ref_traj.shape[0] > self.horizon:
                 ref_traj = ref_traj[:self.horizon]
             else:
                 last = ref_traj[-1]
                 pad = np.tile(last, (self.horizon - ref_traj.shape[0], 1))
                 ref_traj = np.vstack([ref_traj, pad])
        
        # Lane Actives & Normals
        ref_normals = context.get("reference_normals", None)
        lane_active_val = 0.0
        
        # Dynamic Computation of Lane Normals if missing
        if ref_normals is None and "lane_centerline" in context and context["lane_centerline"] is not None:
             lc = context["lane_centerline"]
             if len(lc) > 1:
                 # Helper to compute normals locally
                 # 1. Segment tangents of centerline
                 cl_diffs = lc[1:] - lc[:-1]
                 cl_lens = np.linalg.norm(cl_diffs, axis=1, keepdims=True)
                 cl_tangents = cl_diffs / (cl_lens + 1e-6)
                 # 2. Segment normals (-y, x)
                 cl_normals = np.stack([-cl_tangents[:, 1], cl_tangents[:, 0]], axis=1) # [N-1, 2]
                 
                 # 3. For each ref point, find closest segment
                 # Vectorized projection
                 # Expand dims: ref [H, 1, 2], cl_start [1, N-1, 2]
                 ref_exp = ref_traj[:, None, :]
                 cl_start = lc[:-1][None, :, :]
                 
                 # vec_ap = p - a
                 vec_ap = ref_exp - cl_start # [H, N-1, 2]
                 
                 # t = dot(ap, ab) / dot(ab, ab)
                 # ab is cl_diffs [1, N-1, 2]
                 cl_diffs_exp = cl_diffs[None, :, :]
                 dot_ap_ab = np.sum(vec_ap * cl_diffs_exp, axis=2)
                 dot_ab_ab = np.sum(cl_diffs_exp * cl_diffs_exp, axis=2)
                 t = dot_ap_ab / (dot_ab_ab + 1e-6)
                 t_clipped = np.clip(t, 0.0, 1.0)
                 
                 # Project point
                 proj_points = cl_start + t_clipped[:, :, None] * cl_diffs_exp # [H, N-1, 2]
                 
                 # Distances
                 dists = np.linalg.norm(ref_exp - proj_points, axis=2)
                 min_idxs = np.argmin(dists, axis=1) # [H]
                 
                 # Select normals and projections
                 # Use normal of the closest segment
                 computed_normals = cl_normals[min_idxs] # [H, 2]
                 closest_projs = proj_points[np.arange(self.horizon), min_idxs] # [H, 2]
                 
                 # Calculate scalar projection for constraint: (y . n) - (proj . n)
                 # We want slack = (y - proj) . n
                 # So p_lane_ref_proj = proj . n
                 computed_projs = np.sum(closest_projs * computed_normals, axis=1)
                 
                 
                 ref_normals = computed_normals
                 self.p_ref_normals.value = computed_normals
                 self.p_lane_ref_proj.value = computed_projs
                 lane_active_val = 1.0

        if ref_normals is not None and lane_active_val == 0.0:
             if len(ref_normals) >= self.horizon:
                 if len(ref_normals) > self.horizon: ref_normals = ref_normals[:self.horizon]
                 lane_active_val = 1.0
                 self.p_ref_normals.value = ref_normals
                 lane_proj = np.sum(ref_traj * ref_normals, axis=1)
                 self.p_lane_ref_proj.value = lane_proj
            # [Fix] Apply warm start BEFORE linearization to ensure constraints are built around the correct trajectory
        if warm_start_traj is not None and warm_start_traj.shape == (self.horizon, 2):
            if verbose: print(f"  [Solver] Applied Warm Start. Shape: {warm_start_traj.shape}")
            if verbose: print(f"  [Solver Debug] WarmStart[0]: {warm_start_traj[0]}, y0: {y0}")
            
            # Check for Time Index Mismatch (t=0 vs t=1)
            aligned_ws = warm_start_traj.copy()
            norm_diff = np.linalg.norm(warm_start_traj[0] - y0)
            if norm_diff > 0.01:
                # Mismatch detected. Perform Shift.
                if verbose: print(f"  [Solver] Detected Start Mismatch ({norm_diff:.4f}). Shifting Warm Start to align t=0.")
                aligned_ws[1:] = warm_start_traj[:-1]
                aligned_ws[0] = y0
                
            self.y.value = aligned_ws
        else:
            if warm_start_traj is not None:
                print(f"  [Solver WARNING] Warm Start shape mismatch! Expected ({self.horizon}, 2), got {warm_start_traj.shape if warm_start_traj is not None else 'None'}")

        # 1. Linearization Point
        # Use previous solution (warm start) or zeros
        ref_traj = self.y.value if self.y.value is not None else np.zeros((self.horizon, 2))

        
        if lane_active_val == 0.0 and ref_normals is None:
            self.p_ref_normals.value = np.zeros((self.horizon, 2))
            self.p_lane_ref_proj.value = np.zeros(self.horizon)
        
            
        # Weights (Linear)
        self.p_w_jerk.value = coeffs[0]
        self.p_w_smooth.value = coeffs[1]
        self.p_w_lane.value = coeffs[2] * lane_active_val
            
        # 3. Speed Limit
        sl = context.get("speed_limit", None)
        speed_active_val = 0.0
        if sl is not None:
             speed_active_val = 1.0
             self.p_speed_limit.value = float(sl)
        else:
             speed_active_val = 0.0
             self.p_speed_limit.value = 10.0
        self.p_w_speed.value = coeffs[3] * speed_active_val
             
        # 4. Progress
        # 4. Progress
        ds = context.get("desired_speed", None)
        progress_active_val = 0.0
        
        # Determine Progress Target: Prioritize Warm Start prediction over Context speed
        target_dist = 0.0
        use_progress = False
        
        if warm_start_traj is not None:
             # Priority 1: Use Warm Start Intention (Arc Length)
             # This prevents the solver from stopping if context.desired_speed is conservatively 0
             ws_diffs = warm_start_traj[1:] - warm_start_traj[:-1]
             target_dist = np.sum(np.linalg.norm(ws_diffs, axis=1))
             use_progress = True
        elif ds is not None:
             # Priority 2: Use Context Desired Speed (Constant Velocity Assumption)
             target_dist = float(ds) * (self.horizon - 1) * self.dt
             use_progress = True
             
        if use_progress:
             progress_active_val = 1.0
             self.p_progress_target.value = target_dist
             
             # Compute tangents from reference trajectory (ref_traj is typically warm_start_traj or linear extrapolation)
             ref_vels = ref_traj[1:] - ref_traj[:-1] # [H-1, 2]
             ref_speeds = np.linalg.norm(ref_vels, axis=1, keepdims=True)
             ref_tangents = ref_vels / (ref_speeds + 1e-6) # Normalize
             
             # Handle 0 speed case (use previous tangent or forward vector)
             is_static = (ref_speeds < 1e-3).flatten()
             if np.any(is_static):
                 fv = context.get("forward_vector")
                 if fv is None: fv = np.array([1.0, 0.0])
                 ref_tangents[is_static] = fv
                 
             self.p_ref_tangents.value = ref_tangents
        else:
             progress_active_val = 0.0
             self.p_progress_target.value = 0.0
             self.p_ref_tangents.value = np.zeros((self.horizon - 1, 2))
             
        self.p_w_progress.value = coeffs[4] * progress_active_val
             
        # 5. Clearance
        all_obs_dists = []
        all_obs_normals = []
        
        threshold = context.get("safety_radius", 2.0) + context.get("clearance_soft_margin", 1.0)
        self.p_safety_thresh.value = threshold
        clearance_active_val = 1.0 
        
        static_obs = context.get("static_obstacles", None)
        dyn_obs_dict = context.get("dynamic_obstacles", None)
        
        def linearize_obs(obs_traj_h2):
             diff = ref_traj - obs_traj_h2 
             dist = np.linalg.norm(diff, axis=1) 
             safe_dist = np.maximum(dist, 1e-6)
             normal = diff / safe_dist[:, None] 
             return dist, normal
             
        if static_obs is not None and len(static_obs) > 0:
             diffs = ref_traj[:, None, :] - static_obs[None, :, :]
             dists = np.linalg.norm(diffs, axis=2)
             min_idxs = np.argmin(dists, axis=1)
             closest_static_traj = static_obs[min_idxs]
             d, n = linearize_obs(closest_static_traj)
             all_obs_dists.append(d)
             all_obs_normals.append(n)
             
        if dyn_obs_dict is not None:
             for obs_traj in dyn_obs_dict.values():
                  len_o = obs_traj.shape[0]
                  if len_o >= self.horizon:
                      ot = obs_traj[:self.horizon]
                  else:
                      last = obs_traj[-1]
                      pad = np.tile(last, (self.horizon - len_o, 1))
                      ot = np.vstack([obs_traj, pad])
                  d, n = linearize_obs(ot)
                  all_obs_dists.append(d)
                  all_obs_normals.append(n)
                  
        num_obs = len(all_obs_dists)
        
        val_dists = np.zeros((self.MAX_OBSTACLES, self.horizon))
        val_normals = np.zeros((self.MAX_OBSTACLES, self.horizon, 2))
        val_ref_proj = np.zeros((self.MAX_OBSTACLES, self.horizon))
        val_w_obs = np.zeros(self.MAX_OBSTACLES)
        
        for m in range(min(num_obs, self.MAX_OBSTACLES)):
             val_dists[m] = all_obs_dists[m]
             val_normals[m] = all_obs_normals[m]
             val_ref_proj[m] = np.sum(ref_traj * all_obs_normals[m], axis=1)
             val_w_obs[m] = coeffs[5] * clearance_active_val * 1.0 
             
        self.p_obs_dists.value = val_dists
        self.p_obs_normals.value = val_normals.reshape(-1, 2)
        self.p_obs_ref_proj.value = val_ref_proj
        self.p_w_obs.value = val_w_obs
        
        # --- Solve ---
        try:
             # if warm_start_traj is not None and warm_start_traj.shape == (self.horizon, 2):
             #     self.y.value = warm_start_traj
                 
             # Relax tolerances for robustness against random weights
             solver_opts = {"tol_gap_abs": 1e-4, "tol_gap_rel": 1e-4, "tol_feas": 1e-4}
             
             try:
                 self.problem.solve(solver=cp.CLARABEL, warm_start=True, **solver_opts)
                 
                 # Check for inaccurate solution and re-solve with verbose if requested
                 if self.problem.status == "optimal_inaccurate":
                     print(f"  [Solver Warning] Status is 'optimal_inaccurate'. Re-solving with verbose=True for diagnostics...")
                     self.problem.solve(solver=cp.CLARABEL, verbose=True, **solver_opts)
                     
             except Exception as e_inner:
                 print(f"  [Solver Warning] Standard solve failed ({e_inner}). Retrying with verbose=True...")
                 try:
                     self.problem.solve(solver=cp.CLARABEL, verbose=True, **solver_opts)
                 except Exception as e_verbose:
                     print(f"  [Solver Error] Verbose solve also failed: {e_verbose}")
             
             if self.problem.status in ["optimal", "optimal_inaccurate"]:
                  y_opt = self.y.value
                  
                  # Retrieve atom values? 
                  # Just use the weighted costs as proxies since structure is different
                  raw_vals = np.array([expr.value for expr in self.raw_costs_list])
                  
                  total_cost = self.problem.value
                  if y_opt is None: 
                      return np.zeros((self.horizon, 2)), np.zeros(6), float("inf"), False
                  return y_opt, raw_vals, total_cost, True
             else:
                  print(f"  [Solver Status] Failed with status: {self.problem.status}")
                  return np.zeros((self.horizon, 2)), np.zeros(6), float("inf"), False
        except Exception as e:
             print(f"Solver Exception: {e}")
             return np.zeros((self.horizon, 2)), np.zeros(6), float("inf"), False
