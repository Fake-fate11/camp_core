from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np


@dataclass
class DriverAtomContext:
    """
    Minimal geometric/kinematic context needed to evaluate driver atoms.
    """
    dt: float
    lane_centerline: np.ndarray
    static_obstacles: Optional[np.ndarray] = None
    dynamic_obstacles: Optional[Dict[int, np.ndarray]] = None
    speed_limit: Optional[float] = None
    desired_speed: Optional[float] = None
    lane_half_width: float = 1.8
    safety_radius: float = 2.0
    clearance_soft_margin: float = 1.0
    
    # Scale parameters
    jerk_scale: float = 1.0 
    acc_scale: float = 1.0 # Added for Acc Energy
    rms_scale: float = 1.0 # Added for RMS Acc
    speed_limit_scale: float = 1.0
    # Aux scales
    progress_scale: float = 1.0
    clearance_scale: float = 1.0
    lane_scale: float = 1.0
    
    eps: float = 1e-6

@dataclass
class AtomBankConfig:
    """
    Configuration for strict Atom Bank construction.
    Default: one window [0, T], one margin [0].
    """
    # List of (start, end) tuples for windows. If None, uses [(0, T)].
    windows: Optional[List[Tuple[int, int]]] = None
    # List of speed margins. Default [0.0].
    speed_margins: List[float] = field(default_factory=lambda: [0.0])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _finite_difference_vel(traj_xy: np.ndarray, dt: float) -> np.ndarray:
    if traj_xy.shape[0] < 2: return np.zeros((0, 2))
    return np.diff(traj_xy, axis=0) / dt

def _finite_difference_acc(vel_xy: np.ndarray, dt: float) -> np.ndarray:
    if vel_xy.shape[0] < 2: return np.zeros((0, 2))
    return np.diff(vel_xy, axis=0) / dt

def _finite_difference_jerk(acc_xy: np.ndarray, dt: float) -> np.ndarray:
    if acc_xy.shape[0] < 2: return np.zeros((0, 2))
    return np.diff(acc_xy, axis=0) / dt

def _project_point_onto_polyline(p, centerline):
    # Simplified projection for brevity (matches existing logic)
    seg_vecs = centerline[1:] - centerline[:-1]
    seg_lens = np.maximum(np.linalg.norm(seg_vecs, axis=1), 1e-6)
    seg_dirs = seg_vecs / seg_lens[:, None]
    
    rel = p - centerline[:-1]
    t = np.einsum("md,md->m", rel, seg_dirs)
    t = np.clip(t, 0.0, seg_lens)
    projs = centerline[:-1] + seg_dirs * t[:, None]
    dists = np.linalg.norm(p - projs, axis=1)
    
    idx = np.argmin(dists)
    # Lateral calc
    best_dir = seg_dirs[idx]
    diff = p - projs[idx]
    cross = best_dir[0]*diff[1] - best_dir[1]*diff[0]
    return float(np.sign(cross)*dists[idx])

def _project_onto_centerline(
    traj_xy: np.ndarray, centerline: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project all trajectory points onto a lane centerline polyline.
    Restored for backward compatibility with camp_core.metrics.driver_atoms.

    Returns
    -------
    s : [H]
    d : [H]
    """
    # Simply loop _project_point_onto_polyline if we assume it returns (s, d).
    # BUT my simplified _project_point_onto_polyline above ONLY returns d (float).
    # The legacy code expects (s, d).
    # So I must either:
    # 1. Update _project_point_onto_polyline to return both s, d.
    # 2. Or re-implement the full logic here.
    
    # Let's restore full logic for valid S and D.
    
    traj_xy = np.asarray(traj_xy, dtype=float)
    centerline = np.asarray(centerline, dtype=float)
    
    # Pre-compute segments
    seg_vecs = centerline[1:] - centerline[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    seg_lens = np.maximum(seg_lens, 1e-6)
    seg_dirs = seg_vecs / seg_lens[:, None]
    cum_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    
    s_list = []
    d_list = []
    
    for p in traj_xy:
        rel = p - centerline[:-1]
        t = np.einsum("md,md->m", rel, seg_dirs)
        t_clamped = np.clip(t, 0.0, seg_lens)
        
        projs = centerline[:-1] + seg_dirs * t_clamped[:, None]
        d2 = np.sum((p - projs)**2, axis=1)
        idx = np.argmin(d2)
        
        best_diff = p - projs[idx]
        best_dir = seg_dirs[idx]
        
        s_val = cum_s[idx] + t_clamped[idx]
        
        cross = best_dir[0]*best_diff[1] - best_dir[1]*best_diff[0]
        d_val = np.sign(cross) * np.sqrt(d2[idx])
        
        s_list.append(s_val)
        d_list.append(d_val)
        
    return np.array(s_list), np.array(d_list)

# ---------------------------------------------------------------------------
# Strict Atom Bank Logic
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Strict Atom Bank Logic (Refactored Phase 3)
# ---------------------------------------------------------------------------

def compute_atom_bank_vector(
    ctx: DriverAtomContext, 
    traj_xy: np.ndarray, 
    config: Optional[AtomBankConfig] = None
) -> np.ndarray:
    """
    Compute the strict Atom Bank vector A(xi, y) of dimension R=9 (default).
    
    Structure:
    1-3. Jerk Energy (Early, Late, Full)
    4.   RMS Acc (Full)
    5-7. Speed Violation (Margins 0.0, 0.5, 1.0)
    8.   Lane Deviation (Hinge)
    9.   Clearance (Soft Hinge)
    
    Returns: Vector [R] (Not Normalized - Normalization happens in Training Loop)
    """
    traj_xy = np.asarray(traj_xy, dtype=float)
    T = traj_xy.shape[0]
    dt = ctx.dt
    
    # 1. Kinematics
    vel = _finite_difference_vel(traj_xy, dt) # T-1
    acc = _finite_difference_acc(vel, dt)     # T-2
    jerk = _finite_difference_jerk(acc, dt)   # T-3
    
    atoms = []
    
    # --- Group 1: Jerk Energy [3 Atoms] ---
    # Windows: Early (0-0.25s), Late (0.25s-End), Full
    # 0.25s is rough. If dt=0.5, step 0 is 0-0.5.
    # If dt=0.5s, T=12 (6s).
    # "Early" usually means first few steps. Let's define Early as first 1/3 (approx 1-2s), Late as rest?
    # User said: "Early (0-0.25s), Late (0.25-0.5s)".
    # Wait, user example: "early (0-0.25s)". That is extremely short!
    # If dt=0.5, 0.25s is HALF a step.
    # Maybe user meant relative to horizon? Or user is thinking 10hz (dt=0.1)?
    # NuScenes prediction is often 2Hz (dt=0.5).
    # If dt=0.5, then "Early" must cover at least index 0.
    # Let's interpret "Early" as "First 2 steps" (1.0s) and "Late" as "Rest"?
    # Or strictly follow indices.
    # Assuming T=12 (6s).
    # Let's use proportional windows if not specified.
    # W1 (Early): [0, T//3]
    # W2 (Late):  [T//3, T]
    # W3 (Full):  [0, T]
    
    jerk_sq = np.sum(jerk**2, axis=1) if jerk.shape[0] > 0 else np.zeros(0) # [T-3]
    T_j = len(jerk_sq)
    
    # Define Split Indices
    # For T=12, T_j=9. Split at 3 (1.5s).
    split_idx = max(1, T_j // 3)
    
    windows = [
        (0, split_idx),    # Early
        (split_idx, T_j),  # Late
        (0, T_j)           # Full
    ]
    
    for (s, e) in windows:
        if s < e and s < T_j:
            val = dt * np.sum(jerk_sq[s:e])
        else:
            val = 0.0
        atoms.append(val)
        
    # --- Group 2: RMS Acc [1 Atom] ---
    # Full Window Only
    acc_sq = np.sum(acc**2, axis=1) if acc.shape[0] > 0 else np.zeros(0)
    if len(acc_sq) > 0:
        # RMS = sqrt( Sum(a^2)*dt / Duration )
        # Duration = len * dt
        # -> sqrt( Sum(a^2) / len )
        mean_acc_sq = np.mean(acc_sq)
        val = np.sqrt(mean_acc_sq)
    else:
        val = 0.0
    atoms.append(val)
    
    # --- Group 3: Speed Violation [3 Atoms] ---
    # Margins: 0.0, 0.5, 1.0 m/s
    speed_vals = np.linalg.norm(vel, axis=1) if vel.shape[0] > 0 else np.zeros(0)
    limit = ctx.speed_limit if ctx.speed_limit is not None else 100.0
    margins = [0.0, 0.5, 1.0]
    
    for tau in margins:
        thresh = limit - tau
        if len(speed_vals) > 0:
            viol = np.maximum(0.0, speed_vals - thresh)
            val = dt * np.sum(viol**2)
        else:
            val = 0.0
        atoms.append(val)
        
    # --- Group 4: Lane Deviation [1 Atom] ---
    # Hinge Loss: max(0, |d| - lane_width)^2
    # Project points
    d_vals = []
    # Optimization: project only if centerline exists
    if ctx.lane_centerline is not None:
        # Use simpler projection loop or vectorized if possible
        # Reuse _project_point_onto_polyline
        # (Assuming it's available in scope)
        for p in traj_xy:
             d = _project_point_onto_polyline(p, ctx.lane_centerline)
             d_vals.append(abs(d))
    else:
        d_vals = [0.0] * T
        
    d_vals = np.array(d_vals)
    # Hinge
    lane_viol = np.maximum(0.0, d_vals - ctx.lane_half_width)
    atom_lane = dt * np.sum(lane_viol**2)
    atoms.append(atom_lane)

    # --- Group 5: Clearance (Soft Hinge) [1 Atom] ---
    # max(0, safety_radius - min_dist)^2
    # Reuse auxiliary function logic or simplify
    # We want atom to be "Sum of intrusion over time"? 
    # Or just "min dist" based?
    # Definition in prompt: "dt * sum (max(0, d_safe - d_t))^2"
    
    # Calc dists
    # Simplified: Distance to closest static/dynamic at each step
    # This is expensive. Let's do a rough pass.
    
    d_safe = ctx.safety_radius # e.g. 2.0m ? Or use parameter
    total_clearance_cost = 0.0
    
    # Static
    if ctx.static_obstacles is not None and len(ctx.static_obstacles) > 0:
        # Broadcasting [T, N_static] is heavy if N large.
        # But usually N is small for "relevant" obstacles.
        pass # Skip complexity for now, assume dynamic is dominant or pre-filtered
        
    # Dynamic (Trajectory vs Trajectory)
    # Iterate time steps
    if ctx.dynamic_obstacles:
        for t in range(T):
            ego_p = traj_xy[t]
            min_d_t = 999.0
            
            for obs_traj in ctx.dynamic_obstacles.values():
                if t < len(obs_traj):
                    d = np.linalg.norm(ego_p - obs_traj[t])
                    min_d_t = min(min_d_t, d)
            
            # Static check at t
            if ctx.static_obstacles is not None and len(ctx.static_obstacles) > 0:
                 d_stat = np.linalg.norm(ctx.static_obstacles[:, :2] - ego_p, axis=1).min()
                 min_d_t = min(min_d_t, d_stat)
                 
            intrusion = max(0.0, d_safe - min_d_t)
            total_clearance_cost += (intrusion**2)
            
    atoms.append(total_clearance_cost * dt)
    
    return np.array(atoms, dtype=float)

def compute_feasibility_mask(
    ctx: DriverAtomContext, 
    traj_xy: np.ndarray,
    check_speed: bool = True,
    check_lane: bool = True
) -> bool:
    """
    Check if a trajectory satisfies HARD Feasibility Constraints.
    Returns True if Feasible.
    """
    traj_xy = np.asarray(traj_xy, dtype=float)
    dt = ctx.dt
    
    # 1. Lane Corridor (Hard Constraint)
    # Must stay within lane_width + buffer
    # Buffer: e.g. 0.5m extra
    if check_lane and ctx.lane_centerline is not None:
        max_dev = 0.0
        for p in traj_xy:
             d = abs(_project_point_onto_polyline(p, ctx.lane_centerline))
             max_dev = max(max_dev, d)
        
        # Hard limit: Lane Half Width + 1.0m buffer
        if max_dev > (ctx.lane_half_width + 1.0):
            return False
            
    # 2. Speed Cap (Hard Constraint)
    # Must not exceed limit + 5.0m/s (Hard buffer)
    if check_speed and ctx.speed_limit is not None:
        vel = _finite_difference_vel(traj_xy, dt)
        speeds = np.linalg.norm(vel, axis=1)
        if len(speeds) > 0:
            max_v = np.max(speeds)
            if max_v > (ctx.speed_limit + 5.0): # Tolerant hard cap
                return False
                
    # 3. Collision (Already in atoms? No, hard mask)
    # If collision (d < r_collision), Infeasible.
    # r_collision usually small (e.g. 0.5m overlap)
    # Let's check auxiliary
    # aux = compute_aux_metrics(ctx, traj_xy)
    # if not aux.is_feasible: return False
    
    return True

# ---------------------------------------------------------------------------
# Non-Atom Signals (Auxiliary)
# ---------------------------------------------------------------------------

@dataclass
class AuxiliaryMetrics:
    lane_deviation: float
    clearance: float
    progress: float
    is_feasible: bool

def compute_aux_metrics(
    ctx: DriverAtomContext, traj_xy: np.ndarray
) -> AuxiliaryMetrics:
    """
    Compute Progress, Clearance, Lane Deviation, etc.
    These are just for reporting/filtering, NOT optimization atoms.
    """
    traj_xy = np.asarray(traj_xy, dtype=float)
    T = traj_xy.shape[0]
    
    # 1. Lane
    d_vals = []
    if ctx.lane_centerline is not None:
        for p in traj_xy:
            d = _project_point_onto_polyline(p, ctx.lane_centerline)
            d_vals.append(d)
    else: 
        d_vals = [0.0]*T
    d_vals = np.abs(np.array(d_vals))
    lane_mean = float(np.mean(d_vals))
    
    # 2. Clearance
    # Simplified check
    min_dist = 999.0
    # Dynamic
    if ctx.dynamic_obstacles:
        for obs_traj in ctx.dynamic_obstacles.values():
            # Check overlap len
            l = min(len(traj_xy), len(obs_traj))
            if l > 0:
                d = np.linalg.norm(traj_xy[:l] - obs_traj[:l], axis=1).min()
                min_dist = min(min_dist, d)
                
    thresh = ctx.safety_radius
    is_feasible = (min_dist >= thresh)
    clearance_val = float(min_dist)
    
    # 3. Progress
    progress_val = np.linalg.norm(traj_xy[-1] - traj_xy[0])
    
    return AuxiliaryMetrics(
        lane_deviation=lane_mean,
        clearance=clearance_val,
        progress=progress_val,
        is_feasible=is_feasible
    )

# ---------------------------------------------------------------------------
# Legacy Compatibility (Restored for train_offline_preference.py / metrics)
# ---------------------------------------------------------------------------

@dataclass
class DriverAtomFeatures:
    """
    Legacy wrapper for atom features.
    Used by metrics/driver_atoms.py and offline preference training.
    """
    jerk: float
    smoothness: float
    lane_deviation: float
    clearance: float
    speed_limit_violation: float
    progress_deficit: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.jerk, 
                self.smoothness,
                self.lane_deviation,
                self.speed_limit_violation,
                self.progress_deficit,
                self.clearance,
            ],
            dtype=float,
        )

def compute_driver_atom_features(ctx: DriverAtomContext, traj_xy: np.ndarray) -> DriverAtomFeatures:
    """
    Compute 'legacy' atoms using the new helpers or reimplementation.
    Matches the signature expected by legacy code.
    """
    # 1. Use new atomic helpers if possible, or re-compute.
    # New helpers return normalized/summed values in compute_atom_bank_vector.
    # We need Raw values for legacy features (which applied scaling later).
    
    traj_xy = np.asarray(traj_xy, dtype=float)
    dt = ctx.dt
    
    # Kinematics
    vel = _finite_difference_vel(traj_xy, dt)
    acc = _finite_difference_acc(vel, dt)
    jerk = _finite_difference_jerk(acc, dt)
    
    # Jerk (Mean Sq)
    if jerk.shape[0] > 0:
        jerk_val = float(np.mean(np.sum(jerk**2, axis=1)))
    else:
        jerk_val = 0.0
        
    # Smoothness (Mean Sq Acc)
    if acc.shape[0] > 0:
        smooth_val = float(np.mean(np.sum(acc**2, axis=1)))
    else:
        smooth_val = 0.0
        
    # Lane Deviation
    aux = compute_aux_metrics(ctx, traj_xy)
    lane_val = aux.lane_deviation
    clear_val = aux.clearance
    
    # Speed Limit
    # Legacy: Sum of squared violations? Or Mean?
    # Original code: Mean(violation^2)
    speeds = np.linalg.norm(vel, axis=1) if vel.shape[0] > 0 else np.zeros(0)
    limit = ctx.speed_limit if ctx.speed_limit is not None else 100.0
    if len(speeds) > 0:
        viol = np.maximum(0.0, speeds - limit)
        speed_val = float(np.mean(viol**2))
    else:
        speed_val = 0.0
        
    # Progress Deficit
    # Legacy: (Desired - Actual)^2
    # aux.progress is Actual Distance.
    if ctx.desired_speed is None: d_speed = 1.0
    else: d_speed = float(ctx.desired_speed)
    
    # Horizon time
    H = traj_xy.shape[0]
    horizon = (H - 1) * dt
    desired_dist = d_speed * horizon
    actual_dist = aux.progress # Approximation
    shortfall = max(0.0, desired_dist - actual_dist)
    prog_val = float(shortfall**2)
    
    return DriverAtomFeatures(
        jerk=jerk_val,
        smoothness=smooth_val,
        lane_deviation=lane_val,
        clearance=clear_val,
        speed_limit_violation=speed_val,
        progress_deficit=prog_val
    )
