import numpy as np
from typing import Optional, Dict, Any
from camp_core.metrics.driver_atoms import compute_driver_atoms, DriverAtomContext

def compute_ade(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """
    Compute Average Displacement Error.
    pred_traj: [H, 2]
    gt_traj: [H, 2]
    """
    diff = pred_traj - gt_traj
    dist = np.linalg.norm(diff, axis=-1)
    return float(np.mean(dist))

def compute_fde(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """
    Compute Final Displacement Error.
    """
    diff = pred_traj[-1] - gt_traj[-1]
    return float(np.linalg.norm(diff))

def _compute_speed_limit_cost(traj: np.ndarray, limit: float, dt: float) -> float:
    """
    Penalize speed > limit.
    """
    if traj.shape[0] < 2:
        return 0.0
    
    vel = np.linalg.norm(np.diff(traj, axis=0), axis=1) / dt
    violation = np.maximum(vel - limit, 0.0)
    return float(np.mean(violation))

def compute_safety_score(traj: np.ndarray, ctx: DriverAtomContext, weights: Optional[np.ndarray] = None) -> float:
    """
    Compute a Safety Score based on driver atoms.
    Higher score = Unsafe (Higher Cost).
    
    Atoms: [Smoothness, LaneDeviation, SpeedLimit, Progress, Clearance]
    Note: driver_atoms.py returns [Smooth, Lane, Clearance, Progress].
    We need to re-order and add SpeedLimit.
    
    Target Order: [Smooth, Lane, Speed, Progress, Clearance] (matching DriverAwareInnerSolver)
    """
    # 1. Get base atoms [Smooth, Lane, Clearance, Progress]
    # Note: compute_driver_atoms signature: (ctx, traj, centerline, obstacles)
    # But ctx has the data if we populated it? 
    # No, DriverAtomContext is just config + data?
    # Let's check DriverAtomContext definition in driver_atoms.py again.
    # It has fields for config, but not the map data itself?
    # Wait, compute_driver_atoms takes (ctx, traj, centerline, obstacles).
    # But in compare_methods.py, ctx is passed as a single object.
    # And extract_driver_context returns a DriverAtomContext populated with data?
    # Let's check extract_driver_context in nuscenes_trajdata_bridge.py.
    # It returns DriverAtomContext(..., lane_centerline=..., dynamic_obstacles=...)
    # BUT DriverAtomContext definition in driver_atoms.py (which I viewed) DOES NOT have lane_centerline field!
    # It only has config fields (dt, scales, etc).
    # This is a mismatch!
    
    # Let's re-read driver_atoms.py carefully.
    # @dataclass class DriverAtomContext: ... dt, scales ...
    # It does NOT have lane_centerline.
    
    # But extract_driver_context in nuscenes_trajdata_bridge.py (which I viewed) instantiates it with:
    # DriverAtomContext(..., lane_centerline=lane_centerline, ...)
    # This means the definition of DriverAtomContext in driver_atoms.py might be outdated or I missed something.
    # Or maybe python dataclasses allow extra fields if not frozen? No.
    # Or maybe there's another definition?
    
    # Let's assume the one in driver_atoms.py is the one being used.
    # If so, extract_driver_context is passing arguments that don't exist in __init__.
    # This would cause a TypeError at runtime!
    # "TypeError: __init__() got an unexpected keyword argument 'lane_centerline'"
    
    # However, the user script ran until "Solver failed".
    # This implies extract_driver_context might have worked?
    # Or maybe it failed inside the loop and printed "Context extraction failed"?
    # The user log shows: "Initializing Data Loader... Solver failed..."
    # It doesn't show "Context extraction failed".
    # Wait, generate_training_data.py calls extract_driver_context?
    # Yes, likely.
    
    # If DriverAtomContext doesn't have those fields, we need to fix it.
    # I will update DriverAtomContext in driver_atoms.py to include the data fields.
    # AND I will update metrics.py to use them.
    
    # Re-reading driver_atoms.py:
    # It defines compute_driver_atoms(ctx, traj, centerline, obstacles).
    # It expects centerline and obstacles as separate args.
    
    # So if extract_driver_context returns a ctx with these fields, I can extract them from ctx.
    
    # I will update DriverAtomContext to include these fields to be safe and correct.
    
    # For metrics.py:
    lane_centerline = getattr(ctx, "lane_centerline", None)
    # dynamic_obstacles? driver_atoms.py expects static obstacles?
    # driver_atoms.py: obstacles_xy: [M, 2] static.
    # extract_driver_context returns dynamic_obstacles (dict).
    # We might need to flatten dynamic obstacles or use static ones.
    # For now, let's use static_obstacles if present.
    static_obstacles = getattr(ctx, "static_obstacles", None)
    
    # atoms_6 = [Jerk, Smooth, Lane, Speed, Progress, Clearance]
    atoms_6 = compute_driver_atoms(ctx, traj, lane_centerline, static_obstacles)
    
    # Use the computed atoms directly
    vals = atoms_6
    if weights is None:
        # Default safety weights
        # [Jerk, Smooth, Lane, Speed, Progress, Clearance]
        # Exclude Progress (0.0). Average the other 5 (0.2).
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.0, 0.2])
        
    score = np.dot(vals, weights)
    return float(score)
