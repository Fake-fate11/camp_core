from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader
import numpy as np

# Import from atoms, not metrics
from camp_core.atoms.driver_atoms import DriverAtomContext

try:
    from trajdata import AgentBatch, UnifiedDataset  # type: ignore
except ModuleNotFoundError as e:
    raise ImportError(
        "Could not import 'trajdata'. "
        "Make sure it is installed in the active conda environment. "
        "If you are using NVlabs/adaptive-prediction, run:\n"
        "  cd /ocean/projects/tra250008p/slin24/MetaLearning/adaptive-prediction/unified-av-data-loader\n"
        "  pip install -e ."
    ) from e
except Exception as e:
    raise RuntimeError(
        "trajdata appears to be installed but failed to import correctly. "
        "Please check the 'unified-av-data-loader' submodule and its installation. "
        f"Original import error: {e}"
    ) from e


@dataclass
class NuscenesDatasetConfig:
    data_root: str
    cache_dir: str
    split: str = "nusc_trainval-train"
    batch_size: int = 8
    num_workers: int = 0
    shuffle: bool = False
    pin_memory: bool = True
    unified_dataset_kwargs: Optional[Dict[str, Any]] = field(default=None)


class NuscenesTrajdataBridge:
    """Thin wrapper around trajdata.UnifiedDataset for nuScenes."""

    def __init__(self, cfg: NuscenesDatasetConfig) -> None:
        self.cfg = cfg

        os.makedirs(cfg.cache_dir, exist_ok=True)

        # Infer dataset_id from split name
        # If split contains "mini", use "nusc_mini", else "nusc_trainval"
        if "mini" in cfg.split:
            dataset_id = "nusc_mini"
        else:
            dataset_id = "nusc_trainval"
            
        data_dirs = {dataset_id: cfg.data_root}
        desired_data = [cfg.split]

        unified_kwargs: Dict[str, Any] = dict(
            desired_data=desired_data,
            data_dirs=data_dirs,
            incl_vector_map=False, # Disable to match official Trajectron++ (uses raster maps only)
            incl_raster_map=True, # Enable raster map loading for CNN encoder
            raster_map_params={"px_per_m": 2, "map_size_px": 100, "offset_frac_xy": (-0.75, 0.0)}, # Standard params matching Trajectron config
        )
        if cfg.unified_dataset_kwargs:
            unified_kwargs.update(cfg.unified_dataset_kwargs)

        self.dataset: UnifiedDataset = UnifiedDataset(**unified_kwargs)

    def make_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=self.dataset.get_collate_fn(pad_format="right"),
            pin_memory=self.cfg.pin_memory,
        )

    def get_dataloader(self) -> DataLoader:
        return self.make_dataloader()

    def __len__(self) -> int:
        return len(self.dataset)


def extract_driver_context(
    batch: AgentBatch,
    idx: int,
    map_api: Optional[Any] = None,
    horizon: int = 12
) -> DriverAtomContext:
    """
    Extract DriverAtomContext from a trajdata AgentBatch for a specific agent.

    Parameters
    ----------
    batch : AgentBatch
        Batch of data from trajdata.
    idx : int
        Index of the agent in the batch.
    map_api : Any, optional
        Map API object (e.g., from nuScenes or trajdata's internal map interface)
        to query vector map data.
    horizon : int, optional
        Prediction horizon for dynamic obstacles. Default is 12.

    Returns
    -------
    ctx : DriverAtomContext
        Context populated with lane centerline, obstacles, etc.
    """
    # 1. Basic metadata
    dt_val = getattr(batch, "dt", 0.5)
    
    # Handle tensor dt
    if hasattr(dt_val, "numel") and dt_val.numel() > 1:
        try:
            dt = float(dt_val[idx])
        except:
            dt = float(dt_val[0])
    elif hasattr(dt_val, "item"):
        dt = dt_val.item()
    else:
        dt = float(dt_val)

    # 2. Agent State
    # batch.agent_hist is [B, T_hist, D]
    # batch.curr_agent_state is [B, D]
    hist = batch.agent_hist[idx].cpu().numpy() # [T_hist, D]
    curr_pos = hist[-1, :2]
    
    # Estimate heading if not available
    if hist.shape[0] > 1:
        diff = hist[-1, :2] - hist[-2, :2]
        if np.linalg.norm(diff) > 1e-3:
            heading = np.arctan2(diff[1], diff[0])
        else:
            heading = 0.0
    else:
        heading = 0.0

    # 3. Lane Centerline Extraction
    lane_centerline = None
    vec_map = None
    map_speed_limit = None # Initialize to None
    # Default forward vector based on heading (fallback)
    forward_vector = np.array([np.cos(heading), np.sin(heading)])

    # Get global state for map query and transformation
    # batch.curr_agent_state is [B, D]
    # Assuming StateArrayXYXdYdXddYddH format: 0:x, 1:y, 6:h
    global_state = batch.curr_agent_state[idx].cpu().numpy()
    global_x, global_y = global_state[0], global_state[1]
    # Heading might be at index 6 or -1 depending on state format
    # If D=7, it's likely 6.
    if global_state.shape[0] >= 7:
        global_h = global_state[6]
    else:
        # Fallback: try to infer from history if state doesn't have heading?
        # But history is local...
        # Actually, if we don't have global heading, we can't transform map to local.
        # Let's assume index 6 is heading for now as per debug output.
        global_h = 0.0

    # Try to get vector map from batch (if collated)
    if hasattr(batch, "vector_maps") and batch.vector_maps is not None:
        try:
            # batch.vector_maps might be a list of VectorMap objects
            vec_map = batch.vector_maps[idx]
        except Exception:
            pass
    
    # Fallback: use map_name and external map_api
    if vec_map is None and map_api is not None:
        try:
            if batch.map_names is not None:
                map_name = batch.map_names[idx]
                if hasattr(map_api, "get_map"):
                    vec_map = map_api.get_map(map_name)
                elif hasattr(map_api, "maps") and hasattr(map_api.maps, "get_map"):
                    vec_map = map_api.maps.get_map(map_name)
        except Exception as e:
            print(f"Map API lookup failed: {e}")

    if vec_map is not None:
        try:
            # Query closest lane using GLOBAL coordinates
            query_pt = np.array([global_x, global_y, 0.0])
            
            # get_closest_lane might return a Lane object or similar
            lane = vec_map.get_closest_lane(query_pt)
            
            if lane is not None:
                # Extract speed limit if available
                if hasattr(lane, "speed_limit") and lane.speed_limit is not None:
                     # Usually in m/s. Check for valid range.
                     val = float(lane.speed_limit)
                     if val > 1.0: 
                         map_speed_limit = val
                
                # Debug: Check lane properties
                # print(f"[DEBUG] Lane properties: {dir(lane)}")
                # print(f"[DEBUG] Lane next_lanes: {getattr(lane, 'next_lanes', 'N/A')}")
                
                # Extract centerline points (GLOBAL coordinates)
                raw_centerline = None
                if hasattr(lane, "center") and hasattr(lane.center, "points"):
                    raw_centerline = lane.center.points[:, :2]
                elif hasattr(lane, "points"):
                     raw_centerline = lane.points[:, :2]
                
                if raw_centerline is not None:
                    # Extend centerline with successors
                    current_lane = lane
                    # Calculate current length
                    accumulated_len = np.linalg.norm(np.diff(raw_centerline, axis=0), axis=1).sum()
                    target_len = 80.0 # Extend up to 80m
                    
                    # print(f"[DEBUG] Initial centerline length: {accumulated_len:.2f}m, target: {target_len}m")
                    
                    # Prevent infinite loops
                    steps = 0
                    max_steps = 10
                    
                    while accumulated_len < target_len and steps < max_steps:
                        steps += 1
                        # print(f"[DEBUG] Extension step {steps}: current length {accumulated_len:.2f}m")
                        
                        if not hasattr(current_lane, 'next_lanes') or not current_lane.next_lanes:
                            # print(f"[DEBUG] No next_lanes, stopping extension")
                            break
                            
                        # Get next lane ID (take first one for now)
                        # next_lanes is a set, so we need to convert to list or use iterator
                        next_lane_id = next(iter(current_lane.next_lanes))
                        # print(f"[DEBUG] Trying to fetch next lane: {next_lane_id}")
                        
                        # Fetch next lane object
                        # vec_map.lanes is a list, not a dict, so we need to search by id
                        next_lane = None
                        if hasattr(vec_map, 'lanes') and isinstance(vec_map.lanes, list):
                            # Search for lane with matching id
                            for lane_obj in vec_map.lanes:
                                if hasattr(lane_obj, 'id') and lane_obj.id == next_lane_id:
                                    next_lane = lane_obj
                                    # print(f"[DEBUG] Found next lane in lanes list")
                                    break
                            # if next_lane is None:
                            #     print(f"[DEBUG] Lane {next_lane_id} not found in vec_map.lanes list")
                        elif hasattr(vec_map, 'get_lane'):
                            next_lane = vec_map.get_lane(next_lane_id)
                            # print(f"[DEBUG] vec_map.get_lane returned: {next_lane is not None}")
                        #else:
                            # print(f"[DEBUG] vec_map has neither lanes list nor get_lane method")
                            
                        if next_lane is None:
                            # print(f"[DEBUG] Failed to retrieve next lane, stopping extension")
                            break
                            
                        # Append points
                        if hasattr(next_lane, "center") and hasattr(next_lane.center, "points"):
                            new_points = next_lane.center.points[:, :2]
                            # print(f"[DEBUG] Appending {len(new_points)} points from next lane")
                            raw_centerline = np.concatenate([raw_centerline, new_points], axis=0)
                            accumulated_len += np.linalg.norm(np.diff(new_points, axis=0), axis=1).sum()
                            current_lane = next_lane
                        #else:
                            # print(f"[DEBUG] Next lane has no center.points, stopping extension")
                            break
                    
                    # print(f"[DEBUG] Final centerline length: {accumulated_len:.2f}m after {steps} steps")

                    # Transform Global -> Local
                    
                    # Check for NaNs in global state
                    if np.isnan(global_x) or np.isnan(global_y) or np.isnan(global_h):
                        # print(f"[DEBUG] Global state contains NaNs: x={global_x}, y={global_y}, h={global_h}")
                        lane_centerline = None # Trigger fallback
                    elif np.isnan(raw_centerline).any():
                        # print(f"[DEBUG] Raw centerline contains NaNs!")
                        lane_centerline = None # Trigger fallback
                    else:
                        # Translate
                        dx = raw_centerline[:, 0] - global_x
                        dy = raw_centerline[:, 1] - global_y
                        
                        # Rotate by -heading
                        # x' = x cos(-h) - y sin(-h) = x cos(h) + y sin(h)
                        # y' = x sin(-h) + y cos(-h) = -x sin(h) + y cos(h)
                        c = np.cos(global_h)
                        s = np.sin(global_h)
                        
                        rot_x = dx * c + dy * s
                        rot_y = -dx * s + dy * c
                        
                        lane_centerline = np.stack([rot_x, rot_y], axis=1)
                    
                    # Compute forward vector (tangent at y=0)
                    if lane_centerline is not None:
                        # Find closest point to (0,0)
                        dists_sq = np.sum(lane_centerline**2, axis=1)
                        idx_min = np.argmin(dists_sq)
                        
                        # Compute tangent
                        if idx_min < len(lane_centerline) - 1:
                            tangent = lane_centerline[idx_min+1] - lane_centerline[idx_min]
                        elif idx_min > 0:
                            tangent = lane_centerline[idx_min] - lane_centerline[idx_min-1]
                        else:
                            tangent = np.array([1.0, 0.0])
                            
                        norm = np.linalg.norm(tangent)
                        if norm > 1e-6:
                            forward_vector = tangent / norm
                        else:
                            forward_vector = np.array([1.0, 0.0])

        except Exception as e:
            print(f"Vector map query failed: {e}")

    # 4. Fallback Construction
    if lane_centerline is None:
        # Construct a straight line extending from current state
        # Length 50m, 20 points
        # LOCAL FRAME: Start at (0,0), direction (1,0)
        dists = np.linspace(0, 50, 20)
        lane_centerline = np.stack([dists, np.zeros_like(dists)], axis=1)
        # print(f"Warning: Using fallback straight line map for agent {idx}")
        
        # Forward vector is just the heading direction (x-axis in local frame)
        forward_vector = np.array([1.0, 0.0])

    # 5. Obstacles
    # Extract neighbors from batch.neigh_hist
    # batch.neigh_hist is [B, MaxNeigh, T, D]
    # We need to:
    # 1. Select neighbors for this agent (idx)
    # 2. Filter out padding (check num_neigh or validity)
    # 3. Transform to local frame
    
    dynamic_obstacles = {}
    
    if hasattr(batch, "neigh_hist") and batch.neigh_hist is not None:
        # neigh_hist: [B, M, T, D]
        # num_neigh: [B]
        
        num_n = 0
        if hasattr(batch, "num_neigh"):
            num_n = int(batch.num_neigh[idx])
            
        if num_n > 0:
            # Get neighbors for this agent
            # shape: [M, T, D]
            # Get neighbors for this agent
            # shape: [M, T, D]
            agent_neighs = batch.neigh_hist[idx] 
            
            # Use GT Future if available (Oracle Planning)
            agent_neighs_fut = None
            if hasattr(batch, "neigh_fut") and batch.neigh_fut is not None:
                agent_neighs_fut = batch.neigh_fut[idx] # [M, T_fut, D]
            
            for n_i in range(num_n):
                # Neighbor n_i
                n_hist = agent_neighs[n_i].cpu().numpy()
                curr_n_pos = n_hist[-1, :2]
                lx, ly = curr_n_pos[0], curr_n_pos[1]
                
                obs_traj = None
                
                # Method 1: Use Ground Truth Future (Best for Safety Validation)
                if agent_neighs_fut is not None:
                     n_fut = agent_neighs_fut[n_i].cpu().numpy() # [T_fut, D]
                     if n_fut.shape[0] >= horizon:
                         # Valid future
                         obs_traj = n_fut[:horizon, :2]
                     elif n_fut.shape[0] > 0:
                         # Partial future: Pad with last known velocity? Or just clamp?
                         # For now, append static/last pos
                         curr = n_fut[:, :2]
                         pad_len = horizon - curr.shape[0]
                         last_p = curr[-1]
                         padding = np.tile(last_p, (pad_len, 1))
                         obs_traj = np.concatenate([curr, padding], axis=0)
                         
                # Method 2: Fallback to Constant Velocity
                if obs_traj is None:
                    n_vel = np.zeros(2)
                    if n_hist.shape[0] > 1:
                        diff = n_hist[-1, :2] - n_hist[-2, :2]
                        # dt is batch.dt
                        n_vel = diff / dt
                    
                    H_pred = horizon
                    future_pos = []
                    for t in range(H_pred):
                        pos_t = np.array([lx, ly]) + n_vel * t * dt
                        future_pos.append(pos_t)
                    obs_traj = np.stack(future_pos)
                
                # Sanitize: Remove any NaNs
                if np.isnan(obs_traj).any():
                    # If NaNs exist, try to fill or skip
                    # Simple fix: replace NaNs with last valid or 0? 
                    # If heavily corrupted, skip this obstacle
                     if np.isnan(obs_traj).all():
                         continue
                     else:
                         # Forward fill / backward fill logic or just simple replace
                         # For speed, let's just use 0.0 or skip if safety critical
                         # Better: skip this neighbor if data is bad
                         continue
                
                # Filter by distance
                # Check distance of first point to ego (0,0)
                dist_to_ego = np.linalg.norm(obs_traj[0])
                if dist_to_ego < 100.0:
                    dynamic_obstacles[n_i] = obs_traj

    #  DEBUG: Understand agent_hist structure
    # if hasattr(batch, 'agent_hist') and batch.agent_hist is not None:
    #     hist = batch.agent_hist[idx]
    #     print(f"\n[DEBUG HIST] Shape: {hist.shape}")
    #     print(f"[DEBUG HIST] Type: {type(hist)}")
    #     print(f"[DEBUG HIST] Has _format: {hasattr(hist, '_format')}")
    #     if hasattr(hist, '_format'):
    #         print(f"[DEBUG HIST] Format: {hist._format}")
    #     print(f"[DEBUG HIST] First 3 frames:\n{hist[:3].cpu().numpy()}")
    #     print(f"[DEBUG HIST] Last 3 frames:\n{hist[-3:].cpu().numpy()}")
    
    # DEBUG: Understand agent_fut structure for future speed prediction
    # if hasattr(batch, 'agent_fut') and batch.agent_fut is not None:
    #     fut = batch.agent_fut[idx]
    #     print(f"\n[DEBUG FUT] Shape: {fut.shape}")
    #     print(f"[DEBUG FUT] Type: {type(fut)}")
    #     print(f"[DEBUG FUT] Has _format: {hasattr(fut, '_format')}")
    #     if hasattr(fut, '_format'):
    #         print(f"[DEBUG FUT] Format: {fut._format}")
    #     print(f"[DEBUG FUT] First 3 frames:\n{fut[:3].cpu().numpy()}")
    #     print(f"[DEBUG FUT] Last 3 frames:\n{fut[-3:].cpu().numpy()}")
    

    # Calculate desired speed using GROUND TRUTH future velocities (when available)
    # This dramatically improves accuracy for static/slow vehicles
    desired_speed = 5.0  # Ultimate fallback
    current_speed = 0.0
    avg_historical_speed = 0.0
    fut_avg_speed = 0.0
    
    # Method 1: Extract future speeds from agent_fut (BEST - uses ground truth)
    if hasattr(batch, 'agent_fut') and batch.agent_fut is not None:
        try:
            fut = batch.agent_fut[idx].cpu().numpy()  # [H, D] where H=future horizon
            
            if fut.shape[1] >= 4:  # Has velocity columns
                # Extract future velocities (columns 2,3: xd, yd)
                fut_vx = fut[:, 2]
                fut_vy = fut[:, 3]
                fut_speeds = np.sqrt(fut_vx**2 + fut_vy**2)
                fut_speeds = np.nan_to_num(fut_speeds, nan=0.0) # Handle NaNs
                
                # If future speeds are mostly zero, it might be a stopped vehicle.
                # In this case, rely on current/historical speed.
                if np.all(fut_speeds < 0.1): # If all future speeds are very low
                    # print(f"[DEBUG] GT future speeds are all near zero, falling back to historical.")
                    pass
                else:
                    # Use average of first half of trajectory (more reliable near-term)
                    horizon_mid = max(1, len(fut_speeds) // 2)
                    valid_fut_speeds = fut_speeds[:horizon_mid][fut_speeds[:horizon_mid] > 0.1]
                    
                    if len(valid_fut_speeds) > 0:
                        fut_avg_speed = float(np.mean(valid_fut_speeds))
                        # print(f"[DEBUG] Using GT future speed: {fut_avg_speed:.4f} m/s")
            else:
                 print(f"[DEBUG] Future shape mismatch: {fut.shape}")
        except Exception as e:
            print(f"[DEBUG] Could not extract future speeds: {e}")
    
    # Method 2: Extract historical speeds from agent_hist (FALLBACK)
    if hasattr(batch, 'agent_hist') and batch.agent_hist is not None:
        # Use agent_hist_len to get the actual valid length (excluding padding)
        if hasattr(batch, 'agent_hist_len') and batch.agent_hist_len is not None:
            hist_len = int(batch.agent_hist_len[idx].cpu().item())
            # Slice to get only valid frames (no padding)
            hist = batch.agent_hist[idx, :hist_len].cpu().numpy()  # [hist_len, 8]
        else:
            # Fallback if agent_hist_len is not available
            hist = batch.agent_hist[idx].cpu().numpy()  # [T, 8]
            hist_len = hist.shape[0]
        
        if hist.shape[0] > 0 and hist.shape[1] >= 4:
            # print(f"[DEBUG RAW] Hist shape: {hist.shape} (valid len: {hist_len})")
            # print(f"[DEBUG RAW] Last hist frame: {hist[-1]}")
            
            hist_vx = hist[:, 2]
            hist_vy = hist[:, 3]
            hist_speeds = np.sqrt(hist_vx**2 + hist_vy**2)
            hist_speeds = np.nan_to_num(hist_speeds, nan=0.0) # Handle NaNs
            
            # Extract current speed from the last valid frame (now guaranteed to be valid, not padding)
            # Still check for NaN in case trajdata itself has incomplete data
            last_valid_idx = -1
            for i in range(hist.shape[0] - 1, -1, -1):
                if not np.isnan(hist[i, :2]).any(): # Only check x, y
                    last_valid_idx = i
                    break
            
            if last_valid_idx != -1:
                # Found a frame with valid position
                # Check if it has valid velocity
                has_valid_vel = not np.isnan(hist[last_valid_idx, 2:4]).any()
                
                if has_valid_vel:
                    hist_vx = hist[last_valid_idx, 2]
                    hist_vy = hist[last_valid_idx, 3]
                    raw_speed = np.sqrt(hist_vx**2 + hist_vy**2)
                    current_speed = float(raw_speed)
                else:
                    current_speed = 0.0 # Placeholder, will be fixed by diff calculation below
                
                # Fallback: Calculate velocity from position differences if raw velocity is 0 or NaN
                # We need another valid frame before last_valid_idx
                if (current_speed < 0.01 or not has_valid_vel) and last_valid_idx > 0:
                    prev_valid_idx = -1
                    for i in range(last_valid_idx - 1, -1, -1):
                        if not np.isnan(hist[i, :2]).any(): # Only check x, y
                            prev_valid_idx = i
                            break
                    
                    if prev_valid_idx != -1:
                        # Calculate displacement
                        disp = hist[last_valid_idx, :2] - hist[prev_valid_idx, :2]
                        time_diff = (last_valid_idx - prev_valid_idx) * float(dt)
                        if time_diff > 0:
                            calc_vel = np.linalg.norm(disp) / time_diff
                            # print(f"[DEBUG] Calculated velocity from position diff (idx {prev_valid_idx}->{last_valid_idx}): {calc_vel:.2f} m/s")
                            if calc_vel > 0.1:
                                current_speed = calc_vel
            else:
                # All history positions are NaN
                current_speed = 0.0
                # print("[DEBUG] All agent_hist frames are NaN!")

            valid_hist_speeds = hist_speeds[hist_speeds > 0.5]
            
            if len(valid_hist_speeds) > 0:
                avg_historical_speed = float(np.mean(valid_hist_speeds))
            else:
                avg_historical_speed = 0.0
                
    # Method 3: Extract from curr_agent_state (if available and current_speed is still 0 or suspicious)
    if current_speed < 0.01 and hasattr(batch, 'curr_agent_state') and batch.curr_agent_state is not None:
        try:
            # curr_agent_state: [B, D] or [D]
            # Assuming batch.curr_agent_state[idx] gives the state for this agent
            curr_state = batch.curr_agent_state
            if isinstance(curr_state, torch.Tensor):
                cs = curr_state[idx].cpu().numpy()
            else:
                cs = curr_state[idx].numpy()
            
            # print(f"[DEBUG RAW] Curr state: {cs}")
                
            if cs.shape[0] >= 4:
                # vx, vy are usually at indices 2, 3
                vx, vy = cs[2], cs[3]
                cs_speed = np.sqrt(vx**2 + vy**2)
                if not np.isnan(cs_speed):
                    current_speed = float(cs_speed)
                    # print(f"[DEBUG] Extracted current_speed from curr_agent_state: {current_speed:.2f}")
        except Exception as e:
            pass
            
    # Final NaN check for current_speed
    if np.isnan(current_speed):
        current_speed = 0.0
    
    # Decision Logic: Prioritize GT future > current > historical > default
    has_data = (hasattr(batch, 'agent_hist') and batch.agent_hist is not None)
    
    if fut_avg_speed > 0.1:
        # BEST: Use ground truth future speed
        desired_speed = fut_avg_speed
    elif current_speed > 0.5:
        # GOOD: Vehicle is currently moving
        desired_speed = current_speed
    elif avg_historical_speed > 0.5:
        # OK: Vehicle has moved historically
        desired_speed = avg_historical_speed
    elif has_data:
        # FALLBACK 1: We have history/future data but speeds are low -> Static vehicle
        desired_speed = 0.0
    else:
        # FALLBACK 2: No data available -> Use conservative default
        desired_speed = 5.0
    
    # Apply safety bounds
    # Min: 0.0 m/s - allow completely static
    # Max: 20 m/s - reasonable highway speed
    desired_speed = np.clip(desired_speed, 0.0, 20.0)
    
    # print(f"[DEBUG] Speed - Current: {current_speed:.2f}, Hist: {avg_historical_speed:.2f}, Future: {fut_avg_speed:.2f}, Desired: {desired_speed:.2f} m/s")
    
    # Finalize Speed Limit
    # Priority: Map > Default(10.0)
    # map_speed_limit is initialized to None at the start of the function (via multi-edit)
    # Fallback check mainly for safety
    if 'map_speed_limit' not in locals():
         map_speed_limit = None

    if map_speed_limit is not None:
        final_speed_limit = map_speed_limit
    else:
        final_speed_limit = 10.0
        
    # Apply safety bounds
    # Min: 0.0 m/s - allow completely static
    # Max: 20 m/s - reasonable highway speed
    desired_speed = np.clip(desired_speed, 0.0, 20.0)
    
    # Cap desired speed at speed limit to prevent solver conflict
    # If the user wants to speed, they can, but the base desire should be compliant
    desired_speed = min(desired_speed, float(final_speed_limit))
    
    # print(f"[DEBUG] Speed - Current: {current_speed:.2f}, Hist: {avg_historical_speed:.2f}, Future: {fut_avg_speed:.2f}, Desired: {desired_speed:.2f} m/s")
    
    ctx = DriverAtomContext(
        dt=float(dt),
        lane_centerline=lane_centerline,
        static_obstacles=None, 
        dynamic_obstacles=dynamic_obstacles,
        speed_limit=float(final_speed_limit), 
        desired_speed=float(desired_speed),  
        lane_half_width=2.0, 
        safety_radius=1.0,
        clearance_soft_margin=4.0,
        # agent_hist is no longer supported in strict DriverAtomContext
        # agent_hist=np.array(batch.agent_hist[idx].cpu().numpy()),
        jerk_scale=1.0, 
        acc_scale=1.0,
        rms_scale=1.0,
        lane_scale=1.0,
        clearance_scale=1.0,
        progress_scale=20.0, # Relax progress constraint to prioritize lane adherence
        eps=1e-6,
        # forward_vector=forward_vector # Not supported in strict context
    )
    # Debug print
    # print(f"[DEBUG] Created DriverAtomContext: {ctx}")  # Too verbose
    # print(f"[DEBUG] Has smooth_scale: {hasattr(ctx, 'smooth_scale')}")
    return ctx
