
import os
import argparse
import sys
import json
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../camp_core')))

from camp_core.data_interfaces.nuscenes_trajdata_bridge import NuscenesTrajdataBridge, NuscenesDatasetConfig, extract_driver_context
from camp_core.atoms.driver_atoms import compute_atom_bank_vector, compute_feasibility_mask
from camp_core.base_predictor.trajectron_loader import (
    TrajectronLoadConfig,
    build_trajectron_adapter_from_checkpoint,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/ocean/projects/tra250008p/slin24/data/nuscenes", help="Path to NuScenes data")
    parser.add_argument("--cache_dir", type=str, default="/ocean/projects/tra250008p/slin24/data/nusc_cache_trajdata", help="Cache dir")
    parser.add_argument("--output_file", type=str, default="models/production/atom_scales.json")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    
    # Trajectron Paths
    parser.add_argument("--trajectron_conf", type=str, 
        default="/ocean/projects/tra250008p/slin24/data/nuScenes_v1.0/config_cvpr.json")
    parser.add_argument("--trajectron_model_dir", type=str, 
        default="/ocean/projects/tra250008p/slin24/MetaLearning/models/trajectron_baseline")
    parser.add_argument("--trajectron_epoch", type=int, default=20)
        
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"Loading NuScenes Dataset...")
    cfg = NuscenesDatasetConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        split="nusc_trainval-train",
        batch_size=4,
        num_workers=args.num_workers,
        shuffle=True
    )
    bridge = NuscenesTrajdataBridge(cfg)
    loader = bridge.get_dataloader()
    
    # Init Trajectron
    print("Initializing Trajectron...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajectron_conf = TrajectronLoadConfig(
        conf_path=args.trajectron_conf,
        model_dir=args.trajectron_model_dir,
        device=device,
        epoch=args.trajectron_epoch
    )
    adapter = build_trajectron_adapter_from_checkpoint(
        trajectron_conf,
        embedding_dim=64, # Default dummy
        mode="encoder"
    )
    adapter.eval()
    
    # Init Collections
    all_atoms = [] # List of [R] (representing max_k per scene)
    all_maxs = [] # Keep Track of True Max for Calibration Check
    
    count = 0
    pbar = tqdm(total=args.num_samples)
    
    def get_candidates(adapter, batch, k=12):
        trajectron = adapter.base_model
        ph = trajectron.hyperparams.get("prediction_horizon", 12)
        device = next(trajectron.parameters()).device
        batch.to(device)
        with torch.no_grad():
             predictions = trajectron.predict(
                 batch,
                 prediction_horizon=ph,
                 num_samples=k,
                 z_mode=False,
                 gmm_mode=False,
                 output_dists=False
             )
        
        from trajdata import AgentType
        batch_preds = []
        for i in range(len(batch.agent_name)):
            node_type = AgentType(batch.agent_type[i].item()) if hasattr(batch, 'agent_type') else 'VEHICLE'
            agent_name = batch.agent_name[i]
            key = f"{str(node_type)}/{agent_name}"
            if key in predictions:
                 p = predictions[key] 
                 if hasattr(p, 'cpu'): p = p.cpu().numpy()
                 batch_preds.append(p)
            else:
                 batch_preds.append(np.zeros((k, ph, 2)))
        return batch_preds

    # 2. Iterate
    for batch in loader:
        if count >= args.num_samples:
            break
            
        # Prepare Batch
        batch.to(device)
        
        try:
             # Generate Candidates (Mode: K=12)
             cands_batch = get_candidates(adapter, batch, k=12) # List of [K, H, 2]
             cands_np = np.stack(cands_batch) # [B, K, H, 2] usually, but list serves fine if B loop
        except Exception as e:
            print(f"Trajectron prediction error: {e}")
            continue
            
        B = len(batch.agent_name)
        
        for i in range(B):
            if count >= args.num_samples:
                break
                
            # Extract Context
            try:
                # Need map_api access? NuscenesTrajdataBridge or from batch?
                # Inside compute_atom_scales, 'bridge' is available.
                ctx = extract_driver_context(batch, i, map_api=bridge.dataset) 
            except Exception as e:
                # print(f"Error extracting context: {e}")
                continue
            
            # Use Generated Candidates
            # Shape [K, T, 2]. T might be 20. We need 12 steps (6s) usually for Atoms.
            
            traj_k = cands_np[i] # [K, T, 2]
            if traj_k.shape[1] > 12:
                traj_k = traj_k[:, :12, :] # Crop to 12
            elif traj_k.shape[1] < 2:
                continue
                
            K = traj_k.shape[0]
            scene_atoms = []
            
            # Compute Feasibility FIRST (Critical for robust max)
            # We ONLY want to scale based on FEASIBLE candidates.
            # Infeasible ones (off-road, collision) have extreme values that shouldn't skew the scale.
            
            feas_indices = []
            for k in range(K):
                single_traj = traj_k[k] # [12, 2]
                
                # Check Feasibility
                is_feasible = compute_feasibility_mask(ctx, single_traj)
                
                if is_feasible:
                    at = compute_atom_bank_vector(ctx, single_traj) # [R]
                    scene_atoms.append(at)
                    feas_indices.append(k)
            
            # Fallback: If NO candidates are feasible, SKIP this scene for stats.
            # We don't want to pollute scales with completely infeasible scenarios.
            if len(scene_atoms) == 0:
                 continue
            
            scene_atoms_np = np.stack(scene_atoms) # [M_feas, R]
            
            # Robust Scaling Strategy (Method 5): 
            # Comfort (0-3): Trimmed Max (P90) vs Straight Max (for calibration check)
            # Safety (4-8): Max (Strict)
            
            # We collect TWO sets for Comfort check:
            # 1. The one we use for Scale (P90/Max mix)
            # 2. The True Max (to check tail ratio)
            
            scene_rep = np.zeros(scene_atoms_np.shape[1])
            scene_max_only = np.max(scene_atoms_np, axis=0)
            
            # Comfort: Use P90 for Main Scale
            comfort_dims = [0, 1, 2, 3]
            scene_rep[comfort_dims] = np.percentile(scene_atoms_np[:, comfort_dims], 90, axis=0)
            
            # Safety: Use Max
            safety_dims = [4, 5, 6, 7, 8]
            scene_rep[safety_dims] = scene_max_only[safety_dims]
            
            all_atoms.append(scene_rep)
            all_maxs.append(scene_max_only) # For analysis
            
            count += 1
            pbar.update(1)

    pbar.close()
    
    # 3. Compute Scales
    if len(all_atoms) == 0:
        print("No atoms collected!")
        return

    all_atoms_np = np.stack(all_atoms, axis=0) # [N, R]
    all_maxs_np = np.stack(all_maxs, axis=0)   # [N, R]
    
    print(f"Collected {all_atoms_np.shape[0]} samples. Computing scales...")
    
    # R=9
    # 1-3. Jerk (3)
    # 4. RMS (1)
    # 5-7. Speed (3)
    # 8. Lane (1)
    # 9. Clearance (1)
    
    scales = []
    
    labels = ["Jerk_Early", "Jerk_Late", "Jerk_Full", "RMS_Acc", "Speed_0.0", "Speed_0.5", "Speed_1.0", "Lane_Dev", "Clearance"]
    
    for d in range(all_atoms_np.shape[1]):
        col = all_atoms_np[:, d]
        
        # Robust cleaning
        valid_mask = np.isfinite(col)
        valid_col = col[valid_mask]
        
        # Non-zero for Sparse Analysis
        eps_nz = 1e-6
        non_zeros = valid_col[valid_col > eps_nz]
        
        if len(valid_col) == 0:
            scale = 1.0
        else:
            nz_ratio = len(non_zeros) / len(valid_col)
            
            # Robust Lower Bound: P50 of positive values (or 1e-6)
            # Removes the arbitrary 1.0 floor which suppressed small features (Clearance).
            if len(non_zeros) > 0:
                 p50_pos = np.nanpercentile(non_zeros, 50)
                 s_min = max(p50_pos, 1e-6)
            else:
                 s_min = 1e-6

            # Sparse Dimensions (Speed, Lane, Clearance)
            if nz_ratio < 0.2:
                if len(non_zeros) > 5:
                    p95_nz = np.nanpercentile(non_zeros, 95)
                    scale = p95_nz
                    source = "Sparse_NZ_P95"
                else:
                    if len(non_zeros) > 0:
                        scale = np.max(non_zeros)
                        source = "Sparse_Max"
                    else:
                        scale = 1.0 
                        source = "Empty_Default"
            else:
                # Dense (Jerk, RMS): P95 of All (Actually P95 of P90-Rep)
                p95_all = np.nanpercentile(valid_col, 95)
                scale = p95_all
                source = "Comfort_SceneP90"
                
            # Tail Check
            true_max_col = all_maxs_np[:, d]
            true_max_p95 = np.nanpercentile(true_max_col, 95)
            tail_ratio = true_max_p95 / (scale + 1e-6)
                
            print(f"  [Dim {d} {labels[d]}] Ratio={nz_ratio:.3f}. Method={source}. RawScale={scale:.4f} (MinBound={s_min:.4e})")
            print(f"       Tail Check: Scale(P95 of Rep)={scale:.4f} vs P95(TrueMax)={true_max_p95:.4f}. TailRatio={tail_ratio:.2f}")
            
            # --- DEBUG: Sanity Check Distribution ---
            if len(valid_col) > 0:
                p50_val = np.nanpercentile(valid_col, 50)
                p95_val = np.nanpercentile(valid_col, 95)
                max_val = np.nanmax(valid_col)
                print(f"    Distribution (Scene-Reps): P50={p50_val:.4f}, P95={p95_val:.4f}, Max={max_val:.4f}")
            # ----------------------------------------
        
        # Enforce Minimum Scale
        if scale < s_min:
            print(f"    -> Clamping Scale {scale:.4f} to {s_min:.4e}")
            scale = s_min
            
        # Global absolute floor (safety)
        scale = max(scale, 1e-6)
        
        scales.append(float(scale))
        
    print(f"\nFinal Scales: {scales}")
    
    # 4. Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(scales, f, indent=2)
    print(f"Saved to {args.output_file}")
    
if __name__ == "__main__":
    main()
