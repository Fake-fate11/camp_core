
import argparse
import os
import sys
import time
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import gc

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from camp_core.data_interfaces.nuscenes_trajdata_bridge import (
    NuscenesDatasetConfig,
    NuscenesTrajdataBridge,
    extract_driver_context,
)
from camp_core.atoms.driver_atoms import compute_atom_bank_vector, compute_feasibility_mask
from camp_core.base_predictor.trajectron_loader import (
    TrajectronLoadConfig,
    build_trajectron_adapter_from_checkpoint,
)
from camp_core.mapping_heads.linear_head import LinearMappingHead
from camp_core.outer_master.benders_master import BendersCut
# from camp_core.outer_master.parametric_cvxpy_master import (
#     ParametricCVXPYMaster, 
#     ParametricCVXPYMasterConfig
# )
from camp_core.outer_master.parametric_torch_master import (
    ParametricTorchMaster,
    ParametricTorchMasterConfig
)


def get_top_k_predictions(adapter, batch, k=6, z_mode=False, gmm_mode=True):
    """
    Get K candidate trajectories from Trajectron++.
    Returns list of [K, H, 2] arrays (one per agent in batch).
    """
    trajectron = adapter.base_model
    ph = trajectron.hyperparams.get("prediction_horizon", 12)
    device = next(trajectron.parameters()).device
    batch.to(device)
    
    with torch.no_grad():
        if gmm_mode and k > 1:
             predictions_list = []
             # Sample K modes independently
             for _ in range(k):
                 p_dict = trajectron.predict(
                     batch,
                     prediction_horizon=ph,
                     num_samples=1,
                     z_mode=z_mode,
                     gmm_mode=gmm_mode,
                     output_dists=False
                 )
                 predictions_list.append(p_dict)
        else:
             predictions = trajectron.predict(
                 batch,
                 prediction_horizon=ph,
                 num_samples=k,
                 z_mode=z_mode,
                 gmm_mode=gmm_mode,
                 output_dists=False
             )
             predictions_list = [predictions]
    
    from trajdata import AgentType
    batch_preds = []
    
    for i in range(len(batch.agent_name)):
        node_type = AgentType(batch.agent_type[i].item()) if hasattr(batch, 'agent_type') else 'VEHICLE'
        agent_name = batch.agent_name[i]
        key = f"{str(node_type)}/{agent_name}"
        
        agent_k_preds = []
        for p_dict in predictions_list:
            if key in p_dict:
                 p = p_dict[key] # [1, H, 2]
                 if hasattr(p, 'cpu'): p = p.cpu().numpy()
                 agent_k_preds.append(p)
            else:
                 agent_k_preds.append(np.zeros((1, ph, 2)))
        
        if len(predictions_list) > 1:
            combined = np.concatenate(agent_k_preds, axis=0) # [K, H, 2]
        else:
            combined = agent_k_preds[0]
            
        batch_preds.append(combined)
             
    return batch_preds

def parse_args():
    parser = argparse.ArgumentParser(description="Train CAMP-Select (Selection Only)")
    parser.add_argument("--data_root", type=str, default="/ocean/projects/tra250008p/slin24/datasets/nuscenes")
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.unified_data_cache"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_scenarios", type=int, default=50)
    
    # Model Params
    parser.add_argument("--num_atoms", type=int, default=9, help="Jerk(3)+RMS(1)+Speed(3)+Lane(1)+Clear(1)=9")
    parser.add_argument("--embedding_dim", type=int, default=64)
    
    # Master Params
    parser.add_argument("--risk_type", type=str, default="cvar", choices=["mean", "cvar"])
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_path", type=str, default="models/camp_select_linear.pt")
    
    # Regularization
    parser.add_argument("--prior_reg", type=float, default=1.0, help="Bayesian Trust Region Strength")
    parser.add_argument("--anchor_weight", type=float, default=0.0, help="Offline Anchor Gamma")
    
    # Trajectron
    parser.add_argument("--trajectron_conf", type=str, required=True)
    parser.add_argument("--trajectron_model_dir", type=str, required=True)
    parser.add_argument("--trajectron_epoch", type=int, default=20)
    parser.add_argument("--offline_weights_path", type=str, default="models/offline_weights.npy")
    parser.add_argument("--split", type=str, default="nusc_trainval-train")

    return parser.parse_args()

def load_atom_scales(scale_path: str, device: torch.device) -> torch.Tensor:
    if os.path.exists(scale_path):
        print(f"Loading Atom Scales from {scale_path}", flush=True)
        with open(scale_path, "r") as f:
            scales = json.load(f)
        return torch.tensor(scales, dtype=torch.float32, device=device)
    else:
        print(f"Warning: Scale file {scale_path} not found. Using Identity scales.", flush=True)
        # Default R=9
        return torch.ones(9, dtype=torch.float32, device=device)

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # ... (Data Loading code omitted for brevity) ...
    
    # Load Offline Anchor weights
    if args.anchor_weight > 0 and os.path.exists(args.offline_weights_path):
        prior_weights = np.load(args.offline_weights_path)
        print(f"Loaded Offline Anchor weights from {args.offline_weights_path}")
    else:
        prior_weights = None

    # Load Scales
    atom_scales = load_atom_scales("models/production/atom_scales.json", device)

    # 1. Dataset & Adapter
    print("Initializing Data Bridge...", flush=True)
    # ...
    
    print(f"=== CAMP-Select Training ===")
    print(f"Risk: {args.risk_type}, Alpha: {args.alpha}")
    print(f"Regularization: Prior={args.prior_reg}, Anchor={args.anchor_weight}")
    
    start_time = time.time()
    
    # 1. Init Data
    cfg = NuscenesDatasetConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,
        shuffle=True,
    )
    bridge = NuscenesTrajdataBridge(cfg)
    dataloader = bridge.get_dataloader()
    map_api = bridge.dataset
    
    # 2. Init Trajectron
    traj_cfg = TrajectronLoadConfig(
        conf_path=args.trajectron_conf,
        model_dir=args.trajectron_model_dir,
        epoch=args.trajectron_epoch,
        device=args.device,
    )
    adapter = build_trajectron_adapter_from_checkpoint(
        traj_cfg, 
        embedding_dim=args.embedding_dim,
        mode="encoder"
    )
    adapter.to(device)
    adapter.eval()
    
    # 3. Cache Candidates and Atoms
    print("\n[Phase 1] Caching Candidates and Atoms...")
    
    # Load Scales
    atom_scales = load_atom_scales("models/production/atom_scales.json", device)

    scenarios = [] 
    
    pbar = tqdm(total=args.num_scenarios)
    for batch in dataloader:
        if len(scenarios) >= args.num_scenarios:
            break
            
        # Get embeddings
        with torch.no_grad():
            batch.to(device)
            emb_out = adapter.embed_batch(batch) # [B, D]
            batch_embs = emb_out["scene_embeddings"]
            
        # Get Candidates
        candidates_batch = get_top_k_predictions(adapter, batch, k=12, gmm_mode=True)
        
        B = len(batch.agent_name)
        for i in range(B):
            if len(scenarios) >= args.num_scenarios:
                break
                
            try:
                # Extract Context
                ctx = extract_driver_context(batch, i, map_api=map_api)
                
                # Compute Atoms for Candidates (Robust R=9)
                cands = candidates_batch[i] # [K, H, 2]
                
                atoms_list = []
                feas_list = []
                for k in range(len(cands)):
                    traj = cands[k]
                    at = compute_atom_bank_vector(ctx, traj) # [R]
                    is_f = compute_feasibility_mask(ctx, traj) # bool
                    atoms_list.append(at)
                    feas_list.append(is_f)
                
                atoms_k = np.stack(atoms_list) # [K, R]
                feas_mask_k = np.array(feas_list, dtype=bool) # [K]
                
                # Normalize using global scales (numpy)
                # Ensure scales on CPU
                scales_np = atom_scales.cpu().numpy()
                atoms_k_norm = atoms_k / scales_np
                
                # Robust Clipping (Method 4): Clip to 10.0 to prevent extreme outliers
                # This preserves convexity while stabilizing the Master Problem.
                atoms_k_norm = np.clip(atoms_k_norm, a_min=None, a_max=10.0)
                
                scenarios.append({
                    "id": f"s_{len(scenarios)}",
                    "embedding": batch_embs[i].cpu(), # Keep on CPU until needed, or GPU if fits
                    "atoms": atoms_k_norm, # numpy [K, R] Normalized
                    "feas_mask": torch.tensor(feas_mask_k, dtype=torch.bool), # Store as Tensor
                    "candidates": cands 
                })
                pbar.update(1)
            except Exception as e:
                print(f"Error in skipped scenario: {e}", flush=True)
                pass
                
    pbar.close()
    print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Phase 1 Loop Finished.", flush=True)
    print(f"Cached {len(scenarios)} scenarios.")
    if len(scenarios) == 0:
        print("No scenarios collected. Exiting.")
        return

    # 4. Load Offline Weights & Init Head
    offline_weights = None
    if os.path.exists(args.offline_weights_path):
        offline_weights = np.load(args.offline_weights_path)
        print(f"Loaded Offline Weights: {offline_weights}", flush=True)
    else:
        print("Warning: No offline weights found. Using uniform prior.", flush=True)
        offline_weights = np.ones(args.num_atoms) / args.num_atoms

    # Init Mapping Head (Linear)
    print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Initializing Linear Head...", flush=True)
    mapping_head = LinearMappingHead(
        embedding_dim=args.embedding_dim, 
        num_atoms=args.num_atoms,
        use_bias=True 
    ).to(device)
    
    # 5. Init Master & Precompute Tensors
    print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Stacking embeddings and moving to device...", flush=True)
    all_embeddings = torch.stack([s["embedding"] for s in scenarios]).to(device)
    print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Embeddings shape: {all_embeddings.shape}", flush=True)

    # Pre-stack Atoms and Masks for Vectorized Inner Loop
    print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Pre-stacking Atoms/Masks for {len(scenarios)} scenarios...", flush=True)
    
    # Convert numpy atoms to tensor and stack
    # Note: sc["atoms"] is [K, R] numpy
    all_atoms_tensor = torch.stack([torch.from_numpy(s["atoms"]).float() for s in scenarios]).to(device) # [N, K, R]
    all_masks_tensor = torch.stack([s["feas_mask"] for s in scenarios]).to(device) # [N, K] bool
    
    # 3. Initialize Master (Benders Solver) - GPU Version
    print(f"Initializing GPU Benders Master (Torch)...")
    master_config = ParametricTorchMasterConfig(
        num_atoms=args.num_atoms,
        embedding_dim=args.embedding_dim,
        risk_type=args.risk_type,
        alpha=args.alpha,
        prior_reg_strength=args.prior_reg,
        offline_anchor_weight=args.anchor_weight,
        device=args.device,
        max_iter=50  # Fast PGD iterations
    )
    master = ParametricTorchMaster(
        config=master_config,
        scene_embeddings=all_embeddings, # Renamed from scene_embeddings_tensor
        prior_weights=offline_weights
    )

    print("Starting Benders Decomposition Training (CAMP-Select)...")
    
    # Init Weights (Identity or from Master init which is random)
    # Master solves for Theta -> W.
    # Initially we can push Master's random theta to Head.
    master.update_head_weights(mapping_head, master.theta.detach().cpu().numpy())
    
    start_time = time.time()
    
    for iteration in range(1, 15 + 1): # Fixed 15 epochs for robustness (Renamed from epoch to iteration)
        iter_start = time.time()
        print(f"\n[{time.strftime('%H:%M:%S')}] --- Iteration {iteration} ---", flush=True) # Changed from Epoch to Iteration
        
        # A. Inner Loop (Vectorized)
        print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Starting Inner Loop (Objective Evaluation)...", flush=True)
        
        # A. Get current weights
        mapping_head.eval()
        with torch.no_grad():
            current_raw = mapping_head(all_embeddings) # [N, R] GPU Tensor
            # Softplus alternative? No, paper uses Linear directly?
            # Code used np.maximum(0).
            # Let's stay in Tensor land for speed if possible
            w_curr_t = torch.relu(current_raw)
            sums_t = w_curr_t.sum(dim=1, keepdim=True) + 1e-8
            w_curr_t = w_curr_t / sums_t # [N, R]
            
            # For logging
            mean_w = w_curr_t.mean(dim=0).cpu().numpy()
            
        print(f"Mean Weights: {mean_w}", flush=True)
        
        # B. Inner Step (Vectorized Selection)
        # N = Num Scenarios, K = Num Candidates, R = Num Atoms
        # weights: [N, R] -> [N, 1, R]
        # atoms:   [N, K, R]
        # scores:  [N, K]
        
        scores_raw = (all_atoms_tensor * w_curr_t.unsqueeze(1)).sum(dim=-1) # [N, K]
        
        # 1. Selection (Best Cost) -> Argmin
        # Infeasible candidates should be ignored -> Set to +inf
        scores_sel = scores_raw.clone()
        scores_sel[~all_masks_tensor] = float('inf')
        
        best_vals, _ = torch.min(scores_sel, dim=1) # [N]
        # Handle all-infeasible case (inf) -> use min of raw (fallback)
        all_inf_mask = torch.isinf(best_vals)
        if all_inf_mask.any():
            fallback_vals, _ = torch.min(scores_raw[all_inf_mask], dim=1)
            best_vals[all_inf_mask] = fallback_vals

        min_cost = best_vals.mean().item()
        
        # 2. Cut Generation (Worst Cost) -> Argmax
        # Infeasible candidates should have minimal impact on MAX -> Set to -inf
        scores_cut = scores_raw.clone()
        scores_cut[~all_masks_tensor] = float('-inf')
        
        worst_vals, worst_idxs = torch.max(scores_cut, dim=1) # [N], [N]
        
        # Handle all-infeasible case for cut
        # If all infeasible, cut should be based on worst raw? Or just any?
        # Fallback to worst raw
        all_inf_cut_mask = torch.isinf(worst_vals)
        if all_inf_cut_mask.any():
             # Fallback: argmax of raw scores
             fb_vals, fb_idxs = torch.max(scores_raw[all_inf_cut_mask], dim=1)
             worst_vals[all_inf_cut_mask] = fb_vals
             worst_idxs[all_inf_cut_mask] = fb_idxs
        
        total_q = worst_vals.mean().item()
        
        # Gather Gradients (Atoms of worst candidates)
        # Index into [N, K, R] using [N] indices
        # torch.gather or advanced indexing
        batch_indices = torch.arange(len(scenarios), device=device)
        gradients = all_atoms_tensor[batch_indices, worst_idxs] # [N, R]
        
        # CPU Transfer Phase for Master
        # We process cuts in Python currently. Master.add_cut is sequential store.
        # But this part is lightweight (just object creation).
        
        w_curr_np = w_curr_t.cpu().numpy()
        worst_vals_np = worst_vals.cpu().numpy()
        gradients_np = gradients.cpu().numpy()
        
        cuts_added = 0
        
        # Only iterate to add cuts. (Could be optimized if master accepted batch cuts)
        # But this loop is O(N) simple ops, ~0.1s for 20k.
        for i, sc in enumerate(scenarios):
             cut = BendersCut(
                scenario_id=sc["id"],
                w_anchor=w_curr_np[i],
                value=worst_vals_np[i],
                gradient=gradients_np[i]
             )
             master.add_cut(i, cut)
             cuts_added += 1

        print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Inner Loop Finished (Vectorized).", flush=True)
        print(f"  Inner: Select Avg Max Cost (Robust) = {total_q:.4f}")
        print(f"  Inner: Select Avg Min Cost (Optim)  = {min_cost:.4f}")
        
        # C. Master Step
        print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Starting Master Solve...", flush=True)
        
        target_size = max(2000, int(len(scenarios) * 0.1))
        master_batch_size = min(len(scenarios), target_size)
        
        active_indices = None
        if len(scenarios) > master_batch_size:
            active_indices = np.random.choice(len(scenarios), master_batch_size, replace=False)
            print(f"  [Master] Subsampling {master_batch_size}/{len(scenarios)} scenarios.", flush=True)
            
        res = master.solve(verbose=False, active_indices=active_indices)
        
        if res["status"] in ["optimal", "optimal_inaccurate"]:
            print(f"[{time.strftime('%H:%M:%S')}] [DEBUG] Master Solve Finished. Status: {res['status']}", flush=True)
            print(f"  Master Solved. Loss: {res['loss']:.4f}", flush=True)
            master.update_head_weights(mapping_head, res["Theta"])
            
            # Debug Weights
            with torch.no_grad():
                # GPU Master has phi_aug (tensor). Convert to cpu numpy for debug logic if needed
                # phi_aug is [M, D+1]. The first D columns are embeddings.
                test_emb = master.phi_aug[:5, :-1].cpu().numpy() 
                
                # Theta is [R, D+1]. We want to show W = Theta * phi
                # Just use stored weights
                w_debug = mapping_head(torch.tensor(test_emb).to(device)).detach().cpu().numpy()
                print(f"  [Debug] Head Output (first 2): \n{w_debug[:2]}")
                
        else:
            print(f"  Master Failed: {res['status']}", flush=True)
            if "error" in res:
                print(f"  Error: {res['error']}", flush=True)
            
    # 7. Save
    print("\n[Phase 3] Saving Model...", flush=True)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save({
        "head": mapping_head.state_dict(),
        "config": vars(args),
        "offline_weights": offline_weights
    }, args.output_path)
    
    print(f"Saved to {args.output_path}", flush=True)
    print(f"Total Time: {(time.time() - start_time)/60:.2f} min", flush=True)

if __name__ == "__main__":
    main()
