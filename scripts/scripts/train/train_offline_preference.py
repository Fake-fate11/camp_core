import json
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gc
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from camp_core.data_interfaces.nuscenes_trajdata_bridge import (
    NuscenesDatasetConfig,
    NuscenesTrajdataBridge,
    extract_driver_context,
)
from camp_core.inner_solver.driver_solver import DriverAwareInnerSolver
from camp_core.atoms.driver_atoms import DriverAtomContext, compute_atom_bank_vector

def parse_args():
    parser = argparse.ArgumentParser(description="Train Offline Preference (BTL) for CAMP")
    parser.add_argument("--data_root", type=str, default="/ocean/projects/tra250008p/slin24/datasets/nuscenes", help="Path to nuScenes dataset")
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.unified_data_cache"), help="Path to cache directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--num_candidates", type=int, default=10, help="Number of candidate trajectories per scene")
    parser.add_argument("--num_scenarios", type=int, default=50, help="Number of scenarios to use")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--l1_reg", type=float, default=0.01, help="L1 regularization strength")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output_path", type=str, default="models/offline_weights.npy", help="Path to save weights")
    return parser.parse_args()

def load_atom_scales(scale_path: str, device: torch.device) -> torch.Tensor:
    if os.path.exists(scale_path):
        print(f"Loading Atom Scales from {scale_path}")
        with open(scale_path, "r") as f:
            scales = json.load(f)
        return torch.tensor(scales, dtype=torch.float32, device=device)
    else:
        print(f"Warning: Scale file {scale_path} not found. Using Identity scales.")
        return torch.ones(9, dtype=torch.float32, device=device)

def generate_candidates(
    inner_solver: DriverAwareInnerSolver,
    ctx: DriverAtomContext,
    y0: np.ndarray,
    num_candidates: int
) -> List[np.ndarray]:
    """
    Generate candidate trajectories by solving the inner problem with random weights.
    Returns:
        candidates: List of [H, 2] trajectories
    """
    candidates = []
    
    alphas = np.ones(len(inner_solver.atoms)) * 0.5
    weights_samples = np.random.dirichlet(alphas, size=num_candidates)
    
    for w in weights_samples:
        y_opt, _, _, success = inner_solver.solve(y0, w, ctx)
        if success:
            candidates.append(y_opt)
            
    return candidates

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # 1. Data Loading
    print("Initializing Data Loader...")
    cfg = NuscenesDatasetConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False, # [DEBUG] Shuffle causes hang on large datasets?
    )
    bridge = NuscenesTrajdataBridge(cfg)
    dataloader = bridge.get_dataloader()
    
    # [INFO] Log dataset size
    if hasattr(dataloader, 'dataset'):
         print(f"[INFO] Total training samples in dataset: {len(dataloader.dataset)}")
    print(f"[INFO] Total training batches: {len(dataloader)}")
    
    map_api = bridge.dataset
    
    # Load Scales
    atom_scales_t = load_atom_scales("models/production/atom_scales.json", device)
    atom_scales = atom_scales_t.cpu().numpy()
    
    # 2. Initialize Solver
    inner_solver = DriverAwareInnerSolver(horizon=12, dt=0.5)
    
    # 3. Collect Data (GT and Candidates)
    print("Collecting data and generating candidates...")
    dataset_features = [] # List of (gt_atoms, candidate_atoms) [R=9]
    
    scenarios_collected = 0
    print(f"Total Scenarios to Collect: {args.num_scenarios}", flush=True)
    
    print("Starting batch loop...", flush=True)
    count = 0
    for batch in dataloader:
        if scenarios_collected >= args.num_scenarios:
            break
            
        B = batch.curr_agent_state.shape[0]
        for i in range(B):
            if scenarios_collected >= args.num_scenarios:
                break
                
            # Future
            fut = batch.agent_fut[i].cpu().numpy()
            fut_xy = fut[:11, :2]
            if np.isnan(fut_xy).any(): continue
            curr_pos = np.zeros(2) # local frame
            gt_traj = np.concatenate(([curr_pos], fut_xy), axis=0) # [H, 2]
            horizon = len(gt_traj)

            try:
                # Context
                ctx = extract_driver_context(batch, i, map_api=map_api, horizon=horizon)
            except Exception as e:
                continue
            
            # --- Robust Atoms (R=9) ---
            # 1. GT Atoms
            gt_atoms = compute_atom_bank_vector(ctx, gt_traj) # [R]
            
            # 2. Candidates
            candidates = generate_candidates(inner_solver, ctx, curr_pos, args.num_candidates)
            if len(candidates) == 0: continue
            
            cand_atoms_list = []
            for c in candidates:
                at = compute_atom_bank_vector(ctx, c)
                cand_atoms_list.append(at)
            
            cand_atoms = np.stack(cand_atoms_list) # [K, R]
            
            # Normalize HERE (using global scales)
            gt_norm = gt_atoms / atom_scales
            cand_norm = cand_atoms / atom_scales
            
            # Basic validation
            if np.isnan(gt_norm).any() or np.isnan(cand_norm).any(): continue

            dataset_features.append((gt_norm, cand_norm))
            scenarios_collected += 1
            if scenarios_collected % 100 == 0:
                 print(f"Collected: {scenarios_collected}/{args.num_scenarios}", flush=True)
        
        del batch
        torch.cuda.empty_cache()
        gc.collect()
        
    print(f"Collected {len(dataset_features)} scenarios.")
    if len(dataset_features) == 0:
        print("ERROR: No valid scenarios collected.")
        return
    
    # 4. Train Offline Weights (BTL)
    print("Training offline weights (Robust R=9)...")
    
    NUM_ATOMS = 9
    w_logits = nn.Parameter(torch.zeros(NUM_ATOMS, device=device))
    optimizer = optim.Adam([w_logits], lr=args.lr)
    
    # Iterate
    for epoch in range(args.epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        
        # Softmax? Or ReLu + Simplex?
        # BTL usually assumes w > 0.
        # Softmax enforces sum=1, >0.
        w = torch.softmax(w_logits, dim=0)
        
        # Batch? Or full batch?
        # Full batch for stability usually ok if N < 20000.
        # But for speed, let's just do full pass accumulation.
        
        # To tensor
        # Optimize: Pre-stack
        # Due to variable K candidates, cannot fully stack.
        # But if K is constant, we can. generate_candidates might return variable K if failures.
        # Let's verify. Inner solver failures -> variable K.
        # So keep loop.
        
        loss_list = []
        for gt_feat, cand_feats in dataset_features:
            gt_t = torch.tensor(gt_feat, dtype=torch.float32, device=device)
            cands_t = torch.tensor(cand_feats, dtype=torch.float32, device=device)
            
            # Score = <w, A> (Lower is better cost? Or Higher is better utility?)
            # Atoms are COST. We want to MINIMIZE cost.
            # BTL model: P(y > x) = exp(-C(y)) / (exp(-C(y)) + exp(-C(x)))
            #          = 1 / (1 + exp(C(y) - C(x)))
            # Here we maximize likelihood of GT being chosen over candidates.
            # GT is y. Candidates x_i.
            # We want C(gt) < C(cand).
            # Diff = C(gt) - C(cand). We want Diff < 0.
            # Loss = log(1 + exp(Diff)) ? Softplus(Diff).
            
            # Delta = A_gt - A_cand
            delta = gt_t.unsqueeze(0) - cands_t # [K, R]
            diffs = delta @ w # [K]
            
            # Loss sum softplus
            # Mean over K?
            l = torch.nn.functional.softplus(diffs).mean()
            loss_list.append(l)
        
        final_loss = torch.stack(loss_list).mean() + args.l1_reg * torch.norm(w, 1)
        
        final_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {final_loss.item():.4f}")
            # print(f"Weights: {w.detach().cpu().numpy()}")

    final_w = torch.softmax(w_logits, dim=0).detach().cpu().numpy()
    print(f"Final Weights (R=9): {final_w}")
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(args.output_path, final_w)
    print(f"Saved weights to {args.output_path}")

if __name__ == "__main__":
    main()
