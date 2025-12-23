
import argparse
import os
import sys
import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from camp_core.data_interfaces.nuscenes_trajdata_bridge import (
    NuscenesDatasetConfig,
    NuscenesTrajdataBridge,
    extract_driver_context,
)
from camp_core.atoms.driver_atoms import compute_atom_bank_vector, compute_aux_metrics, compute_feasibility_mask
from camp_core.base_predictor.trajectron_loader import (
    TrajectronLoadConfig,
    build_trajectron_adapter_from_checkpoint,
)
from camp_core.mapping_heads.linear_head import LinearMappingHead

# Import Reranker Model (Redefine or import if separate file)
# For simplicity, re-defining logic or importing if available.
# We'll assume train_reranker.py is in scripts/train and we can't easily import from script.
# So we define the class here again (common practice for experimental scripts to be self-contained or use common lib).
# Ideally, RerankerModel should be in camp_core.models. Moving forward, let's define it here.

class RerankerModel(torch.nn.Module):
    def __init__(self, embedding_dim, num_atoms, hidden_dim=64):
        super().__init__()
        input_dim = embedding_dim + num_atoms
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    def forward(self, phi, atoms):
        # atoms: [B, K, R]
        B, K, R = atoms.shape
        phi_exp = phi.unsqueeze(1).expand(B, K, -1)
        inp = torch.cat([phi_exp, atoms], dim=-1)
        return self.net(inp).squeeze(-1)


def get_candidates(adapter, batch, k=30, gmm_mode=False):
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
             gmm_mode=gmm_mode,
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

def compute_atoms(candidates, ctx):
    # Use Strict Atom Bank (already normalized)
    atoms_list = []
    for traj in candidates:
        feat = compute_atom_bank_vector(ctx, traj) 
        atoms_list.append(feat)
    return np.array(atoms_list)

def compute_metrics(pred, gt):
    ade = np.mean(np.linalg.norm(pred - gt, axis=1))
    fde = np.linalg.norm(pred[-1] - gt[-1])
    return {"ADE": ade, "FDE": fde}

def load_atom_scales(scale_path: str, device: torch.device) -> torch.Tensor:
    if os.path.exists(scale_path):
        with open(scale_path, "r") as f:
            scales = json.load(f)
        return torch.tensor(scales, dtype=torch.float32, device=device)
    else:
        # Default R=9
        return torch.ones(9, dtype=torch.float32, device=device)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CAMP-Select and Baselines")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_scenarios", type=int, default=100)
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--trajectron_conf", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--trajectron_model_dir", type=str, required=True)
    parser.add_argument("--trajectron_epoch", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--split", type=str, default="nusc_trainval-val") # Valid split
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Models
    print("Loading Models...")
    models = {}
    
    # CAMP Linear
    camp_path = os.path.join(args.models_dir, "camp_select_linear.pt")
    if os.path.exists(camp_path):
        ckpt = torch.load(camp_path, map_location=device)
        # Infer dims from ckpt
        # state_dict keys: linear.weight [R, D]
        w = ckpt["head"]["linear.weight"]
        R, D = w.shape
        model = LinearMappingHead(embedding_dim=D, num_atoms=R, use_bias=True).to(device)
        model.load_state_dict(ckpt["head"])
        model.eval()
        models["CAMP-Select"] = model
        print(f"Loaded CAMP-Select (Linear) from {camp_path}")

    # Reranker GT
    rr_gt_path = os.path.join(args.models_dir, "reranker_gt.pt")
    if os.path.exists(rr_gt_path):
        # Infer dims? Or Assume R=9
        rr_gt = RerankerModel(embedding_dim=64, num_atoms=9).to(device)
        rr_gt.load_state_dict(torch.load(rr_gt_path, map_location=device))
        rr_gt.eval()
        models["Reranker-GT"] = rr_gt
        print(f"Loaded Reranker-GT from {rr_gt_path}")

    # Reranker Safe
    rr_safe_path = os.path.join(args.models_dir, "reranker_safe.pt")
    if os.path.exists(rr_safe_path):
        rr_safe = RerankerModel(embedding_dim=64, num_atoms=9).to(device)
        rr_safe.load_state_dict(torch.load(rr_safe_path, map_location=device))
        rr_safe.eval()
        models["Reranker-Safe"] = rr_safe
        print(f"Loaded Reranker-Safe from {rr_safe_path}")
        
    # Offline Weights (Static)
    off_path = os.path.join(args.models_dir, "offline_weights.npy")
    w_off = None
    if os.path.exists(off_path):
        w_off = np.load(off_path)
        print(f"Loaded Offline Weights")
        
    # 2. Pipeline Init
    cfg = NuscenesDatasetConfig(
        data_root=args.data_root,
        cache_dir=args.models_dir, # Reuse output dir for cache
        batch_size=4,
        num_workers=args.num_workers,
        split="nusc_trainval-val", # validation split
        shuffle=False,
    )
    bridge = NuscenesTrajdataBridge(cfg)
    loader = bridge.get_dataloader()
    
    traj_cfg = TrajectronLoadConfig(
        conf_path=args.trajectron_conf,
        model_dir=args.trajectron_model_dir,
        epoch=args.trajectron_epoch,
        device=str(device)
    )
    adapter = build_trajectron_adapter_from_checkpoint(
        traj_cfg, 
        embedding_dim=64, # Default
        mode="encoder"
    )
    adapter.to(device).eval()
    
    # 3. Eval Loop

# 3. Eval Loop
    results = []
    
    map_api = bridge.dataset

    # Load Scales for Evaluation
    atom_scales = load_atom_scales("models/production/atom_scales.json", device)
    print(f"Loaded Atom Scales: {atom_scales.cpu().numpy()}")
    
    camp_weights_list = []
    
    print("Starting Evaluation...")
    count = 0 
    pbar = tqdm(total=args.num_scenarios)
    print("[DEBUG] Starting Loop...", flush=True)
    
    batch_idx = 0
    for batch in loader:
        if count >= args.num_scenarios: break
        
        print(f"[DEBUG] Batch {batch_idx} loaded. Predicting...", flush=True)
        cands_batch = get_candidates(adapter, batch, k=30) # List of [K, H, 2]
        
        # Prepare embeddings if using CAMP
        if "CAMP-Select" in models or "Reranker-GT" in models or "Reranker-Safe" in models:
            with torch.no_grad():
                batch.to(device)
                emb_out = adapter.embed_batch(batch)
                embs = emb_out["scene_embeddings"] # [B, D]
        
        B = len(batch.agent_name)
        print(f"[DEBUG] Batch {batch_idx}: {B} agents.", flush=True)
        batch_idx += 1
        
        B = len(batch.agent_name)
        for i in range(B):
            if count >= args.num_scenarios: break
            
            gt_raw = batch.agent_fut[i].cpu().numpy()
            if gt_raw.shape[0] < 12: continue
            gt = gt_raw[:12, :2]
            candidates = cands_batch[i] # [K, H, 2]
            K = candidates.shape[0]
            
            try:
                # Need map_api access? NuscenesTrajdataBridge or from batch?
                # extract_driver_context usually needs bridge.dataset or similar map source
                ctx = extract_driver_context(batch, i, map_api=map_api) 
                
                # Compute Atoms & Feasibility (Robust Phase 3)
                cand_atoms_list = []
                cand_feas_list = []
                for k in range(K):
                    traj_k = candidates[k]
                    at = compute_atom_bank_vector(ctx, traj_k)
                    is_f = compute_feasibility_mask(ctx, traj_k)
                    cand_atoms_list.append(at)
                    cand_feas_list.append(is_f)
                    
                atoms = np.stack(cand_atoms_list) # [K, R]
                atoms_t = torch.from_numpy(atoms).float().to(device)
                
                # Normalize
                atoms_t_norm = atoms_t / atom_scales
                
                feas_mask = torch.tensor(cand_feas_list, dtype=torch.bool, device=device)
                
            except Exception as e:
                print(f"Context error: {e}", flush=True)
                continue
                
            # Helper
            def record_result(method_name, k_idx, m_res):
                aux_mets = compute_aux_metrics(ctx, candidates[k_idx])
                results.append({
                    "Method": method_name,
                    "Scene": count,
                    **m_res,
                    "SafetyCost": np.sum(atoms[k_idx]), # Use Raw un-normalized for logging? Or normalized? 
                                                      # Usually raw is more interpretable if units make sense.
                                                      # But model optimizes normalized.
                                                      # Let's log Raw sum for consistency with old metric
                    "Feasible": 1.0 if aux_mets.is_feasible else 0.0,
                    "Progress": aux_mets.progress
                })

            # 1. Pred-Top1
            m_res = compute_metrics(candidates[0], gt)
            record_result("Pred-Top1", 0, m_res)
            
            # 2. Static-Offline
            if w_off is not None:
                # Use Normalized Atoms for Selection
                w_off_t = torch.tensor(w_off, device=device, dtype=torch.float32)
                scores = (atoms_t_norm @ w_off_t)
                
                # Filter by Feasibility
                if feas_mask.any():
                    valid_idx = torch.nonzero(feas_mask).squeeze(1)
                else:
                    valid_idx = torch.arange(K, device=device)
                    
                scores_valid = scores[valid_idx]
                best_local = torch.argmin(scores_valid)
                k_star = valid_idx[best_local].item()
                
                m_res = compute_metrics(candidates[k_star], gt)
                record_result("Static-Offline", k_star, m_res)
                
            # 3. CAMP-Select
            if "CAMP-Select" in models:
                emb = embs[i:i+1] # [1, D]
                with torch.no_grad():
                    theta = models["CAMP-Select"](emb).squeeze(0) # [R] or [R+1]?
                    # LinearHead returns logits? Or LinearHead already includes w_off logic?
                    # LinearHead returns `out`
                    # If using `LinearMappingHead`, output is [R] (weights).
                    # Need to verify if it returns weights or logits.
                    # Usually just linear output.
                    # Need Projection to Simplex?
                    # The Master solves for Theta such that w >= 0, sum(w)=1.
                    # But NN output is unbounded unless we apply activation.
                    # Wait, Master constraints apply to the SOLUTION variable.
                    # The NN is just `Linear`. It learns to output the right values.
                    # But usually we apply ReLU + Norm during Inference to ensure constraints.
                    
                    w_raw = theta.cpu().numpy()
                    w_abs = np.maximum(w_raw, 0)
                    w_sum = np.sum(w_abs) + 1e-8
                    w_camp = w_abs / w_sum
                    w_camp_t = torch.tensor(w_camp, device=device, dtype=torch.float32)
                                        
                    camp_weights_list.append(w_camp)
                    
                scores = (atoms_t_norm @ w_camp_t)
                
                if feas_mask.any():
                    valid_idx = torch.nonzero(feas_mask).squeeze(1)
                else:
                    valid_idx = torch.arange(K, device=device)
                    
                scores_valid = scores[valid_idx]
                best_local = torch.argmin(scores_valid)
                k_star = valid_idx[best_local].item()
                
                m_res = compute_metrics(candidates[k_star], gt)
                record_result("CAMP-Select", k_star, m_res)
                
            # 4. Rerankers
            for name in ["Reranker-GT", "Reranker-Safe"]:
                if name in models:
                    emb = embs[i:i+1] # [1, D]
                    atoms_t = torch.from_numpy(atoms).float().to(device).unsqueeze(0) # [1, K, R]
                    with torch.no_grad():
                        scores = models[name](emb, atoms_t).cpu().numpy()[0] # [K]
                    k_star = np.argmin(scores)
                    m_res = compute_metrics(candidates[k_star], gt)
                    record_result(name, k_star, m_res)
                
            count += 1
            pbar.update(1)
            
    pbar.close()
    
    # Summary
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    
    print("\n=== Results Summary ===")
    print(df.groupby("Method")[["ADE", "FDE", "SafetyCost", "Feasible", "Progress"]].mean())
    print("\nCVaR (SafetyCost, lower is better):")
    
    # CVaR (alpha=0.9 of Cost)
    def compute_cvar(x, alpha=0.9):
        # x is Cost. High cost is bad. Tail is right tail.
        # VaR = quantile(alpha)
        x = np.array(x)
        if len(x) == 0: return 0.0
        var = np.percentile(x, alpha * 100)
        cvar = x[x >= var].mean()
        return cvar
        
    for name, group in df.groupby("Method"):
        cvar = compute_cvar(group["SafetyCost"])
        print(f"{name}: {cvar:.4f}")

    print("\n=== Feature Weights Analysis ===")
    if w_off is not None:
        print(f"Static-Offline Weights: {w_off}")
        
    if len(camp_weights_list) > 0:
        avg_w = np.mean(np.vstack(camp_weights_list), axis=0)
        print(f"CAMP-Select Mean Weights: {avg_w}")
        
    print("Reranker-GT/Safe: Non-linear MLP (No explicit weights)")

if __name__ == "__main__":
    main()
