
import argparse
import os
# Force sync execution for debugging CUDA errors
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from camp_core.data_interfaces.nuscenes_trajdata_bridge import (
    NuscenesDatasetConfig,
    NuscenesTrajdataBridge,
    extract_driver_context,
)
from camp_core.atoms.driver_atoms import compute_atom_bank_vector
from camp_core.base_predictor.trajectron_loader import (
    TrajectronLoadConfig,
    build_trajectron_adapter_from_checkpoint,
)

class RerankerModel(nn.Module):
    """
    Score s(xi, y) = MLP([phi(xi), A(xi, y)])
    Output scalar score (lower is better).
    """
    def __init__(self, embedding_dim, num_atoms, hidden_dim=64):
        super().__init__()
        input_dim = embedding_dim + num_atoms
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, phi, atoms):
        # phi: [B, D]
        # atoms: [B, K, R] or [B, R]
        
        if atoms.dim() == 3:
            # Broadcast phi for K candidates
            B, K, R = atoms.shape
            phi_exp = phi.unsqueeze(1).expand(B, K, -1) # [B, K, D]
            inp = torch.cat([phi_exp, atoms], dim=-1) # [B, K, D+R]
            score = self.net(inp).squeeze(-1) # [B, K]
        else:
            # Single pair
            inp = torch.cat([phi, atoms], dim=-1)
            score = self.net(inp).squeeze(-1)
            
        return score

def compute_ade(pred, gt):
    # pred: [T, 2]
    # gt: [T, 2]
    return np.mean(np.linalg.norm(pred - gt, axis=1))

def get_top_k_predictions(adapter, batch, k=12, gmm_mode=True):
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
             p = predictions[key] # [K, H, 2]
             if hasattr(p, 'cpu'): p = p.cpu().numpy()
             batch_preds.append(p)
        else:
             batch_preds.append(np.zeros((k, ph, 2)))
             
    return batch_preds

def load_atom_scales(scale_path: str, device: torch.device) -> torch.Tensor:
    if os.path.exists(scale_path):
        print(f"Loading Atom Scales from {scale_path}")
        with open(scale_path, "r") as f:
            scales = json.load(f)
        return torch.tensor(scales, dtype=torch.float32, device=device)
    else:
        return torch.ones(9, dtype=torch.float32, device=device)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Reranker Baseline")
    parser.add_argument("--data_root", type=str, default="/ocean/projects/tra250008p/slin24/datasets/nuscenes")
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.unified_data_cache"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_scenarios", type=int, default=100)
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_safe", type=float, default=0.0, help="Safety Regularization Strength")
    parser.add_argument("--safety_temp", type=float, default=1.0, help="Temperature for Soft Selection")
    
    parser.add_argument("--output_path", type=str, default="models/reranker_gt.pt")
    
    # Trajectron
    parser.add_argument("--trajectron_conf", type=str, required=True)
    parser.add_argument("--trajectron_model_dir", type=str, required=True)
    parser.add_argument("--trajectron_epoch", type=int, default=20)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--split", type=str, default="nusc_trainval-train")

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=== Reranker Training ===")
    print(f"Epochs: {args.epochs}, Lambda Safe: {args.lambda_safe}")
    
    # 1. Pipeline Init
    cfg = NuscenesDatasetConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,
        shuffle=False # [DEBUG] Disable shuffle to prevent hang
    )
    bridge = NuscenesTrajdataBridge(cfg)
    dataloader = bridge.get_dataloader()
    map_api = bridge.dataset
    
    # 2. Collect Data (Online or Cache?)
    # Data Checkpointing
    data_cache_path = os.path.join(os.path.dirname(args.output_path), "reranker_train_data.pt")
    
    # Load Scales
    atom_scales_t = load_atom_scales("models/production/atom_scales.json", device)
    atom_scales = atom_scales_t.cpu().numpy()
    
    if os.path.exists(data_cache_path):
        print(f"[INFO] Loading cached training data from {data_cache_path}...", flush=True)
        train_data = torch.load(data_cache_path)
        print(f"[INFO] Loaded {len(train_data)} samples.", flush=True)
    else:
        # Initialize Trajectron ONLY if we need to collect data
        print(f"[INFO] Cache not found. Initializing Trajectron for data collection...", flush=True)
        traj_cfg = TrajectronLoadConfig(
            conf_path=args.trajectron_conf,
            model_dir=args.trajectron_model_dir,
            epoch=args.trajectron_epoch,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        adapter = build_trajectron_adapter_from_checkpoint(
            traj_cfg, 
            embedding_dim=args.embedding_dim,
            mode="encoder"
        )
        adapter.eval()

        train_data = []
        print(f"[DEBUG] Starting loop. Goal: {args.num_scenarios}", flush=True)
        batch_idx = 0
        for batch in dataloader:
            if len(train_data) >= args.num_scenarios:
                break
                
            print(f"[DEBUG] Batch {batch_idx} loaded.", flush=True)
            
            try:
                with torch.no_grad():
                    batch.to(device)
                    print(f"[DEBUG] Embedding batch {batch_idx}...", flush=True)
                    t0 = time.time()
                    emb_out = adapter.embed_batch(batch)
                    torch.cuda.synchronize()
                    embs = emb_out["scene_embeddings"]
                    print(f"[DEBUG] Embedding done in {time.time()-t0:.2f}s", flush=True)
                    
                print(f"[DEBUG] Predicting k={12}...", flush=True)
                t0 = time.time()
                k_preds = get_top_k_predictions(adapter, batch, k=12, gmm_mode=False)
                torch.cuda.synchronize()
                print(f"[DEBUG] Prediction done in {time.time()-t0:.2f}s", flush=True)
            except RuntimeError as e:
                print(f"[ERROR] CUDA Error during Trajectron inference in Batch {batch_idx}: {e}", flush=True)
                print("[DEBUG] Skipping this batch and terminating collection to save progress.", flush=True)
                break
            
            B = len(batch.agent_name)
            for i in range(B):
                if len(train_data) >= args.num_scenarios:
                    break
                    
                try:
                    # GT Trajectory
                    gt_raw = batch.agent_fut[i].cpu().numpy() # [H_gt, D]
                    # Slice to match prediction horizon (12) and XY dims (2)
                    if gt_raw.shape[0] < 12:
                         continue
                    gt = gt_raw[:12, :2]
                    
                    # Context & Atoms
                    ctx = extract_driver_context(batch, i, map_api=map_api)
                    preds = k_preds[i] # [K, H, 2]
                    
                    # Compute Atoms [K, R]
                    atoms_list = []
                    for traj in preds:
                        feat = compute_atom_bank_vector(ctx, traj) 
                        atoms_list.append(feat)
                    
                    atoms_np = np.stack(atoms_list) # [K, R]
                    
                    # NORMALIZE
                    atoms_np = atoms_np / atom_scales
                    
                    # Labels (ADE)
                    ades = np.array([compute_ade(p, gt) for p in preds])
                    
                    train_data.append({
                        "embedding": embs[i].cpu().numpy(),
                        "atoms": atoms_np,
                        "ades": ades
                    })
                    print(f"[DEBUG] Collected Sample {len(train_data)}/{args.num_scenarios}", flush=True)
                except Exception as e:
                    print(f"[ERROR] Failed to process agent {i}: {e}", flush=True)
                    pass
            
            batch_idx += 1
            
        # Checkpoint immediately after collection
        os.makedirs(os.path.dirname(data_cache_path), exist_ok=True)
        print(f"[INFO] Saving training data to {data_cache_path}...", flush=True)
        torch.save(train_data, data_cache_path)
        print(f"[INFO] Checkpoint saved. Run the script again if it crashes now.", flush=True)
    
    if len(train_data) == 0:
        print("No training data collected.")
        return

    # 3. Model
    sample = train_data[0]
    D = sample["embedding"].shape[0]
    R = sample["atoms"].shape[1]
    
    model = RerankerModel(embedding_dim=D, num_atoms=R).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Define Safe Weights for Regularization
    # Atoms R=9:
    # 0-2: Jerk (3)
    # 3: RMS (1)
    # 4-6: Speed (3)
    # 7: Lane (1)
    # 8: Clearance (1)
    
    w_safe = torch.zeros(R, device=device)
    # Penalize Safety Violations
    if R >= 9:
        w_safe[4] = 1.0 # Speed 0.0 margin
        w_safe[5] = 1.0 # Speed 0.5 margin
        w_safe[6] = 1.0 # Speed 1.0 margin
        w_safe[7] = 1.0 # Lane Deviation
        w_safe[8] = 1.0 # Clearance (Soft)
    elif R >= 4:
         # Fallback for old bank
         w_safe[3] = 1.0
    
    # 4. Train Loop
    print("Starting Training...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        # Shuffle
        np.random.shuffle(train_data)
        
        for sc in train_data:
            # Prepare Inputs
            phi = torch.from_numpy(sc["embedding"]).float().to(device).unsqueeze(0) # [1, D]
            atoms = torch.from_numpy(sc["atoms"]).float().to(device).unsqueeze(0) # [1, K, R]
            ades = sc["ades"]
            
            # --- Sampling Negatives ---
            k_best = np.argmin(ades)
            
            # Sample Hard Negative: Not best, but low ADE
            sorted_idx = np.argsort(ades)
            candidates = sorted_idx[1:min(5, len(ades))]
            if len(candidates) > 0:
                k_hard = np.random.choice(candidates)
            else:
                k_hard = np.random.choice([x for x in range(len(ades)) if x != k_best])
                
            # Random Negative
            k_rand = np.random.randint(0, len(ades))
            while k_rand == k_best:
                k_rand = np.random.randint(0, len(ades))
                
            # Pick one negative strategy (mixed)
            if np.random.rand() < 0.5:
                k_neg = k_hard
            else:
                k_neg = k_rand
                
            # --- Forward ---
            optimizer.zero_grad()
            
            # Calculate scores for ALL candidates (needed for Softmax Reg)
            scores = model(phi, atoms) # [1, K]
            
            s_pos = scores[0, k_best]
            s_neg = scores[0, k_neg]
            
            # Rank Loss
            rank_loss = torch.nn.functional.softplus(s_pos - s_neg)
            
            # Safety Reg
            reg_loss = 0.0
            if args.lambda_safe > 0:
                # E [Q_safe]
                # Q_safe = <w_safe, A>
                # atoms: [1, K, R]
                q_safe = (atoms * w_safe).sum(dim=-1) # [1, K]
                
                # Softmax selection
                # pi_k propto exp(-s_k / tau)
                probs = torch.softmax(-scores / args.safety_temp, dim=1) # [1, K]
                expected_safety_cost = (probs * q_safe).sum()
                
                reg_loss = args.lambda_safe * expected_safety_cost
                
            loss = rank_loss + reg_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(train_data):.4f}")
        
    # 5. Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_path)
    print(f"Saved Reranker to {args.output_path}")

if __name__ == "__main__":
    main()
