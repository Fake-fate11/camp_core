
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import time
import json
import numpy as np
import torch
from tqdm import tqdm
import gc

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from camp_core.data_interfaces.nuscenes_trajdata_bridge import (
    NuscenesDatasetConfig,
    NuscenesTrajdataBridge,
    extract_driver_context,
)
from camp_core.inner_solver.driver_solver import DriverAwareInnerSolver
from camp_core.atoms.driver_atoms import DriverAtomContext, compute_driver_atom_features
from camp_core.base_predictor.trajectron_loader import (
    TrajectronLoadConfig,
    build_trajectron_adapter_from_checkpoint,
)
from camp_core.mapping_heads.hyper_network import HyperNetworkMappingHead
from camp_core.outer_master.benders_master import BendersCut
from camp_core.outer_master.parametric_master import ParametricBendersMaster, ParametricMasterConfig

# Helper to evaluate cost (Consistent with compare_methods.py)
def evaluate_cost(solver, trajectory, weights, context):
    """
    Evaluate the cost of a trajectory given weights using the solver's atoms.
    """
    # Compute atom features for the trajectory
    feats = compute_driver_atom_features(context, trajectory)
    raw_vals = feats.as_vector()
    
    # Normalize using context scales (matching solver's normalization)
    scale_keys = [
        "jerk_scale", "smooth_scale", "corridor_scale",
        "speed_limit_scale", "progress_scale", "clearance_scale"
    ]
    
    normalized_vals = []
    # Assumes solver.atom_names order matches scale_keys order logic in driver_solver.py
    # DriverAwareInnerSolver.atoms init order: Jerk, Smoothness, LaneDeviation, SpeedLimit, Progress, Clearance
    # scale_keys in evaluate_cost must match this order.
    
    for i in range(len(weights)):
        val = raw_vals[i]
        # Context might differ slightly in structure (obj vs dict)
        # extract_driver_context returns a DriverAtomContext object
        # getattr usually works.
        scale = getattr(context, scale_keys[i], 1.0)
        if scale < 1e-6:
            scale = 1.0
        normalized_vals.append(val / scale)
    
    # Weighted cost
    cost = np.dot(weights, normalized_vals)
    return cost

# Helper function for predictions (Consistent with compare_methods.py)
# Helper function for predictions (Consistent with compare_methods.py)
def get_top_k_predictions(adapter, batch, k=6, z_mode=False, gmm_mode=True):
    trajectron = adapter.base_model
    ph = trajectron.hyperparams.get("prediction_horizon", 12)
    device = next(trajectron.parameters()).device
    batch.to(device) # In-place
    
    with torch.no_grad():
        if gmm_mode and k > 1:
             # Trajectron assert fails if gmm_mode=True and num_samples > 1.
             # Workaround: Call predict K times with num_samples=1
             # This samples Z each time (if z_mode=False) and gets the Mode Y.
             predictions_list = []
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
                 
             # Merge logic: We need to stack for each agent.
             # This is slightly expensive but robust.
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
        
        # Concat along dimension 0 (Samples)
        # Each p is [1, H, 2], stacking K gives [K, H, 2]
        # Or if k=1 in else block, we have list of 1 array of shape [K, H, 2]
        
        if len(predictions_list) > 1:
            # We looped K times, each result is [1, H, 2]
            combined = np.concatenate(agent_k_preds, axis=0) # [K, H, 2]
        else:
            # We ran once, result is [K, H, 2]
            combined = agent_k_preds[0]
            
        batch_preds.append(combined)
             
    return batch_preds

def parse_args():
    parser = argparse.ArgumentParser(description="Train CAMP Level-A (Parametric Benders) on nuScenes")
    parser.add_argument("--data_root", type=str, default="/ocean/projects/tra250008p/slin24/datasets/nuscenes", help="Path to nuScenes dataset")
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.unified_data_cache"), help="Path to cache directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--num_scenarios", type=int, default=50, help="Number of scenarios to use")
    parser.add_argument("--num_atoms", type=int, default=6, help="Number of atoms")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for mapping head")
    parser.add_argument("--risk_type", type=str, default="cvar", choices=["mean", "cvar"], help="Risk type")
    parser.add_argument("--alpha", type=float, default=0.9, help="CVaR alpha")
    parser.add_argument("--prior_reg_strength", type=float, default=10.0, help="Strength of prior regularization")
    parser.add_argument("--max_iter", type=int, default=20, help="Maximum Benders iterations")
    parser.add_argument("--master_steps", type=int, default=100, help="Number of GD steps for Master per iteration")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for Master")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save model")
    
    # Trajectron
    parser.add_argument("--trajectron_conf", type=str, required=True, help="Path to Trajectron config")
    parser.add_argument("--trajectron_model_dir", type=str, required=True, help="Path to Trajectron model directory")
    parser.add_argument("--trajectron_epoch", type=int, default=20)
    parser.add_argument("--offline_weights_path", type=str, default="models/offline_weights.npy", 
                        help="Path to offline learned weights (from Stage1) for initialization")
    parser.add_argument("--load_offline_weights", type=str, default=None, help="Alias for offline_weights_path")
    parser.add_argument("--epochs", type=int, default=None, help="Alias for max_iter (number of outer Benders iterations)")
    parser.add_argument("--split", type=str, default="nusc_trainval-train", help="Data split")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Handle aliases
    # if args.epochs is not None:
    #     args.max_iter = args.epochs
    
    if args.load_offline_weights is not None:
        args.offline_weights_path = args.load_offline_weights

    device = torch.device(args.device)
    
    # Record training start time
    training_start_time = time.time()
    
    # 1. Data Loading
    print("Initializing Data Loader...")
    cfg = NuscenesDatasetConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,
        shuffle=True, # Shuffle to get random scenarios
    )
    bridge = NuscenesTrajdataBridge(cfg)
    dataloader = bridge.get_dataloader()
    
    # [INFO] Log dataset size
    if hasattr(dataloader, 'dataset'):
         print(f"[INFO] Total training samples in dataset: {len(dataloader.dataset)}")
    print(f"[INFO] Total training batches: {len(dataloader)}")
    
    map_api = bridge.dataset
    
    # 2. Initialize Models
    print("Initializing Models...")
    
    # Trajectron Adapter (Frozen)
    traj_cfg = TrajectronLoadConfig(
        conf_path=args.trajectron_conf,
        model_dir=args.trajectron_model_dir,
        epoch=args.trajectron_epoch,
        device=args.device,
    )
    # Note: We use "encoder" mode to get embeddings
    adapter = build_trajectron_adapter_from_checkpoint(
        load_cfg=traj_cfg,
        embedding_dim=args.embedding_dim,
        mode="encoder",
    )
    adapter.to(device)
    adapter.eval()
    
    # Mapping Head (Trainable) - Upgraded to MLP
    mapping_head = HyperNetworkMappingHead(
        embedding_dim=args.embedding_dim,
        num_atoms=args.num_atoms,
        hidden_dims=[args.hidden_dim, args.hidden_dim]
    ).to(device)
    
    # Load offline weights for initialization (from Stage 1)
    if os.path.exists(args.offline_weights_path):
        offline_weights = np.load(args.offline_weights_path)
        print(f"Loaded offline weights from {args.offline_weights_path}")
        print(f"Offline weights: {offline_weights}")
        
        # Initialize mapping head to output offline weights
        # Strategy for MLP: 
        # Set final layer bias to log(w_off) and final layer weights to near zero.
        with torch.no_grad():
            offline_weights_tensor = torch.from_numpy(offline_weights).float().to(device)
            # Inverse softmax: approximate logits
            logits = torch.log(offline_weights_tensor + 1e-8)
            
            # Access MLP layers
            # Structure: self.mlp = nn.Sequential(Linear, ReLU, Linear, ReLU, ..., Linear)
            # The last layer is at index -1
            if hasattr(mapping_head, 'mlp'):
                last_layer = mapping_head.mlp[-1]
                if isinstance(last_layer, torch.nn.Linear):
                    last_layer.bias.data = logits
                    last_layer.weight.data.normal_(0, 1e-4) # Small random weights
                    print("Initialized MLP head with offline weights (Bias trick)")
    else:
        print(f"No offline weights found at {args.offline_weights_path}, using random initialization")
    
    # Inner Solver
    inner_solver = DriverAwareInnerSolver(horizon=12, dt=0.5)
    
    # 3. Collect Scenarios (Fixed Set)
    print(f"Collecting {args.num_scenarios} scenarios...")
    scenarios = [] # List of (ctx, y0, embedding)
    
    scenarios_collected = 0
    pbar = tqdm(total=args.num_scenarios)
    
    # We iterate until we collect enough valid scenarios
    for batch in dataloader:
        if scenarios_collected >= args.num_scenarios:
            break
            
        # Get embeddings        # [B, 5]
        # Get Candidates for Warm Start (using consistent strategy)
        k_trajs_batch = get_top_k_predictions(adapter, batch, k=6, z_mode=False, gmm_mode=True)
        
        with torch.no_grad():
            adapter_out = adapter.embed_batch(batch)
            batch_embeddings = adapter_out["scene_embeddings"].to(device) # [B, D]
            
            # Check for NaNs in embeddings
            if torch.isnan(batch_embeddings).any():
                print(f"[WARNING] Batch embeddings contain NaNs! Skipping batch.")
                continue
            
        B = batch.curr_agent_state.shape[0]
        for i in range(B):
            if scenarios_collected >= args.num_scenarios:
                break
                
            # Check validity first
            curr_state = batch.curr_agent_state[i].cpu().numpy()
            if np.isnan(curr_state).any():
                continue
                
            try:
                ctx = extract_driver_context(batch, i, map_api=map_api)
                
                # Initial state y0
                # We work in the agent's local frame, so current position is always [0, 0]
                curr_state = batch.curr_agent_state[i].cpu().numpy()
                if np.isnan(curr_state).any():
                    continue
                y0 = np.zeros(2)
                
                # [Fix] Update context desired_speed based on Top-1 Prediction
                # This is critical for Benders Decomposition to optimize moving trajectories
                top1 = k_trajs_batch[i][0]
                pred_disp = np.linalg.norm(top1[-1] - top1[0])
                pred_speed = pred_disp / ((len(top1) - 1) * 0.5)
                if pred_speed > 0.5:
                     ctx.desired_speed = float(pred_speed)
                
                emb = batch_embeddings[i]
                
                scenarios.append({
                    "ctx": ctx,
                    "y0": y0,
                    "embedding": emb,
                    "candidates": k_trajs_batch[i],
                    "id": f"s_{scenarios_collected}"
                })
                scenarios_collected += 1
                pbar.update(1)
            except Exception as e:
                # Skip if extraction fails
                continue
        
        # Memory cleanup
        del batch
        del adapter_out
        del batch_embeddings
        torch.cuda.empty_cache()
        gc.collect()
                
    pbar.close()
    
    # Stack embeddings for Master
    all_embeddings = torch.stack([s["embedding"] for s in scenarios]) # [M, D]
    
    # Load Offline Weights for Trust Region (Prior)
    offline_weights_path = args.offline_weights_path
    prior_weights = None
    if os.path.exists(offline_weights_path):
        try:
            prior_weights = np.load(offline_weights_path)
            print(f"Loaded Offline Prior Weights: {prior_weights}")
        except Exception as e:
            print(f"Failed to load offline weights: {e}")
    else:
        print(f"Warning: No offline weights found at {offline_weights_path}. Trust Region disabled.")
    
    # 4. Initialize Master
    master_config = ParametricMasterConfig(
        num_atoms=args.num_atoms,
        risk_type=args.risk_type,
        alpha=args.alpha,
        lr=args.lr,
        num_steps=args.master_steps,
        prior_reg_strength=args.prior_reg_strength, 
    )
    master = ParametricBendersMaster(
        config=master_config, 
        mapping_head=mapping_head, 
        scene_embeddings=all_embeddings,
        prior_weights=prior_weights
    )
    
    # 5. Benders Loop
    print(f"Starting Benders Loop ({args.max_iter} iterations)...")
    
    for iteration in range(args.max_iter):
        print(f"\n=== Iteration {iteration} ===")
        
        # A. Inner Step: Solve and Generate Cuts
        total_inner_cost = 0.0
        
        # Get current weights from mapping head
        master.mapping_head.eval()
        with torch.no_grad():
            current_weights = master.mapping_head(all_embeddings).cpu().numpy() # [M, R]
            
        # Check for NaNs in weights
        if np.isnan(current_weights).any():
            print(f"[WARNING] Current weights contain NaNs! Replacing with uniform weights.")
            current_weights = np.nan_to_num(current_weights, nan=1.0/args.num_atoms)
            
        print(f"  Avg Weights: {current_weights.mean(axis=0)}")
        
        cuts_added = 0
        
        for i, scenario in enumerate(tqdm(scenarios, desc="Inner Solve")):
            ctx = scenario["ctx"]
            # Debug context for first scenario
            if i == 0:
                 d_speed = getattr(ctx, "desired_speed", "N/A")
                 print(f"  [Debug] Scenario 0 desired_speed: {d_speed}")
            
            y0 = scenario["y0"]
            w_i = current_weights[i]
            
            # Warm Start Selection (using consistent candidates)
            warm_start_traj = None
            if "candidates" in scenario:
                 candidates = scenario["candidates"] # [K, H, 2]
                 if len(candidates) > 0:
                     best_k_idx = 0
                     min_k_cost = float("inf")
                     for k in range(len(candidates)):
                         c = evaluate_cost(inner_solver, candidates[k], w_i, ctx)
                         if c < min_k_cost:
                             min_k_cost = c
                             best_k_idx = k
                     warm_start_traj = candidates[best_k_idx]
            
            # Solve Inner
            y_opt, atom_vals, cost, success = inner_solver.solve(y0, w_i, ctx, warm_start_traj=warm_start_traj)
            
            if success:
                # Create Cut
                cut = BendersCut(
                    scenario_id=scenario["id"],
                    w_anchor=w_i,
                    value=cost,
                    gradient=atom_vals
                )
                master.add_cut(i, cut)
                cuts_added += 1
                total_inner_cost += cost
            else:
                # If inner solver fails, we can't generate a cut.
                # In robust optimization, this might mean infinite cost?
                # For now, just skip.
                pass
                
        print(f"  Inner Solves Successful: {cuts_added}/{len(scenarios)}")
        print(f"  Total Inner Cost: {total_inner_cost:.4f}")
        
        # B. Master Step: Update Theta
        print("  Solving Master...")
        master_result = master.solve(verbose=True)
        print(f"  Master Loss: {master_result['loss']:.4f}")
        
        # Check Convergence (Optional)
        # We can check if Master Loss matches Inner Cost (Gap)
        # Or if Theta changes are small.
        
    # 6. Save Model
    # 6. Save Model
    print("Saving model...")
    # os.makedirs(args.output_dir, exist_ok=True) # Already created at start
    model_save_path = os.path.join(args.output_dir, "camp_level_a.pt")
    
    torch.save({
        "head": mapping_head.state_dict(),
        "head_type": "HyperNetwork" 
    }, model_save_path)
    print(f"Saved model to {model_save_path}")
    
    # Save training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    time_info = {
        "total_training_time_seconds": total_training_time,
        "total_training_time_hours": total_training_time / 3600,
        "num_scenarios": args.num_scenarios,
        "max_iterations": args.max_iter,
    }
    time_file = model_save_path.replace(".pt", "_training_time.json")
    with open(time_file, "w") as f:
        json.dump(time_info, f, indent=2)
    print(f"Training time: {total_training_time:.2f} seconds ({total_training_time/3600:.2f} hours)")
    print(f"Saved training time info to {time_file}")

if __name__ == "__main__":
    main()
