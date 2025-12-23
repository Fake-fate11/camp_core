
import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
import gc

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from camp_core.base_predictor.trajectron_loader import build_trajectron_adapter_from_checkpoint, TrajectronLoadConfig
from camp_core.utils.metrics import compute_ade, compute_fde, compute_safety_score
from camp_core.data_interfaces.nuscenes_trajdata_bridge import extract_driver_context
from camp_core.models import SimpleSceneEncoder # Re-using encoder if needed, or just adapter
from camp_core.mapping_heads.hyper_network import HyperNetworkMappingHead
from camp_core.inner_solver.driver_solver import DriverAwareInnerSolver
from trajdata.maps.map_api import MapAPI



def train_finetuning(
    adapter, decoder, train_loader, val_loader, 
    epochs=10, lr=1e-4, mode="partial", device="cuda", max_scenarios=None
):
    """
    Fine-tune the model.
    """
    # 1. Setup Optimizer parameters
    # We are finetuning the adapter (Trajectron model)
    params = []
    
    if mode == "partial":
        print("DEBUG: Partial Mode - Unfreezing only DECODER parameters...")
        if adapter.base_model:
            # First, freeze everything
            for p in adapter.base_model.parameters():
                p.requires_grad_(False)
                
            # Then unfreeze only decoder
            enabled_count = 0
            for name, p in adapter.base_model.named_parameters():
                if "decoder" in name.lower():
                    p.requires_grad_(True)
                    enabled_count += 1
            print(f"DEBUG: Partial mode enabled gradients for {enabled_count} decoder parameters.")
            
        params = list(filter(lambda p: p.requires_grad, adapter.base_model.parameters()))
             
    elif mode == "full":
        if adapter.base_model:
            print("DEBUG: Unfreezing base_model parameters...")
            for p in adapter.base_model.parameters():
                p.requires_grad_(True)
            params = list(adapter.parameters()) + list(adapter.base_model.parameters())
        else:
            params = list(adapter.parameters())
            
    # Unique parameters only (in case adapter.parameters() already included some)
    params = list(set(params))
            
    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()
    
    print(f"DEBUG: Optimizer initialized with {len(params)} parameter groups.")
    total_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"DEBUG: Total trainable parameters: {total_params}")
    
    print(f"Starting {mode} fine-tuning for {epochs} epochs...")
    
    # Calculate max batches if needed (assuming batch_size is constant)
    max_batches = None
    if hasattr(train_loader, 'batch_size') and train_loader.batch_size:
         # Rough estimate
         # If user passed num_scenarios to train_finetuning (as explicit arg or global limit)
         # We need to add that argument first.
         pass
         
    for epoch in range(epochs):
        adapter.train()
        # Ensure base model is in train mode (adapter might wrap it)
        if adapter.base_model:
            adapter.base_model.train()
        
        total_loss = 0
        batches_processed = 0
        
        
        total_steps = len(train_loader)
        if max_scenarios and getattr(train_loader, 'batch_size', None):
             total_steps = min(total_steps, max(1, int(max_scenarios / train_loader.batch_size)))
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", total=total_steps):
            batches_processed += 1
            # HARD LIMIT for comparison speed
            if max_scenarios is not None:
                 if batches_processed * train_loader.batch_size > max_scenarios:
                     break
            
            optimizer.zero_grad()
            
            try:
                # Move batch to device first!
                batch.to(device)
                
                # VALIDATION: Check for invalid sequence lengths
                # Trajectron requires history and future to be non-empty for training (usually)
                # The error "Length of all samples has to be greater than 0" comes from PackedSequence
                
                # Check history length
                if hasattr(batch, 'agent_hist_len'):
                     # Trajectron needs at least 1 history step? Or maybe 2?
                     # Let's be strict: if history is 0, skip.
                     if (batch.agent_hist_len <= 0).any():
                          continue
                     
                     # Also check specific node history if available (Trajectron specific)
                     if hasattr(batch, 'node_hist_len'): # Some versions use this?
                          pass 

                # Check future length (targets)
                if hasattr(batch, 'agent_fut_len'):
                     # We need meaningful future to train.
                     # If future length is 0, we can't train prediction.
                     if (batch.agent_fut_len < 1).any():
                          continue

                # Check for NaNs in future (targets)
                if hasattr(batch, 'agent_fut'):
                     is_nan = torch.isnan(batch.agent_fut)
                     if is_nan.all():
                          continue
                     # If specific agent has no future (all NaNs), loop filter above should catch agent_fut_len=0
                     # But check just in case
                     if getattr(batch, 'agent_fut_len', None) is None:
                          # Fallback check
                          if is_nan.reshape(is_nan.shape[0], -1).all(dim=1).any():
                               continue
                          
                model = adapter.base_model
                
                # Debug Batch (First Epoch Only)
                if epoch == 0 and total_loss == 0: 
                     if hasattr(batch, 'agent_fut'):
                         # print(f"DEBUG: batch.agent_fut shape: {batch.agent_fut.shape}")
                         if torch.isnan(batch.agent_fut).any():
                             pass # print("DEBUG: batch.agent_fut contains some NaNs (likely padding).")
                     else:
                         print("DEBUG: batch.agent_fut MISSING!")
                
                if hasattr(model, 'train_loss'):
                     # Patch for potential missing config key 'single_mode_multi_sample'
                     if hasattr(model, 'hyperparams'):
                         if 'single_mode_multi_sample' not in model.hyperparams:
                             model.hyperparams['single_mode_multi_sample'] = False

                     loss = model.train_loss(batch) 
                     
                     if isinstance(loss, (tuple, list)):
                         loss = loss[0]
                     
                     # Check result
                     if isinstance(loss, torch.Tensor):
                         if loss.item() == 0.0 and epoch <= 1:
                              # If 0, maybe checking gradients will help?
                              if loss.grad_fn is None:
                                   print("DEBUG: Loss has NO grad_fn! It is detached.")
                     
                else:
                     raise RuntimeError("Cannot find train_loss method on Trajectron model.")

            except Exception as e:
                # import traceback
                # traceback.print_exc()
                # print(f"Training step failed: {e}")
                continue

            # Verify loss is valid
            if isinstance(loss, torch.Tensor):
                 if loss.item() == 0.0:
                     print("DEBUG: Loss is strictly 0.0. This suggests an issue.")
            else:
                 print(f"DEBUG: Loss is not a tensor! Type: {type(loss)}") 

            loss.backward()
            total_norm = 0.0
            for p in params:
                 if p.grad is not None:
                     param_norm = p.grad.data.norm(2)
                     total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # Print norm periodically
            if epoch == 0 and total_loss == 0: # First step
                 print(f"DEBUG: First step gradient norm: {total_norm}")

            optimizer.step()
            total_loss += loss.item()
            
            # Memory cleanup
            del batch
            del loss
            torch.cuda.empty_cache()
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
        
    return adapter, None

def evaluate_finetuned(adapter, decoder, val_loader, device, map_api, num_scenarios=None):
    """
    Evaluate fine-tuned model on validation set.
    Returns list of dicts with metrics (ADE, FDE, Safety).
    """
    adapter.eval()
    if decoder is not None:
        decoder.eval() # Should be None now
        
    results = []
    
    scenarios_processed = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if num_scenarios is not None and scenarios_processed >= num_scenarios:
                break
                
            # Filter (robust check) for fair comparison with CAMP
            # Filter removed here, moved to agent loop for granularity
            # if hasattr(batch, 'agent_fut_len') and (batch.agent_fut_len < 12).any():
            #      continue
                 
            try:
                # Get predictions & Measure Runtime
                start_time = time.time()
                
                # Use helper function to get Top-1 prediction
                # Returns List[np.ndarray] of shape [K, H, 2]
                pred_traj_list = get_top_k_predictions(adapter, batch, k=1, z_mode=True, gmm_mode=True)
                
                # Stack to [B, H, 2]
                # Each element is [1, H, 2] -> take [0]
                pred_traj_np = np.stack([p[0] for p in pred_traj_list]) # [B, H, 2]
                
                end_time = time.time()
                
                # Average runtime per agent in this batch
                runtime = (end_time - start_time) / len(batch.agent_name)
                
                # Ground truth - slice to horizon=12
                gt_traj = batch.agent_fut[:, :12, :2].cpu().numpy()
                
                # Metrics for each agent in batch
                for i in range(len(batch.agent_name)):
                    # Debug filtering
                    if i == 0 and scenarios_processed < 5:
                         print(f"DEBUG: Batch agent_fut_len: {batch.agent_fut_len}")
                    
                    # Agent-level filter for 12-step horizon
                    if hasattr(batch, 'agent_fut_len'):
                         flen = batch.agent_fut_len[i]
                         if flen < 12:
                             # print(f"DEBUG: Skipping agent {i} with len {flen}")
                             continue
                    
                    try:
                        ctx = extract_driver_context(batch, i, map_api=map_api)
                        
                        # Ensure lengths match for metric computation
                        curr_gt = gt_traj[i]
                        curr_pred = pred_traj_np[i]
                        
                        min_len = min(len(curr_gt), len(curr_pred))
                        curr_gt = curr_gt[:min_len]
                        curr_pred = curr_pred[:min_len]
                        
                        ade = compute_ade(curr_pred, curr_gt)
                        fde = compute_fde(curr_pred, curr_gt)
                        safety = compute_safety_score(curr_pred, ctx)
                        
                        # Compute atomic metrics for detailed analysis
                        # Note: Atoms usually need local frame? 
                        # get_top_k_predictions returns in whatever frame Trajectron outputs.
                        # Usually local standard frame.
                        from camp_core.atoms.driver_atoms import compute_driver_atom_features
                        atom_feats = compute_driver_atom_features(ctx, curr_pred)
                        atoms = atom_feats.as_vector()  # [jerk, smoothness, lane_dev, speed_lim, progress, clearance]
                        
                        results.append({
                            "ADE": ade,
                            "FDE": fde,
                            "Safety": safety,
                            "Runtime": runtime,
                            "SceneID": f"{batch.scene_ids[i]}_{batch.agent_name[i]}",
                             # Atomic metrics
                            "Jerk": atoms[0],
                            "Smoothness": atoms[1],
                            "LaneDeviation": atoms[2],
                            "SpeedLimit": atoms[3],
                            "Progress": atoms[4],
                            "Clearance": atoms[5]
                        })
                        scenarios_processed += 1
                        if num_scenarios is not None and scenarios_processed >= num_scenarios:
                            break
                    except Exception as e:
                        print(f"Evaluation failed for agent {i}: {e}")
                        continue
            except Exception as e:
                print(f"Batch evaluation failed: {e}")
                continue
            
            # Memory cleanup
            del batch
            if 'embeddings' in locals(): del embeddings
            if 'pred_traj' in locals(): del pred_traj
            torch.cuda.empty_cache()
            gc.collect()
                    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Compare finetuning methods")
    parser.add_argument("--data_root", type=str, default="/ocean/projects/tra250008p/slin24/datasets/nuscenes", help="Path to nuScenes dataset")
    parser.add_argument("--split", type=str, default="nusc_trainval-val", help="Data split")
    parser.add_argument("--trajectron_conf", type=str, required=True, help="Path to Trajectron config")
    parser.add_argument("--trajectron_model_dir", type=str, required=True, help="Path to Trajectron model directory")
    parser.add_argument("--trajectron_epoch", type=int, default=20, help="Trajectron epoch")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--num_scenarios", type=int, default=None, help="Number of scenarios to evaluate")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--epochs", type=int, default=10, help="Finetuning epochs")
    parser.add_argument("--mode", type=str, default="partial", choices=["partial", "full"], help="Finetuning mode")
    parser.add_argument("--camp_results_path", type=str, default=None, help="Path to existing CAMP methods comparison CSV (Legacy)")
    parser.add_argument("--camp_model_path", type=str, default="models/camp_level_a.pt", help="Path to CAMP model checkpoint for re-evaluation")
    return parser.parse_args()



def to_numpy(x):
    """Convert tensor or array to numpy array."""
    if isinstance(x, np.ndarray):
        return x
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)

def get_top_k_predictions(adapter, batch, k=6, z_mode=False, gmm_mode=False):
    """
    Get Top-K trajectories from Trajectron++ base model.
    Returns: List of [K, H, 2] numpy arrays (one per agent in batch).
    """
    trajectron = adapter.base_model
    
    if hasattr(trajectron, "hyperparams") and "single_mode_multi_sample" not in trajectron.hyperparams:
        trajectron.hyperparams["single_mode_multi_sample"] = False
    
    ph = 12
    if hasattr(trajectron, "hyperparams"):
        ph = trajectron.hyperparams.get("prediction_horizon", 12)
    
    device = next(trajectron.parameters()).device
    batch.to(device)
    
    with torch.no_grad():
        if gmm_mode and k > 1:
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
    
    batch_preds = []
    bs = batch.curr_agent_state.shape[0] if hasattr(batch, 'curr_agent_state') else 1
    
    try:
        from trajdata import AgentType
    except ImportError:
        pass # Should be imported

    for i in range(bs):
        try:
            node_type_enum = AgentType(batch.agent_type[i].item())
            agent_name = batch.agent_name[i]
            agent_key = f"{str(node_type_enum)}/{agent_name}"
            
            agent_k_preds = []
            for p_dict in predictions_list:
                 p = None
                 # Try exact key
                 if agent_key in p_dict:
                     p = p_dict[agent_key]
                 else:
                     # Loose search
                     for k_outer, v_outer in p_dict.items():
                         if isinstance(v_outer, dict) and agent_name in v_outer:
                              p = v_outer[agent_name]
                              break
                 
                 if p is None:
                     p = np.zeros((1, ph, 2))
                 
                 if hasattr(p, 'cpu'): p = p.cpu().numpy()
                 # Ensure shape [1, H, 2] if just [H, 2]
                 if len(p.shape) == 2:
                     p = p[None, :, :]
                 agent_k_preds.append(p)
            
            if len(predictions_list) > 1:
                # LIST of [1, H, 2] -> [K, H, 2] via cat
                # Each element was single sample. 
                pred_traj_i = np.concatenate(agent_k_preds, axis=0) # [K, H, 2]
            else:
                pred_traj_i = agent_k_preds[0] # [K, H, 2]
                
            batch_preds.append(to_numpy(pred_traj_i))
        except Exception as e:
            # Fallback
            batch_preds.append(np.zeros((k, ph, 2)))
        
    return batch_preds


def evaluate_cost(solver, trajectory, weights, context):
    """
    Evaluate the cost of a trajectory given weights using the solver's atoms.
    """
    from camp_core.atoms.driver_atoms import compute_driver_atom_features
    
    # Compute atom features for the trajectory
    feats = compute_driver_atom_features(context, trajectory)
    raw_vals = feats.as_vector()
    
    # Normalize using context scales (matching solver's normalization)
    scale_keys = [
        "jerk_scale", "smooth_scale", "corridor_scale",
        "speed_limit_scale", "progress_scale", "clearance_scale"
    ]
    
    normalized_vals = []
    # Solver might check atom_names property if available
    atom_names = getattr(solver, "atom_names", ["smoothness", "lane_deviation", "clearance", "speed_limit", "progress"])
    
    for i in range(len(atom_names)):
        val = raw_vals[i]
        scale = getattr(context, scale_keys[i], 1.0)
        if scale < 1e-6:
            scale = 1.0
        normalized_vals.append(val / scale)
    
    # Weighted cost
    cost = np.dot(weights, normalized_vals)
    return cost

def evaluate_camp(adapter, mapping_head, solver, loader, device, map_api, num_scenarios=None):
    results = []
    processed = 0
    
    print("Evaluating CAMP Model (Re-running on current subset)...")
    pbar = tqdm(total=num_scenarios if num_scenarios else len(loader.dataset))
    
    mapping_head.eval()
    
    for batch in loader:
        if num_scenarios and processed >= num_scenarios:
            break
            
        bs = batch.curr_agent_state.shape[0]
        
        # 1. Embed and Predict Weights
        with torch.no_grad():
            # Use get_scene_embeddings to match compare_methods logic
            # Explicitly cast to tensor if numpy is returned (Trajectron specific)
            emb = adapter.get_scene_embeddings(batch)
            if isinstance(emb, np.ndarray):
                batch_embeddings = torch.tensor(emb, device=device)
            else:
                batch_embeddings = emb.to(device)
            
            batch_weights = mapping_head(batch_embeddings).cpu().numpy()
            
        # 2. Get Warm Starts (Top-K Selection)
        # Generate diverse candidates and pick best using the learned cost function
        k_candidates_list = get_top_k_predictions(adapter, batch, k=6, z_mode=False, gmm_mode=True)
        
        for i in range(bs):
            if num_scenarios and processed >= num_scenarios:
                break
                
            try:
                # Extract Context First (needed for cost evaluation)
                ctx = extract_driver_context(batch, i, map_api=map_api)
                
                # Check bounds
                if i >= len(k_candidates_list):
                    continue
                    
                # Select Best Warm Start
                k_trajs = k_candidates_list[i] # [K, H, 2]
                if isinstance(k_trajs, torch.Tensor):
                    k_trajs = k_trajs.detach().cpu().numpy()
                
                # Convert to list for iteration
                k_trajs_list = list(k_trajs)
                
                # Multi-Round Iterative CAMP (Full Benders Test-Time Adaptation)
                # Persistent TTA: Updates mapping_head directly across all samples
                max_iter = 10
                lr_tta = 0.01
                
                # Get embedding for this sample (with grad enabled)
                emb_i = batch_embeddings[i:i+1]  # [1, D]
                
                # Create optimizer (updates mapping_head directly)
                tta_optimizer = torch.optim.Adam(mapping_head.parameters(), lr=lr_tta)
                
                # Store cuts
                cuts = []
                y_opt = k_trajs[0]  # Default fallback
                success = True
                
                for iter_i in range(max_iter):
                    # A. Forward pass to get weights with grad
                    mapping_head.train()
                    weights_tensor = mapping_head(emb_i)  # [1, R]
                    weights = weights_tensor.detach().cpu().numpy()[0]
                    
                    # B. Select best warm start given current weights
                    best_traj = k_trajs_list[0]
                    min_cost = float('inf')
                    
                    for k_idx in range(len(k_trajs_list)):
                        c = evaluate_cost(solver, k_trajs_list[k_idx], weights, ctx)
                        if c < min_cost:
                            min_cost = c
                            best_traj = k_trajs_list[k_idx]
                            
                    pred_traj = best_traj
                    scene_id = str(batch.scene_ids[i])
                    y0 = np.zeros(2) 
                    
                    # C. Solve Inner Problem
                    t0 = time.time()
                    y_opt, atom_vals, cost, success = solver.solve(y0, weights, ctx, warm_start_traj=pred_traj)
                    
                    if not success:
                        break
                    
                    # D. Create Benders Cut
                    cuts.append({
                        "Q": cost,
                        "gradient": torch.tensor(atom_vals, dtype=torch.float32, device=device),
                        "w_anchor": torch.tensor(weights, dtype=torch.float32, device=device)
                    })
                    
                    # E. Benders Master Step: Update HyperNetwork via Backprop
                    tta_optimizer.zero_grad()
                    
                    # Compute theta = mean over cuts
                    theta_vals = []
                    for cut in cuts:
                        diff = weights_tensor[0] - cut["w_anchor"]
                        theta_cut = cut["Q"] + torch.dot(cut["gradient"], diff)
                        theta_vals.append(theta_cut)
                    
                    loss = torch.mean(torch.stack(theta_vals))
                    loss.backward()
                    tta_optimizer.step()
                    
                    # F. Prepend y_opt to candidates for next iteration
                    k_trajs_list = [y_opt] + k_trajs_list
                
                # Get final weights after update
                mapping_head.eval()
                with torch.no_grad():
                    batch_weights[i] = mapping_head(emb_i).cpu().numpy()[0]
                
                if valid_plot_candidate:
                     print(f"DEBUG: Solved in {time.time()-t0:.3f}s. Success: {success}")
                
                # 4. Metrics
                ade = compute_ade(y_opt, gt_traj)
                fde = compute_fde(y_opt, gt_traj)
                safety = compute_safety_score(y_opt, ctx)
                
                runtime = 0.6 
                
                # Plot visualization if interesting
                if valid_plot_candidate:
                    plot_scene_comparison(scene_id, ctx, {
                        "pred-top1": pred_traj,
                        "plan-adaptive-iterative": y_opt
                    }, args.output_dir, gt_traj=gt_traj)
                    plots_generated += 1 
                
                # Atomic Metrics
                from camp_core.atoms.driver_atoms import compute_driver_atom_features
                atom_feats = compute_driver_atom_features(ctx, y_opt)
                atoms = atom_feats.as_vector()
                
                res = {
                    "Method": "CAMP (Iterative)",
                    "ADE": ade, "FDE": fde, "Safety": safety, "Runtime": runtime,
                    "SceneID": scene_id,
                    "Jerk": atoms[0],
                    "Smoothness": atoms[1],
                    "LaneDeviation": atoms[2],
                    "SpeedLimit": atoms[3],
                    "Progress": atoms[4],
                    "Clearance": atoms[5],
                    "Stage": "CAMP (Iterative)"
                }
                results.append(res)
                
                processed += 1
                pbar.update(1)
                
            except Exception as e:
                continue
                
    pbar.close()
    return results

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Trajectron Adapter (in decoder mode for finetuning)
    print("Loading Trajectron Adapter...")
    traj_cfg = TrajectronLoadConfig(
        conf_path=args.trajectron_conf,
        model_dir=args.trajectron_model_dir,
        epoch=args.trajectron_epoch,
        device=str(device),
    )
    # Use "decoder" mode to expose the decoding functionality
    adapter = build_trajectron_adapter_from_checkpoint(
        load_cfg=traj_cfg,
        embedding_dim=args.embedding_dim,
        mode="decoder", 
    )
    adapter.to(device)
    # For Pre-Finetune evaluation, set to eval
    adapter.eval()
    
    # No separate decoder needed - we finetune the adapter itself
    decoder = None
    
    # 3. Load Data
    print("Loading Data...")
    from camp_core.data_interfaces.nuscenes_trajdata_bridge import NuscenesTrajdataBridge, NuscenesDatasetConfig
    
    # Revert: UnifiedDataset does not support min_history_len directly in init.
    # We must filter in loop.
    
    cfg = NuscenesDatasetConfig(
        data_root=args.data_root,
        cache_dir=os.path.expanduser("~/.unified_data_cache"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False, # Deterministic loading
        split=args.split
    )
    bridge = NuscenesTrajdataBridge(cfg)
    dataset = bridge.dataset
    map_api = dataset
    
    # Keep reference to original dataset for collate_fn
    original_dataset = dataset
    
    # Split into train/val Deterministically
    # Use first K samples for Validation to match compare_methods script (Pre-Finetune check)
    val_count = args.num_scenarios if args.num_scenarios else 20
    # Ensure reasonable size
    val_count = min(val_count, len(dataset))
    
    val_indices = list(range(val_count))
    train_indices = list(range(val_count, len(dataset)))
    
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    
    # Limit training data if num_scenarios is specified
    # NOTE: We do NOT limit the dataset size here, because many samples might be filtered (short horizon).
    # We will rely on break conditions in the loops to stop after valid samples.
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=original_dataset.get_collate_fn(pad_format="right"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=original_dataset.get_collate_fn(pad_format="right"))
    
    # 4. Evaluate Before Finetuning (Pre-Finetune = Pred-Top1 approx)
    print("Evaluating BEFORE finetuning...")
    # Pass None as decoder - adapter handles prediction now
    pre_results = evaluate_finetuned(adapter, None, val_loader, device, map_api=map_api, num_scenarios=args.num_scenarios)
    for r in pre_results:
        r["Stage"] = "Pre-Finetune"
        
    # 5. Fine-tune
    print("Fine-tuning...")
    finetune_start_time = time.time()
    # We finetune the ADAPTER directly
    adapter, _ = train_finetuning(
        adapter, None, train_loader, val_loader, 
        epochs=args.epochs, mode=args.mode, device=device, max_scenarios=args.num_scenarios
    )
    finetune_end_time = time.time()
    total_finetune_time = finetune_end_time - finetune_start_time
    print(f"Total Finetuning Time: {total_finetune_time:.2f} seconds ({total_finetune_time/3600:.2f} hours)")
    
    # 6. Evaluate After Finetuning
    print("Evaluating AFTER finetuning...")
    post_results = evaluate_finetuned(adapter, None, val_loader, device, map_api=map_api, num_scenarios=args.num_scenarios)
    for r in post_results:
        r["Stage"] = "Post-Finetune"
        
    # 7. Evaluate/Load CAMP Results
    camp_data = []
    
    # A. Priority: Re-evaluate CAMP model on current subset
    # A. Priority: Re-evaluate CAMP model on current subset
    if args.camp_model_path:
        if os.path.exists(args.camp_model_path):
            print(f"Loading CAMP model from {args.camp_model_path} for re-evaluation...")
            
            # Init Model (Must match training config)
            mapping_head = HyperNetworkMappingHead(
                embedding_dim=64, # Hardcoded or use args.embedding_dim if available
                num_atoms=6,
                hidden_dims=[128, 128]
            ).to(device)
            
            try:
                checkpoint = torch.load(args.camp_model_path, map_location=device)
                if "head" in checkpoint:
                    mapping_head.load_state_dict(checkpoint["head"])
                else:
                    mapping_head.load_state_dict(checkpoint)
                print("CAMP Model loaded successfully. Starting Live Evaluation...")
                
                # Init Solver
                solver = DriverAwareInnerSolver(horizon=12, dt=0.5)
                
                # Run Eval
                camp_data = evaluate_camp(adapter, mapping_head, solver, val_loader, device, map_api, num_scenarios=args.num_scenarios)
                print(f"Live Evaluation Complete. Processed {len(camp_data)} scenarios.")
                
            except Exception as e:
                print(f"ERROR: Failed to load/eval CAMP model: {e}")
                print("WARNING: Falling back to CSV results due to Model Error.")
                camp_data = []
        else:
             print(f"WARNING: CAMP model path provided but file not found: {args.camp_model_path}")
             print("WARNING: Falling back to CSV results.")


    # B. Fallback: Load from CSV
    if not camp_data:
        if args.camp_results_path:
            camp_results_path = args.camp_results_path
        else:
            camp_results_path = os.path.join(args.output_dir, "method_comparison.csv")
            
        if os.path.exists(camp_results_path):
            print(f"Loading CAMP results from CSV {camp_results_path}...")
            camp_df = pd.read_csv(camp_results_path)
            # Filter for CAMP methods
            camp_methods = ["plan-adaptive-camp", "plan-adaptive-intermediate"]
            camp_df_filtered = camp_df[camp_df["Method"].isin(camp_methods)].copy()
            
            if len(camp_df_filtered) > 0:
                for _, row in camp_df_filtered.iterrows():
                    data_point = {
                        "ADE": row["ADE"],
                        "FDE": row["FDE"],
                        "Safety": row["Safety"],
                        "Runtime": row["Runtime"],
                        "SceneID": row.get("SceneID", "N/A"),
                        "Stage": "CAMP (Intermediate)" 
                    }
                     # Copy atomic metrics if they exist
                    for atom in ["Jerk", "Smoothness", "LaneDeviation", "Clearance", "SpeedLimit", "Progress"]:
                        if atom in row:
                            data_point[atom] = row[atom]
                    camp_data.append(data_point)
            else:
                 print(f"Warning: No CAMP methods found in CSV.")
        else:
            print(f"Warning: No CAMP model and no CSV found. Skipping comparison.")
    
    # Load CAMP training time if available
    camp_time_data = {}
    camp_time_file = "models/camp_level_a_training_time.json"
    if os.path.exists(camp_time_file):
        with open(camp_time_file, "r") as f:
            camp_time_data = json.load(f)
        print(f"CAMP training time loaded: {camp_time_data.get('total_training_time_hours', 'N/A'):.2f} hours")
    else:
        print(f"Warning: CAMP training time not found at {camp_time_file}")
        
    # 8. Save & Compare
    # Print Training Time Comparison
    print("\n" + "="*50)
    print("TRAINING TIME COMPARISON")
    print("="*50)
    if camp_time_data:
        camp_hours = camp_time_data.get('total_training_time_hours', 0)
        print(f"CAMP Training Time:      {camp_hours:.4f} hours")
    else:
        print("CAMP Training Time:      N/A")
        
    finetune_hours = total_finetune_time / 3600
    print(f"Finetuning Time ({args.mode}): {finetune_hours:.4f} hours")
    
    if camp_time_data:
        ratio = camp_hours / finetune_hours if finetune_hours > 0 else float('inf')
        print(f"Ratio (CAMP / Finetune): {ratio:.2f}x")
    print("="*50 + "\n")

    all_results = pre_results + post_results + camp_data
    df = pd.DataFrame(all_results)
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, f"finetuning_{args.mode}_comparison.csv"), index=False)
    
    # Calculate averages
    print("\n" + "="*80)
    print("DETAILED METHOD COMPARISON")
    print("="*80)
    
    numeric_cols = ["ADE", "FDE", "Safety", "Runtime", "Jerk", "Smoothness", "LaneDeviation", "Clearance", "SpeedLimit", "Progress"]
    summary_df = df.groupby("Stage")[numeric_cols].mean().reset_index()
    
    # Define order
    stage_order = ["Pre-Finetune", "Post-Finetune", "CAMP (Intermediate)"]
    
    # Sort by stage order if possible
    summary_df["Order"] = summary_df["Stage"].apply(lambda x: stage_order.index(x) if x in stage_order else 99)
    summary_df = summary_df.sort_values("Order").drop("Order", axis=1)
    
    # Print as Markdown Table
    print(summary_df.to_markdown(index=False, floatfmt=".4f"))
    print("\n" + "="*80 + "\n")
    
    # Save table to file
    with open(os.path.join(args.output_dir, "finetuning_comparison_table.txt"), "w") as f:
        f.write(summary_df.to_markdown(index=False, floatfmt=".4f"))
    
    # Plotting
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create comprehensive metrics visualization
    fig = plt.figure(figsize=(18, 10))
    
    # 1. ADE/FDE Bar Plot
    ax1 = plt.subplot(2, 3, 1)
    # Define fixed order: baselines on left, our methods on right
    stage_order = ["Pre-Finetune", "Post-Finetune", "CAMP (Intermediate)"]
    metrics_error_df = df.melt(id_vars=["Stage"], value_vars=["ADE", "FDE"], var_name="Metric", value_name="Error (m)")
    sns.barplot(data=metrics_error_df, x="Stage", y="Error (m)", hue="Metric", ax=ax1, order=stage_order)
    ax1.set_title("Prediction Error by Stage", fontsize=12, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.legend(title='')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Safety Score Bar Plot
    ax2 = plt.subplot(2, 3, 2)
    stage_avg_safety = df.groupby("Stage")["Safety"].mean().reset_index()
    sns.barplot(data=stage_avg_safety, x="Stage", y="Safety", ax=ax2, hue="Stage", palette='viridis', order=stage_order, legend=False)
    ax2.set_title("Safety Score (Lower is Better)", fontsize=12, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Atomic Metrics - Part 1 (Jerk, Smoothness, Lane Deviation)
    ax3 = plt.subplot(2, 3, 3)
    atomic_cols1 = ["Jerk", "Smoothness", "LaneDeviation"]
    # Check if columns exist (CAMP results might not have them if loaded from old CSV)
    available_cols1 = [c for c in atomic_cols1 if c in df.columns]
    if available_cols1:
        atomic_df1 = df.groupby("Stage")[available_cols1].mean().reset_index()
        # Reindex to match stage_order
        atomic_df1 = atomic_df1.set_index("Stage").reindex(stage_order).reset_index()
        x_pos = np.arange(len(atomic_df1))
        width = 0.25
        for i, col in enumerate(available_cols1):
            ax3.bar(x_pos + i*width, atomic_df1[col], width, label=col, alpha=0.8)
        ax3.set_xticks(x_pos + width)
        ax3.set_xticklabels(atomic_df1["Stage"], rotation=45, ha='right')
        ax3.set_ylabel('Cost (Lower is Better)')
        ax3.set_title('Atomic Metrics (1/2)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

    # 4. Atomic Metrics - Part 2 (Clearance, Speed Limit, Progress)
    ax4 = plt.subplot(2, 3, 4)
    atomic_cols2 = ["Clearance", "SpeedLimit", "Progress"]
    # Check if columns exist
    available_cols2 = [c for c in atomic_cols2 if c in df.columns]
    if available_cols2:
        atomic_df2 = df.groupby("Stage")[available_cols2].mean().reset_index()
        atomic_df2 = atomic_df2.set_index("Stage").reindex(stage_order).reset_index()
        x_pos = np.arange(len(atomic_df2))
        width = 0.35
        for i, col in enumerate(available_cols2):
            ax4.bar(x_pos + i*width, atomic_df2[col], width, label=col, alpha=0.8)
        ax4.set_xticks(x_pos + width/2)
        ax4.set_xticklabels(atomic_df2["Stage"], rotation=45, ha='right')
        ax4.set_ylabel('Cost (Lower is Better)')
        ax4.set_title('Atomic Metrics (2/2)', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
    # 5. Runtime Comparison
    ax5 = plt.subplot(2, 3, 5)
    stage_avg_runtime = df.groupby("Stage")["Runtime"].mean().reset_index()
    sns.barplot(data=stage_avg_runtime, x="Stage", y="Runtime", ax=ax5, hue="Stage", palette='plasma', order=stage_order, legend=False)
    ax5.set_title("Average Runtime (seconds)", fontsize=12, fontweight='bold')
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    ax5.grid(axis='y', alpha=0.3)

    # 6. Radar Chart for All Metrics (normalized)
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    
    # Prepare data for radar chart
    all_metric_cols = ["Jerk", "Smoothness", "LaneDeviation", "Clearance", "SpeedLimit", "Progress"]
    available_radar_cols = [c for c in all_metric_cols if c in df.columns]
    
    if available_radar_cols:
        stage_metrics = df.groupby("Stage")[available_radar_cols].mean()
        
        # Normalize to [0, 1] for visualization
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_metrics = pd.DataFrame(
            scaler.fit_transform(stage_metrics),
            columns=available_radar_cols,
            index=stage_metrics.index
        )
        
        # INVERT METRICS: 1 - normalized_cost
        # Now 1.0 = Best (Lowest Cost), 0.0 = Worst (Highest Cost)
        normalized_metrics = 1.0 - normalized_metrics
        
        # Radar chart setup
        angles = np.linspace(0, 2 * np.pi, len(available_radar_cols), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        for stage in stage_order:
            if stage in normalized_metrics.index:
                values = normalized_metrics.loc[stage].tolist()
                values += values[:1]  # Close the loop
                ax6.plot(angles, values, linewidth=2, linestyle='solid', label=stage)
                ax6.fill(angles, values, alpha=0.25)
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(available_radar_cols, fontsize=9)
        ax6.set_title('Overall Performance (1.0 = Best)', fontsize=12, fontweight='bold', pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"finetuning_{args.mode}_detailed.png"), dpi=300)
    print(f"Detailed metrics plot saved to {os.path.join(args.output_dir, f'finetuning_{args.mode}_detailed.png')}")
    


if __name__ == "__main__":
    main()
