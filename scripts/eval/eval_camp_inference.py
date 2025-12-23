
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

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
from camp_core.base_predictor.trajectron_loader import (
    TrajectronLoadConfig,
    build_trajectron_adapter_from_checkpoint,
)
from camp_core.mapping_heads.hyper_network import HyperNetworkMappingHead
from camp_core.mapping_heads.simplex_linear import LinearSoftmaxMappingHead

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CAMP Level-A on nuScenes")
    parser.add_argument("--data_root", type=str, default="/ocean/projects/tra250008p/slin24/datasets/nuscenes", help="Path to nuScenes dataset")
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.unified_data_cache"), help="Path to cache directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--num_scenarios", type=int, default=20, help="Number of scenarios to evaluate")
    parser.add_argument("--num_atoms", type=int, default=6, help="Number of atoms")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--model_path", type=str, default="models/camp_level_a.pt", help="Path to trained model")
    
    # Trajectron
    parser.add_argument("--traj_conf_path", type=str, default="/ocean/projects/tra250008p/slin24/MetaLearning/adaptive-prediction/experiments/nuScenes/models/nusc_mm_base_tpp-11_Sep_2022_19_15_45/config.json")
    parser.add_argument("--traj_model_dir", type=str, default="/ocean/projects/tra250008p/slin24/MetaLearning/adaptive-prediction/experiments/nuScenes/models/nusc_mm_base_tpp-11_Sep_2022_19_15_45")
    parser.add_argument("--traj_epoch", type=int, default=20)
    
    return parser.parse_args()

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
        shuffle=False, # Eval on fixed set (or validation split)
        split="nusc_trainval-val" # Use validation split
    )
    bridge = NuscenesTrajdataBridge(cfg)
    print(f"DEBUG: Dataset size: {len(bridge.dataset)}")
    dataloader = bridge.get_dataloader()
    print(f"DEBUG: Dataloader length: {len(dataloader)}")
    map_api = bridge.dataset
    
    # 2. Initialize Models
    print("Initializing Models...")
    
    # Trajectron Adapter (Frozen)
    traj_cfg = TrajectronLoadConfig(
        conf_path=args.traj_conf_path,
        model_dir=args.traj_model_dir,
        epoch=args.traj_epoch,
        device=args.device,
    )
    adapter = build_trajectron_adapter_from_checkpoint(
        load_cfg=traj_cfg,
        embedding_dim=args.embedding_dim,
        mode="encoder",
    )
    adapter.to(device)
    adapter.eval()
    
    # Mapping Head (Trained)
    # Mapping Head (Trained)
    # WARNING: Must match training architecture
    mapping_head = HyperNetworkMappingHead(
        embedding_dim=args.embedding_dim,
        num_atoms=args.num_atoms,
        hidden_dims=[128, 128] # Default from train_camp_level_a.py
    ).to(device)
    
    # Load weights
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and "head" in checkpoint:
            mapping_head.load_state_dict(checkpoint["head"])
        else:
            mapping_head.load_state_dict(checkpoint)
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Warning: Model path {args.model_path} does not exist. Using random weights.")
        
    mapping_head.eval()
    
    # Inner Solver
    inner_solver = DriverAwareInnerSolver(horizon=12, dt=0.5)
    
    # 3. Evaluation Loop
    print(f"Evaluating on {args.num_scenarios} scenarios...")
    
    total_cost = 0.0
    scenarios_evaluated = 0
    pbar = tqdm(total=args.num_scenarios)
    
    for batch in dataloader:
        if scenarios_evaluated >= args.num_scenarios:
            break
            
        with torch.no_grad():
            adapter_out = adapter.embed_batch(batch)
            batch_embeddings = adapter_out["scene_embeddings"].to(device)
            
            # Predict weights
            batch_weights = mapping_head(batch_embeddings).cpu().numpy()
            
        B = batch.curr_agent_state.shape[0]
        for i in range(B):
            if scenarios_evaluated >= args.num_scenarios:
                break
                
            # Check validity first
            curr_state = batch.curr_agent_state[i].cpu().numpy()
            if np.isnan(curr_state).any():
                continue
                
            try:
                ctx = extract_driver_context(batch, i, map_api=map_api)
                
                # Use curr_agent_state for reliability check, but y0 is local [0,0]
                curr_state = batch.curr_agent_state[i].cpu().numpy()
                if np.isnan(curr_state).any():
                    continue
                y0 = np.zeros(2)
                
                w_i = batch_weights[i]
                
                # Solve Inner
                y_opt, atom_vals, cost, success = inner_solver.solve(y0, w_i, ctx)
                
                if success:
                    total_cost += cost
                    scenarios_evaluated += 1
                    pbar.update(1)
                    
            except Exception as e:
                continue
        
        # Memory cleanup
        del batch
        del adapter_out
        del batch_embeddings
        del batch_weights
        torch.cuda.empty_cache()
        gc.collect()
                
    pbar.close()
    
    if scenarios_evaluated > 0:
        avg_cost = total_cost / scenarios_evaluated
        print(f"Average Cost: {avg_cost:.4f}")
    else:
        print("No scenarios evaluated.")

if __name__ == "__main__":
    main()
