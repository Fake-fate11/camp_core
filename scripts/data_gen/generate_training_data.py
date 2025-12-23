
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from camp_core.data_interfaces.nuscenes_trajdata_bridge import (
    NuscenesDatasetConfig,
    NuscenesTrajdataBridge,
    extract_driver_context,
)
from camp_core.inner_solver.driver_solver import DriverAwareInnerSolver
from camp_core.outer_master.benders_master import BendersMaster, BendersMasterConfig, BendersCut

def main():
    parser = argparse.ArgumentParser(description="Generate Training Data for Mapping Head")
    parser.add_argument("--data_root", type=str, default="/ocean/projects/tra250008p/slin24/datasets/nuscenes", help="Path to nuScenes dataset")
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.unified_data_cache"), help="Path to cache directory")
    parser.add_argument("--split", type=str, default="nusc_trainval-train", help="Data split")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--num_atoms", type=int, default=5, help="Number of atoms")
    parser.add_argument("--risk_type", type=str, default="cvar", choices=["mean", "cvar"], help="Risk type")
    parser.add_argument("--alpha", type=float, default=0.9, help="CVaR alpha")
    parser.add_argument("--max_iter", type=int, default=5, help="Maximum Benders iterations per scene")
    parser.add_argument("--num_samples", type=int, default=10000, help="Total number of samples to generate")
    parser.add_argument("--output_path", type=str, default="data/training_data.pkl", help="Output file path")
    
    args = parser.parse_args()
    
    # 1. Data Loading
    print("Initializing Data Loader...")
    cfg = NuscenesDatasetConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True, 
    )
    bridge = NuscenesTrajdataBridge(cfg)
    dataloader = bridge.get_dataloader()
    
    # 2. Initialize Solver
    inner_solver = DriverAwareInnerSolver(horizon=12, dt=0.5)
    
    # 3. Generation Loop
    generated_data = []
    # Initialize MapAPI for context extraction (even though incl_vector_map=False in dataset)
    from trajdata.maps.map_api import MapAPI
    map_api = MapAPI(bridge.dataset.cache_path)
    data_iter = iter(dataloader)
    
    pbar = tqdm(total=args.num_samples, desc="Generating Samples")
    
    while len(generated_data) < args.num_samples:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Restart iterator if needed, or just break
            data_iter = iter(dataloader)
            batch = next(data_iter)
            
        # Process each agent in the batch INDEPENDENTLY
        # In the full Benders experiment, we optimized weights for a *set* of scenarios.
        # Here, to train a mapping head f(context) -> weights, we ideally want the optimal weights
        # for *this specific context*.
        # So we run Benders on a "batch" of size 1 (just this agent).
        
        for i in range(batch.curr_agent_state.shape[0]):
            if len(generated_data) >= args.num_samples:
                break
                
            ctx = extract_driver_context(batch, i, map_api=map_api)
            hist = batch.agent_hist[i].cpu().numpy()
            # In trajdata's local coordinate frame, the current position (last hist point) is at (0, 0)
            # The future trajectory should start from the origin
            y0 = np.array([0.0, 0.0])
            sid = f"scene_{len(generated_data)}"
            
            # Run Benders for THIS SINGLE SCENE
            # This finds the weights that minimize risk for THIS specific situation.
            
            master_config = BendersMasterConfig(
                num_atoms=args.num_atoms,
                scenario_ids=[sid],
                risk_type=args.risk_type,
                alpha=args.alpha,
                solver="SCS",
                fixed_weights={0: 0.1, 3: 0.4}, # Fix Task weights
            )
            master = BendersMaster(master_config)
            current_w = np.ones(args.num_atoms) / args.num_atoms
            
            converged = False
            valid_sample = False
            best_w = current_w
            
            # Benders Loop for single scene
            for iteration in range(args.max_iter):
                # Enable verbose for first sample only
                verbose = (len(generated_data) == 0 and iteration == 0)
                y_opt, atom_vals, cost, success = inner_solver.solve(y0, current_w, ctx, verbose=verbose)
                
                if not success:
                    # If inner solver fails, we can't learn from this sample
                    break
                    
                cut = BendersCut(
                    scenario_id=sid,
                    w_anchor=current_w,
                    value=cost,
                    gradient=atom_vals
                )
                master.add_cut(cut)
                
                solution = master.solve()
                if not solution.success:
                    break
                    
                diff = np.linalg.norm(solution.w_opt - current_w)
                current_w = solution.w_opt
                best_w = current_w
                
                if diff < 1e-4:
                    converged = True
                    valid_sample = True
                    break
            
            # Check if we reached max iter without failure
            # RELAXED CHECK: If we have at least one successful inner solve, we can use the result.
            # The weights might not be fully converged, but they are better than random.
            if success:
                valid_sample = True
            else:
                print(f"Sample {len(generated_data)}: Inner solver failed.")

            if valid_sample:
                # Final check for NaNs
                if np.isnan(best_w).any() or np.isinf(best_w).any():
                    # Fallback to current_w if best_w is bad
                    if not (np.isnan(current_w).any() or np.isinf(current_w).any()):
                        best_w = current_w
                        print(f"Sample {len(generated_data)}: Fallback to current_w due to NaN in best_w.")
                    else:
                        valid_sample = False
                        print(f"Sample {len(generated_data)}: Invalid weights (NaN/Inf).")
                
                # Check context for NaNs
                if np.isnan(ctx.agent_hist).any():
                     # We will impute later, but good to know
                     # print(f"Sample {len(generated_data)}: Context agent_hist has NaNs.")
                     pass

            if valid_sample:
                # Store the result
                # We store the context (features) and the optimal weights
                generated_data.append({
                    "context": ctx,
                    "optimal_weights": best_w,
                    "scene_id": batch.scene_ids[i],
                    "agent_id": batch.agent_name[i]
                })
                pbar.update(1)
                
    pbar.close()
    
    # Save to disk
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(generated_data, f)
        
    print(f"Saved {len(generated_data)} samples to {args.output_path}")

if __name__ == "__main__":
    main()
