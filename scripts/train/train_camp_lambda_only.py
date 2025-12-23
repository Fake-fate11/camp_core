
import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from camp_core.base_predictor.trajectron_loader import build_trajectron_adapter_from_checkpoint, TrajectronLoadConfig
from camp_core.data_interfaces.nuscenes_trajdata_bridge import NuscenesTrajdataBridge, NuscenesDatasetConfig, extract_driver_context
from camp_core.mapping_heads.simplex_linear import LinearSoftmaxMappingHead
from camp_core.atoms.driver_atoms import compute_driver_atom_features
from camp_core.utils.metrics import compute_ade, compute_fde

def parse_args():
    parser = argparse.ArgumentParser(description="Train CAMP Lambda-Only (MaxEnt IRL)")
    
    # Data params
    parser.add_argument("--data_root", type=str, default="/ocean/projects/tra250008p/slin24/datasets/nuscenes", help="Path to nuScenes dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--num_scenarios", type=int, default=200, help="Number of training scenarios")
    parser.add_argument("--split", type=str, default="nusc_trainval-train", help="Data split")
    
    # Model params
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for mapping head")
    parser.add_argument("--num_atoms", type=int, default=6, help="Number of atoms")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save model")
    
    # Training params
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 Regularization (Weight Decay)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    
    # Trajectron
    parser.add_argument("--trajectron_conf", type=str, required=True, help="Path to Trajectron config")
    parser.add_argument("--trajectron_model_dir", type=str, required=True, help="Path to Trajectron model directory")
    parser.add_argument("--trajectron_epoch", type=int, default=20)
    
    # Cache
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.unified_data_cache"), help="Cache directory for Trajdata")
    
    return parser.parse_args()

def get_trajectron_samples(adapter, batch, k=20):
    """
    Generate K samples for each agent in the batch using Trajectron.
    Returns: List of [K, 12, 2] arrays. One per agent in batch.
    """
    model = adapter.base_model
    try:
        # We need to use the model's predict method
        # predict returns a dict: {scene_id: {node_type: {node_id: prediction}}}
        # We want to force it to produce K samples
        
        # Note: Trajectron's 'predict' usually takes 'z_mode' and 'gmm_mode'.
        # To get raw samples, we might need a different call or configure it.
        # But 'predict' with num_samples should work.
        
        # We'll use adapter's wrapper if available or call predict directly.
        # Adapter generally has 'get_predictions', let's check.
        # For now, we assume standard Trajectron predict API:
        # predictions = model.predict(batch, ph=12, num_samples=k, z_mode=False, gmm_mode=False, full_dist=False, all_z_sep=False)
        
        # However, `trajectron_loader.py` suggests we might interact with the adapter.
        # Let's try to access the internal predict logic.
        
        if k > 1:
            # Trajectron assert fails if gmm_mode=True and num_samples > 1.
            # Workaround: Call predict K times with num_samples=1
            # We want High-Quality (Mode) but Diverse (Sampled Z) trajectories.
            predictions_list = []
            for _ in range(k):
                p_dict = model.predict(
                    batch,
                    ph=12,
                    num_samples=1,
                    z_mode=False, # Sample Z for diversity
                    gmm_mode=True, # Use Mode of Y given Z for quality
                    full_dist=False 
                )
                predictions_list.append(p_dict)
        else:
            predictions_dict = model.predict(
                batch,
                ph=12,
                num_samples=k,
                z_mode=False,
                gmm_mode=True,
                full_dist=False 
            )
            predictions_list = [predictions_dict]
            
        batch_samples = []
        
        # We assume batch size B is handled by Trajectron returning all in dict
        # We iterate over agents in batch
        
        for i in range(batch.curr_agent_state.shape[0]):
            scene_id = batch.scene_ids[i]
            node_id = batch.agent_name[i]
            node_type = batch.agent_type[i] if hasattr(batch, 'agent_type') else model.node_type
            if isinstance(node_type, torch.Tensor):
                node_type = 'VEHICLE' 
            
            agent_k_preds = []
            for p_dict in predictions_list:
                try:
                    preds = p_dict[scene_id][node_type][node_id] # [1, 12, 2]
                    agent_k_preds.append(preds)
                except KeyError:
                    agent_k_preds.append(np.zeros((1, 12, 2)))
            
            # Stack/Concat
            if len(predictions_list) > 1:
                batch_samples.append(np.concatenate(agent_k_preds, axis=0)) # [K, 12, 2]
            else:
                batch_samples.append(agent_k_preds[0]) # [K, 12, 2]
                
        return batch_samples

    except Exception as e:
        # print(f"Sampling failed: {e}")
        # Return zeros
        return [np.zeros((k, 12, 2)) for _ in range(batch.curr_agent_state.shape[0])]

def train_epoch(mapping_head, adapter, dataloader, optimizer, device, args):
    mapping_head.train()
    total_loss = 0.0
    scenarios_processed = 0
    batch_count = 0
    
    pbar = tqdm(total=args.num_scenarios, desc="Training Scenarios")
    for batch_idx, batch in enumerate(dataloader):
        if scenarios_processed >= args.num_scenarios:
            break
            
        # Filter (robust check) - Moved to agent loop
        # if not hasattr(batch, 'agent_fut_len') or (batch.agent_fut_len < 12).any():
        #    continue
            
        # Move to GPU
        try:
            with torch.no_grad():
                embeddings = adapter.get_scene_embeddings(batch) # [B, D]
        except Exception as e:
            continue
            
        # [B, 5]
        weights = mapping_head(embeddings)
        
        # 2. Get Samples for Z estimation
        k_samples_list = get_trajectron_samples(adapter, batch, k=25)
        
        batch_loss = 0.0
        valid_agents = 0
        
        for i in range(batch.curr_agent_state.shape[0]):
            # Agent-level Filter
            if hasattr(batch, 'agent_fut_len') and batch.agent_fut_len[i] < 12:
                continue
                
            try:
                # Context
                ctx = extract_driver_context(batch, i, map_api=None)
                
                # GT Features
                gt_traj = batch.agent_fut[i, :12, :2].cpu().numpy()
                gt_feats = compute_driver_atom_features(ctx, gt_traj).as_vector()
                gt_phi = torch.tensor(gt_feats, device=device, dtype=torch.float32)
                
                # Sample Features
                samples = k_samples_list[i] # [K, 12, 2]
                sample_phis_list = []
                for s_traj in samples:
                    f = compute_driver_atom_features(ctx, s_traj).as_vector()
                    sample_phis_list.append(f)
                
                # [K, 5]
                sample_phis = torch.tensor(np.stack(sample_phis_list), device=device, dtype=torch.float32)
                
                # MaxEnt IRL Loss
                w = weights[i]
                
                E_gt = torch.dot(w, gt_phi)
                E_samples = torch.mv(sample_phis, w)
                
                log_Z = torch.logsumexp(-E_samples, dim=0)
                
                loss = E_gt + log_Z
                
                batch_loss += loss
                valid_agents += 1
                
            except Exception as e:
                continue
                
        if valid_agents > 0:
            avg_loss = batch_loss / valid_agents
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()
            total_loss += avg_loss.item()
            batch_count += 1
            
            # Count only valid processed scenarios
            scenarios_processed += valid_agents
            pbar.update(valid_agents) # Optional: manually update if using total=scenarios
            
        # Update postfix
        pbar.set_postfix({"Loss": total_loss / (batch_count + 1e-6), "Count": scenarios_processed})
        
    return total_loss / (batch_count + 1e-6)

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading Data...")
    cfg = NuscenesDatasetConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,
        shuffle=True # Shuffle for training
    )
    bridge = NuscenesTrajdataBridge(cfg)
    dataloader = bridge.get_dataloader()
    
    print("Building Adapter...")
    traj_cfg = TrajectronLoadConfig(
        conf_path=args.trajectron_conf,
        model_dir=args.trajectron_model_dir,
        epoch=args.trajectron_epoch,
        device=args.device,
    )
    adapter = build_trajectron_adapter_from_checkpoint(traj_cfg, embedding_dim=args.embedding_dim, mode='encoder')
    adapter.to(device)
    adapter.eval() # Always freeze encoder
    
    print("Initializing Mapping Head...")
    from camp_core.mapping_heads.hyper_network import HyperNetworkMappingHead
    mapping_head = HyperNetworkMappingHead(
        embedding_dim=args.embedding_dim,
        num_atoms=args.num_atoms,
        hidden_dims=[args.hidden_dim, args.hidden_dim]
    ).to(device)
    
    optimizer = optim.Adam(mapping_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    print("Starting MaxEnt IRL Training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_loss = train_epoch(mapping_head, adapter, dataloader, optimizer, device, args)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f}")
        
    training_time = time.time() - start_time
    
    # Save Model
    save_path = os.path.join(args.output_dir, "camp_lambda_only.pt")
    torch.save({
        "head": mapping_head.state_dict(),
        "head_type": "HyperNetwork"
    }, save_path)
    
    # Save Time info
    time_file = save_path.replace(".pt", "_training_time.json")
    with open(time_file, "w") as f:
        json.dump({"total_time_seconds": training_time}, f)
        
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
