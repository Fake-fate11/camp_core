
import argparse
import os
import sys
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict, Any

# Sklearn imports for metrics
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from camp_core.base_predictor.trajectron_loader import build_trajectron_adapter_from_checkpoint, TrajectronLoadConfig
from trajdata import UnifiedDataset

# ---------------------------------------------------------------------------
# Helper: PyTorch Probe Trainer (for GLS/QG with KL Loss)
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)

class PolynomialProbe(nn.Module):
    def __init__(self, input_dim, output_dim, degree=2):
        super().__init__()
        self.degree = degree
        self.linear = nn.Linear(input_dim, output_dim) # Input will be lifted
        
    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)

def train_probe(model, X_train, y_train, X_val, y_val, epochs=100, lr=1e-2, device="cuda"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        # KLDivLoss expects log-probs
        loss = criterion(torch.log(pred + 1e-8), y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred_val = model(X_val_t)
            val_loss = criterion(torch.log(pred_val + 1e-8), y_val_t).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
    return best_val_loss

# ---------------------------------------------------------------------------
# Metrics Implementation
# ---------------------------------------------------------------------------

def compute_gls(X_train, y_train, X_val, y_val, device):
    """
    GLS: Linear Separability Score.
    Train Linear Probe. Measure Performance (1 - KL) and Gap.
    """
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = LinearProbe(input_dim, output_dim)
    
    # Train
    val_kl = train_probe(model, X_train, y_train, X_val, y_val, device=device)
    
    # Also compute Train KL for Gap
    model.eval()
    with torch.no_grad():
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        pred_train = model(X_train_t)
        train_kl = nn.KLDivLoss(reduction="batchmean")(torch.log(pred_train + 1e-8), y_train_t).item()
        
    # Perf_L = exp(-KL) or just 1 / (1 + KL)? 
    # User says "1 - KL". KL can be > 1.
    # Let's use exp(-KL) as a bounded [0, 1] metric where 1 is perfect.
    perf_l = np.exp(-val_kl)
    perf_l_train = np.exp(-train_kl)
    
    gap_l = max(0, perf_l_train - perf_l)
    
    lambda_gap = 0.5
    gls = np.clip(perf_l - lambda_gap * gap_l, 0, 1)
    
    return gls, perf_l

def compute_qg(X_train, y_train, X_val, y_val, perf_l, device):
    """
    QG: Quadratic Gain.
    Train Polynomial Probe. Measure gain over Linear.
    """
    # Lift features (Degree 2)
    # Use sklearn for lifting
    poly = PolynomialFeatures(degree=2, include_bias=False)
    # Note: This might be memory intensive if N is large.
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    
    input_dim = X_train_poly.shape[1]
    output_dim = y_train.shape[1]
    
    model = PolynomialProbe(input_dim, output_dim) # Just a linear layer on lifted features
    
    val_kl = train_probe(model, X_train_poly, y_train, X_val_poly, y_val, device=device)
    
    perf_p = np.exp(-val_kl)
    
    qg_raw = perf_p - perf_l
    epsilon = 1e-6
    qg = np.clip(qg_raw / max(perf_l, epsilon), 0, 1)
    
    return qg, perf_p

def compute_lci(X_val):
    """
    LCI: Local Curvature Index.
    kNN PCA reconstruction error.
    """
    k = 20
    if len(X_val) < k + 1:
        return 0.0 # Not enough data
        
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_val)
    distances, indices = nbrs.kneighbors(X_val)
    
    errors = []
    for i in range(len(X_val)):
        # Local neighborhood (excluding self if included, usually kNN includes self as 0 dist)
        # indices[i] has k neighbors.
        neighbors = X_val[indices[i]]
        
        # PCA on neighbors
        pca = PCA(n_components=0.95) # Keep 95% variance
        pca.fit(neighbors)
        
        # Reconstruct self (X_val[i])
        xi = X_val[i].reshape(1, -1)
        xi_proj = pca.inverse_transform(pca.transform(xi))
        
        norm_xi = np.linalg.norm(xi)
        if norm_xi < 1e-6:
            err = 0.0
        else:
            err = np.linalg.norm(xi - xi_proj) / norm_xi
        errors.append(err)
        
    mean_error = np.mean(errors)
    
    # Normalize. User says "train min-max or robust z-score sigmoid".
    # Since we only have one set of errors here, we can just use a heuristic or sigmoid.
    # Let's use a simple sigmoid: 1 / (1 + exp(-(x - mu)/sigma))
    # Or just clip if we expect error to be small.
    # Let's assume error > 0.1 is "high curvature".
    # LCI = clip(mean_error * 5, 0, 1) ?
    # Let's use a sigmoid centered at 0.1
    lci = 1 / (1 + np.exp(-(mean_error - 0.1) * 20))
    
    return lci

def compute_kas(X_train, y_train, X_val, y_val):
    """
    KAS: Kernel Advantage Score.
    Kernel Ridge (RBF) vs Linear Regression (MSE).
    """
    # Linear Regression
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_val)
    mse_lin = mean_squared_error(y_val, y_pred_lin)
    
    # Kernel Ridge
    # Heuristic gamma: 1 / n_features
    gamma = 1.0 / X_train.shape[1]
    krr = KernelRidge(kernel="rbf", gamma=gamma, alpha=0.1)
    krr.fit(X_train, y_train)
    y_pred_krr = krr.predict(X_val)
    mse_krr = mean_squared_error(y_val, y_pred_krr)
    
    # Convert MSE to "Performance" (higher is better)
    # Perf = 1 / (1 + MSE)
    perf_l = 1 / (1 + mse_lin)
    perf_k = 1 / (1 + mse_krr)
    
    epsilon = 1e-6
    kas = np.clip((perf_k - perf_l) / max(perf_l, epsilon), 0, 1)
    
    return kas

def compute_id(X_train):
    """
    ID: Intrinsic Dimension.
    Effective Rank.
    """
    pca = PCA()
    pca.fit(X_train)
    sv = pca.singular_values_
    
    # Effective Rank
    # sum(sv)^2 / sum(sv^2)
    # Note: sv are singular values. Eigenvalues are sv^2 / (N-1).
    # Formula usually uses singular values or eigenvalues.
    # User says "eigenvalues sigma_j".
    evals = pca.explained_variance_ # These are eigenvalues
    
    sum_evals = np.sum(evals)
    sum_sq_evals = np.sum(evals**2)
    
    if sum_sq_evals < 1e-9:
        return 0.0
        
    er = (sum_evals**2) / sum_sq_evals
    
    # Normalize by dimension d
    d = X_train.shape[1]
    id_score = np.clip(er / d, 0, 1)
    
    return id_score

# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/training_data.pkl")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="nusc_trainval-train")
    parser.add_argument("--trajectron_conf", type=str, required=True)
    parser.add_argument("--trajectron_model_dir", type=str, required=True)
    parser.add_argument("--trajectron_epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding extraction")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--num_scenarios", type=int, default=None, help="Limit number of scenarios (for debugging)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Generated Weights
    if not os.path.exists(args.data_path):
        print(f"Error: {args.data_path} not found.")
        return
        
    with open(args.data_path, "rb") as f:
        raw_data = pickle.load(f)
        
    # Build Lookup: (scene_id, agent_id) -> weights
    weight_lookup = {}
    for item in raw_data:
        # Check if agent_id exists (it should after our fix)
        if "agent_id" in item:
            # agent_id is likely a string (e.g. "ego", "1")
            key = (str(item["scene_id"]), str(item["agent_id"]))
            weight_lookup[key] = item["optimal_weights"]
        else:
            # Fallback for old data? We can't reliably match without agent_id if multiple agents.
            # But if user re-generated, it should be there.
            pass
            
    print(f"Loaded {len(weight_lookup)} weight samples.")
    if len(weight_lookup) == 0:
        print("No valid samples found in training data (missing agent_id?). Please re-generate data.")
        return

    # 2. Load Trajectron Adapter
    load_cfg = TrajectronLoadConfig(
        conf_path=args.trajectron_conf,
        model_dir=args.trajectron_model_dir,
        epoch=args.trajectron_epoch,
        device=device
    )
    adapter = build_trajectron_adapter_from_checkpoint(load_cfg, embedding_dim=64, mode="encoder")
    
    # 3. Load Dataset and Extract Embeddings
    print("Loading dataset to extract embeddings...")
    dataset = UnifiedDataset(
        desired_data=[args.split],
        data_dirs={
            "nusc_mini": args.data_root,
            "nusc_trainval": args.data_root
        },
        history_sec=(2.0, 2.0),
        future_sec=(6.0, 6.0),
        incl_vector_map=True, # Enable vector map loading (matching generate_training_data.py)
        vector_map_params={"collate": True},
        incl_raster_map=True, # Enable raster map loading for CNN encoder
        raster_map_params={"px_per_m": 2, "map_size_px": 100, "offset_frac_xy": (-0.75, 0.0)}, # Standard params matching Trajectron config
    )
    
    # Keep reference to original dataset for collate_fn
    original_dataset = dataset
    
    if args.num_scenarios is not None:
        if len(dataset) > args.num_scenarios:
            dataset = torch.utils.data.Subset(dataset, range(args.num_scenarios))
            print(f"Limited dataset to {args.num_scenarios} scenarios.")
            
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=original_dataset.get_collate_fn(pad_format="right")
    )
    
    embeddings_list = []
    weights_list = []
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Optimization: Check if any agent in this batch is in our target set
            # BEFORE running the expensive model forward pass
            has_target_agent = False
            for i in range(len(batch.agent_name)):
                sid = str(batch.scene_ids[i])
                aid = str(batch.agent_name[i])
                if (sid, aid) in weight_lookup:
                    has_target_agent = True
                    break
            
            if not has_target_agent:
                continue

            # Get embeddings for WHOLE batch first (efficient)
            try:
                embs = adapter.get_scene_embeddings(batch) # [B, D]
            except Exception as e:
                print(f"Embedding extraction failed: {e}")
                continue
                
            embs_np = embs.cpu().numpy()
            
            for i in range(len(batch.agent_name)):
                sid = str(batch.scene_ids[i])
                aid = str(batch.agent_name[i]) 
                key = (sid, aid)
                
                if key in weight_lookup:
                    w = weight_lookup[key]
                    # Check for NaNs in w
                    if np.isnan(w).any():
                        continue
                        
                    embeddings_list.append(embs_np[i])
                    weights_list.append(w)
            
            # Memory cleanup
            del batch
            del embs
            del embs_np
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            # Early stopping: if we've found enough samples (e.g., 2x the expected), stop searching
            # This prevents scanning the entire dataset when we already have sufficient data
            expected_samples = len(weight_lookup)
            if len(embeddings_list) >= min(expected_samples * 1.5, 10000):
                print(f"Found {len(embeddings_list)} samples (sufficient), stopping early.")
                break
                    
    if len(embeddings_list) == 0:
        print("No matching samples found between dataset and generated weights.")
        return
        
    X = np.stack(embeddings_list)
    y = np.stack(weights_list)
    
    print(f"Matched {len(X)} samples.")
    
    # Check for NaNs
    nan_mask = np.isnan(X).any(axis=1)
    nan_count = nan_mask.sum()
    
    if nan_count > 0:
        nan_ratio = nan_count / len(X)
        print(f"Warning: Found {nan_count} samples ({nan_ratio:.2%}) with NaNs in embeddings.")
        
        if nan_ratio > 0.2:
            print("CRITICAL WARNING: High NaN rate (>20%). This suggests an issue with the Trajectron model or data loading.")
            
        print("Dropping samples with NaNs to preserve manifold geometry for analysis...")
        X = X[~nan_mask]
        y = y[~nan_mask]
        print(f"Remaining samples: {len(X)}")
        
        if len(X) == 0:
            print("Error: No valid samples remaining after dropping NaNs.")
            return
    
    # 4. Split Train/Val
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print(f"Data split: Train {len(X_train)}, Val {len(X_val)}")
    
    # DEBUG: Check data statistics
    print("\n--- Data Statistics ---")
    print(f"X_train: shape={X_train.shape}, mean={np.mean(X_train):.4f}, std={np.std(X_train):.4f}, min={np.min(X_train):.4f}, max={np.max(X_train):.4f}")
    if np.isnan(X_train).any():
        print("WARNING: X_train contains NaNs!")
        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)
        
    print(f"y_train: shape={y_train.shape}, mean={np.mean(y_train):.4f}, std={np.std(y_train):.4f}, min={np.min(y_train):.4f}, max={np.max(y_train):.4f}")
    if np.isnan(y_train).any():
        print("WARNING: y_train contains NaNs!")
    
    # 5. Compute Metrics
    print("\n--- Computing Metrics ---")
    
    # GLS
    print("Computing GLS...")
    gls, perf_l = compute_gls(X_train, y_train, X_val, y_val, device)
    print(f"GLS: {gls:.4f} (Perf_L: {perf_l:.4f})")
    
    # QG
    print("Computing QG...")
    qg, perf_p = compute_qg(X_train, y_train, X_val, y_val, perf_l, device)
    print(f"QG: {qg:.4f} (Perf_P: {perf_p:.4f})")
    
    # LCI
    print("Computing LCI...")
    lci = compute_lci(X_val)
    print(f"LCI: {lci:.4f}")
    
    # KAS
    print("Computing KAS...")
    kas = compute_kas(X_train, y_train, X_val, y_val)
    print(f"KAS: {kas:.4f}")
    
    # ID
    print("Computing ID...")
    id_score = compute_id(X_train)
    print(f"ID: {id_score:.4f}")
    
    # 6. Scoring & Decision
    print("\n--- Decision ---")
    
    score_l = 0.55 * gls + 0.25 * (1 - lci) + 0.20 * (1 - id_score)
    score_p = 0.45 * qg + 0.30 * kas + 0.25 * (1 - lci)
    score_h = 0.40 * lci + 0.35 * id_score + 0.25 * (1 - gls)
    
    print(f"Score_L: {score_l:.4f}")
    print(f"Score_P: {score_p:.4f}")
    print(f"Score_H: {score_h:.4f}")
    
    decision = "H" # Default
    
    if score_l >= max(score_p, score_h) and gls >= 0.7 and qg <= 0.1:
        decision = "L"
    elif score_p >= max(score_l, score_h) and (qg >= 0.15 or kas >= 0.15):
        decision = "P"
    else:
        decision = "H"
        
    print(f"\nRecommended Head: {decision}")
    
    # Map to class names
    head_map = {
        "L": "SimplexLinear",
        "P": "LiftedLinear",
        "H": "HyperNetwork"
    }
    print(f"Model Class: {head_map[decision]}")
    
    # 7. Save Recommendation to File
    recommendation = {
        "recommended_head": decision,
        "head_class": head_map[decision],
        "scores": {
            "SimplexLinear": float(score_l),
            "LiftedLinear": float(score_p),
            "HyperNetwork": float(score_h)
        },
        "metrics": {
            "GLS": float(gls),
            "QG": float(qg),
            "LCI": float(lci),
            "KAS": float(kas),
            "ID": float(id_score)
        }
    }
    
    output_file = "selected_head_type.txt"
    with open(output_file, "w") as f:
        f.write(head_map[decision])
    print(f"\nSaved recommendation to {output_file}")
    
    # Save detailed JSON
    json_file = "head_selection_results.json"
    with open(json_file, "w") as f:
        json.dump(recommendation, f, indent=2)
    print(f"Saved detailed results to {json_file}")
    
    # 8. Generate Visualization
    print("\nGenerating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Head Type Scores
    head_names = ["SimplexLinear\n(L)", "LiftedLinear\n(P)", "HyperNetwork\n(H)"]
    scores = [score_l, score_p, score_h]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax1.bar(head_names, scores, color=colors, alpha=0.7, edgecolor='black')
    
    # Highlight recommended
    recommended_idx = list(head_map.values()).index(head_map[decision])
    bars[recommended_idx].set_edgecolor('gold')
    bars[recommended_idx].set_linewidth(3)
    
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Head Type Scores Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Metrics Radar/Bar Chart
    metric_names = ['GLS', 'QG', 'LCI', 'KAS', 'ID']
    metric_values = [gls, qg, lci, kas, id_score]
    
    bars2 = ax2.barh(metric_names, metric_values, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Score', fontsize=12)
    ax2.set_title('Embedding Manifold Metrics', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1.0)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, metric_values)):
        ax2.text(val, i, f' {val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    viz_file = "head_selection_comparison.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {viz_file}")
    plt.close()

if __name__ == "__main__":
    main()
