# CAMP Core: Conic Atom Meta Policy

This repository implements the CAMP algorithm for adapting motion prediction models using a bi-level optimization framework.

## Installation

1.  **Environment Setup**:
    Ensure you have a Python 3.9+ environment.
    ```bash
    conda create -n camp python=3.9
    conda activate camp
    pip install -r requirements.txt
    ```

2.  **Install Package**:
    Install this package in editable mode:
    ```bash
    pip install -e .
    ```
    *Note: This is crucial to ensure imports resolve correctly.*

## Pipeline Overview

The pipeline consists of the following stages, orchestrated by shell scripts.

### 1. Zero-Stage: Atom Scale Calibration
**Script**: `scripts/tools/compute_atom_scales.py`
Computes the normalization scales (P90) for all driver atoms (jerk, acceleration, etc.) to ensure consistent metric spaces.
*   **Input**: NuScenes Dataset
*   **Output**: `models/production/atom_scales.json`

### 2. Stage 1: Offline Preference Learning
**Script**: `scripts/train/train_offline_preference.py`
Learns a static, robust set of weights ("Anchor Weights") from the dataset that best explain average driver behavior. Represents the prior $w_0$.
*   **Input**: `atom_scales.json`
*   **Output**: `models/production/offline_weights.npy`

### 3. Stage 2: Bi-Level Adaptation (CAMP-Select)
**Script**: `scripts/train/train_camp_select.py`
The core meta-learning stage. Trains a linear mapping head to adapt weights $w$ per-instance to minimize both the upper-level loss (ADE/FDE) and deviation from the prior.
*   **Uses**: GPU-Accelerated Benders Decomposition (ParametricTorchMaster).
*   **Input**: `offline_weights.npy`, `atom_scales.json`
*   **Output**: `models/production/camp_select_linear.pt`

### 4. Stage 3: Reranker Training (Optional Baseline)
**Script**: `scripts/train/train_reranker.py`
Trains a standard reranker baseline for comparison.

### 5. Evaluation
**Script**: `scripts/experiments/compare_methods.py`
Compares CAMP against baselines (Offline, Reranker, GT) on the validation set.

## Running Experiments

Use the provided shell scripts (now moved to root) to run the full pipeline:

*   **Debug/Local Run**:
    ```bash
    bash debug_camp_select_pipeline.sh
    ```
*   **Slurm Cluster Run**:
    ```bash
    sbatch run_experiments_select.sbatch
    ```

## Directory Structure

*   `camp_core/`: Core package source.
    *   `atoms/`: Atom definitions and vectorized computation.
    *   `outer_master/`: Benders decomposition master problem solvers (Torch/CVXPY).
    *   `inner_solver/`: Lower-level trajectory optimization solvers.
    *   `mapping_heads/`: Neural network heads for weight prediction.
    *   `data_interfaces/`: Bridges to NuScenes/Trajdata.
*   `scripts/`: Execution scripts (Training, Tools, Experiments).
