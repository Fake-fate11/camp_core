# CAMP Core: Corrective Adaptation for Motion Prediction

This repository implements the CAMP algorithm for adapting motion prediction models using a bi-level optimization framework.

## Installation

1.  **Environment Setup**:
    Ensure you have a Python 3.9+ environment.
    ```bash
    conda create -n camp python=3.9
    conda activate camp
    # Install comprehensive dependencies (CAMP + Trajectron + NuScenes + Tools)
    pip install -r requirements.txt
    pip install --no-dependencies l5kit==1.5.0
    ```

2.  **Install Package**:
    Install this package in editable mode from the project root:
    ```bash
    cd adaptive-prediction
    cd unified-av-data-loader
    pip install -e .
    cd ..
    pip install -e .
    cd ..
    pip install -e camp_core
    ```
    *Note: Installing as a package ensures all imports relative to `camp_core` resolve correctly.*

## Project Structure

The project is organized as follows:

*   `camp_core/`: The core Python package containing logic for Atoms, Solvers, and Mapping Heads.
    *   `atoms/`: Differentiable driver atom logic.
    *   `outer_master/`: GPU-Accelerated Benders Decomposition Master.
    *   `inner_solver/`: Trajectory optimization solvers.
*   `scripts/`: Execution scripts located at the project root (`MetaLearning/scripts`) to ensure compatibility with Trajectron++ model paths.
    *   `tools/`: Atom scaling tools.
    *   `train/`: Training scripts for Offline Preference and CAMP-Select.
    *   `experiments/`: Evaluation scripts.

## Pipeline Overview

### 1. Zero-Stage: Atom Scale Calibration
**Script**: `scripts/tools/compute_atom_scales.py`
Computes the normalization scales (P90) for all driver atoms.
*   **Input**: NuScenes Dataset
*   **Output**: `models/production/atom_scales.json`

### 2. Stage 1: Offline Preference Learning
**Script**: `scripts/train/train_offline_preference.py`
Learns the prior weights $w_0$ (Anchor Weights).
*   **Input**: `atom_scales.json`
*   **Output**: `models/production/offline_weights.npy`

### 3. Stage 2: Bi-Level Adaptation (CAMP-Select)
**Script**: `scripts/train/train_camp_select.py`
The core meta-learning stage. Trains the Linear Mapping Head using GPU-Accelerated Benders Decomposition.
*   **Input**: `offline_weights.npy`, `atom_scales.json`
*   **Output**: `models/production/camp_select_linear.pt`

### 4. Stage 3: Reranker Training
**Script**: `scripts/train/train_reranker.py`
Trains a standard reranker baseline.

### 5. Evaluation
**Script**: `scripts/experiments/compare_methods.py`
Compares CAMP against baselines on the validation set.

## Running Experiments

Run the pipeline from the project root (`MetaLearning/`):

*   **Debug/Local Run**:
    ```bash
    bash debug_camp_select_pipeline.sh
    ```
*   **Slurm Cluster Run**:
    ```bash
    sbatch run_experiments_select.sbatch
    ```
