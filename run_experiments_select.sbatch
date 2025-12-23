#!/bin/bash
#SBATCH -J camp_full_pipeline
#SBATCH -A tra250008p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 48:00:00
#SBATCH -o logs/camp_full_%j.out
#SBATCH -e logs/camp_full_%j.err


set -e
source ~/.bashrc
conda activate camp-adaptive

echo "=== Starting Full Experiments (CAMP-Select) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/exp_${TIMESTAMP}"
mkdir -p $LOG_DIR
mkdir -p models/production

# Config
DATA_ROOT="/ocean/projects/tra250008p/slin24/datasets/nuscenes"
TRAJ_CONF="adaptive-prediction/experiments/nuScenes/models/nusc_mm_base_tpp-11_Sep_2022_19_15_45/config.json"
TRAJ_MODEL="adaptive-prediction/experiments/nuScenes/models/nusc_mm_base_tpp-11_Sep_2022_19_15_45"
NUM_WORKERS=8

# Outputs
OFFLINE_WEIGHTS="models/production/offline_weights.npy"
CAMP_MODEL="models/production/camp_select_linear.pt"
RERANKER_GT="models/production/reranker_gt.pt"
RERANKER_SAFE="models/production/reranker_safe.pt"

# 0. Stage 0: Calibrate Atom Scales
echo "[Stage 0] Calibrating Atom Scales..."
python scripts/tools/compute_atom_scales.py \
    --data_root $DATA_ROOT \
    --num_samples 400000 \
    --output_file "models/production/atom_scales.json" \
    --trajectron_conf $TRAJ_CONF \
    --trajectron_model_dir $TRAJ_MODEL \
    --num_workers $NUM_WORKERS \
    > "${LOG_DIR}/stage0.log" 2>&1

echo "  Stage 0 Done."

# 1. Stage 1: Offline Preference
echo "[Stage 1] Training Offline Preference..."
python scripts/train/train_offline_preference.py \
    --data_root $DATA_ROOT \
    --num_scenarios 40000 \
    --num_candidates 30 \
    --num_workers $NUM_WORKERS \
    --output_path $OFFLINE_WEIGHTS \
    > "${LOG_DIR}/stage1.log" 2>&1

echo "  Stage 1 Done."

# 2. Stage 2: CAMP-Select (CVXPY Master)
echo "[Stage 2] Training CAMP-Select..."
python scripts/train/train_camp_select.py \
    --data_root $DATA_ROOT \
    --num_scenarios 40000 \
    --max_iter 100 \
    --batch_size 16 \
    --num_workers $NUM_WORKERS \
    --output_path $CAMP_MODEL \
    --trajectron_conf $TRAJ_CONF \
    --trajectron_model_dir $TRAJ_MODEL \
    --offline_weights_path $OFFLINE_WEIGHTS \
    --risk_type "cvar" \
    --alpha 0.9 \
    --prior_reg 1.0 \
    --anchor_weight 0.1 \
    > "${LOG_DIR}/stage2.log" 2>&1

echo "  Stage 2 Done."

# 3. Stage 3: Reranker Baselines
echo "[Stage 3] Training Reranker (GT Only)..."
python scripts/train/train_reranker.py \
    --data_root $DATA_ROOT \
    --num_scenarios 40000 \
    --epochs 50 \
    --lambda_safe 0.0 \
    --num_workers $NUM_WORKERS \
    --output_path $RERANKER_GT \
    --trajectron_conf $TRAJ_CONF \
    --trajectron_model_dir $TRAJ_MODEL \
    > "${LOG_DIR}/stage3_gt.log" 2>&1

echo "[Stage 3] Training Reranker (GT + Safety)..."
python scripts/train/train_reranker.py \
    --data_root $DATA_ROOT \
    --num_scenarios 40000 \
    --epochs 50 \
    --lambda_safe 1.0 \
    --num_workers $NUM_WORKERS \
    --output_path $RERANKER_SAFE \
    --trajectron_conf $TRAJ_CONF \
    --trajectron_model_dir $TRAJ_MODEL \
    > "${LOG_DIR}/stage3_safe.log" 2>&1

echo "  Stage 3 Done."

# 4. Evaluation
echo "[Eval] Running Comparison..."
# Note: Ensure compare_methods.py is updated!
python scripts/experiments/compare_methods.py \
     --data_root $DATA_ROOT \
     --num_scenarios 40000 \
     --models_dir "models/production" \
     --trajectron_conf $TRAJ_CONF \
     --trajectron_model_dir $TRAJ_MODEL \
     --output_dir "results/${TIMESTAMP}" \
     > "${LOG_DIR}/eval.log" 2>&1

echo "=== Experiments Completed Successfully ==="
