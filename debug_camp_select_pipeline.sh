#!/bin/bash
set -e

# Debug Script for CAMP-Select Pipeline
# Usage: ./debug_camp_select_pipeline.sh

echo "=== Starting Debug Pipeline (CAMP-Select) ==="
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/debug_${TIMESTAMP}"
mkdir -p $LOG_DIR
mkdir -p models/debug

# Config
DATA_ROOT="/ocean/projects/tra250008p/slin24/datasets/nuscenes"
TRAJ_CONF="adaptive-prediction/experiments/nuScenes/models/nusc_mm_base_tpp-11_Sep_2022_19_15_45/config.json"
TRAJ_MODEL="adaptive-prediction/experiments/nuScenes/models/nusc_mm_base_tpp-11_Sep_2022_19_15_45"
OFFLINE_WEIGHTS="models/debug/offline_weights.npy"
CAMP_MODEL="models/debug/camp_select_linear.pt"
RERANKER_GT="models/debug/reranker_gt.pt"
RERANKER_SAFE="models/debug/reranker_safe.pt"
NUM_WORKERS=2

# 0. Stage 0: Calibrate Atom Scales
echo "[Stage 0] Calibrating Atom Scales (Max Candidates)..."
python scripts/tools/compute_atom_scales.py \
    --data_root $DATA_ROOT \
    --num_samples 500 \
    --output_file "models/production/atom_scales.json" \
    --trajectron_conf $TRAJ_CONF \
    --trajectron_model_dir $TRAJ_MODEL \
    --num_workers $NUM_WORKERS \
    > "${LOG_DIR}/stage0.log" 2>&1

# 1. Stage 1: Offline Preference (Reuse existing script, maybe with fewer samples)
echo "[Stage 1] Training Offline Preference..."
rm -f $OFFLINE_WEIGHTS # Force regenerate strict R=4 weights
python scripts/train/train_offline_preference.py \
    --data_root $DATA_ROOT \
    --num_scenarios 50 \
    --num_candidates 30 \
    --output_path $OFFLINE_WEIGHTS \
    --num_workers $NUM_WORKERS \
    > "${LOG_DIR}/stage1.log" 2>&1

echo "  Stage 1 Done."

# 2. Stage 2: CAMP-Select (CVXPY Master)
echo "[Stage 2] Training CAMP-Select..."
python scripts/train/train_camp_select.py \
    --data_root $DATA_ROOT \
    --num_scenarios 50 \
    --max_iter 15 \
    --batch_size 2 \
    --output_path $CAMP_MODEL \
    --trajectron_conf $TRAJ_CONF \
    --trajectron_model_dir $TRAJ_MODEL \
    --offline_weights_path $OFFLINE_WEIGHTS \
    --risk_type "cvar" \
    --alpha 0.9 \
    --prior_reg 1.0 \
    --anchor_weight 0.1 \
    --num_workers $NUM_WORKERS \
    > "${LOG_DIR}/stage2.log" 2>&1

echo "  Stage 2 Done."

# Clear Cache for Reranker Debugging (Request from User)
rm -f models/debug/reranker_train_data.pt

# 3. Stage 3: Reranker Baselines
echo "[Stage 3] Training Reranker (GT Only)..."
python scripts/train/train_reranker.py \
    --data_root $DATA_ROOT \
    --num_scenarios 50 \
    --epochs 10 \
    --lambda_safe 0.0 \
    --output_path $RERANKER_GT \
    --trajectron_conf $TRAJ_CONF \
    --trajectron_model_dir $TRAJ_MODEL \
    --num_workers $NUM_WORKERS \
    > "${LOG_DIR}/stage3_gt.log" 2>&1

echo "[Stage 3] Training Reranker (GT + Safety)..."
python scripts/train/train_reranker.py \
    --data_root $DATA_ROOT \
    --num_scenarios 50 \
    --epochs 10 \
    --lambda_safe 1.0 \
    --output_path $RERANKER_SAFE \
    --trajectron_conf $TRAJ_CONF \
    --trajectron_model_dir $TRAJ_MODEL \
    --num_workers $NUM_WORKERS \
    > "${LOG_DIR}/stage3_safe.log" 2>&1

echo "  Stage 3 Done."

# 4. Evaluation (Placeholder until compare_methods updated)
# 4. Evaluation
echo "[Eval] Running Comparison..."
python scripts/experiments/compare_methods.py \
    --data_root $DATA_ROOT \
    --num_scenarios 50 \
    --models_dir "models/debug" \
    --trajectron_conf $TRAJ_CONF \
    --trajectron_model_dir $TRAJ_MODEL \
    --output_dir "${LOG_DIR}/eval_results" \
    --num_workers $NUM_WORKERS \
    > "${LOG_DIR}/eval.log" 2>&1

echo "  Eval Done. Results in ${LOG_DIR}/eval_results"

echo "=== Debug Pipeline Completed Successfully ==="
echo "Logs available in $LOG_DIR"
