set -euo pipefail
CONDA_ENV="/ocean/projects/tra250008p/slin24/conda_envs/adaptive"
REPO_DIR="/ocean/projects/tra250008p/slin24/MetaLearning/adaptive-prediction"
DATA_DIR="/ocean/projects/tra250008p/slin24/datasets/nuscenes"
CACHE_DIR="$REPO_DIR/data/trajdata_cache"
LOG_DIR="$REPO_DIR/experiments/nuScenes/models"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=disabled
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$REPO_DIR"
mkdir -p "$LOG_DIR"
torchrun --nproc_per_node=1 train_unified.py \
  --eval_every=1 --vis_every=1 \
  --batch_size=256 --eval_batch_size=256 \
  --preprocess_workers=16 \
  --log_dir="$LOG_DIR" --log_tag="nusc_baseline_tpp" \
  --train_epochs=20 \
  --conf="$REPO_DIR/experiments/nuScenes/models/nusc_mm_sec4_tpp-13_Sep_2022_11_06_01/config.json" \
  --trajdata_cache_dir="$CACHE_DIR" \
  --data_loc_dict="{\"nusc_trainval\":\"$DATA_DIR\"}" \
  --train_data="nusc_trainval-train" \
  --eval_data="nusc_trainval-train_val" \
  --history_sec=2.0 --prediction_sec=6.0
