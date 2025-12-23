# camp_core/base_predictor/trajectron_loader.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from trajectron.model.model_registrar import ModelRegistrar
from trajectron.model.trajectron import Trajectron
from camp_core.base_predictor.trajectron_adapter import (
    TrajectronAdapter,
    TrajectronAdapterConfig,
)


@dataclass
class TrajectronLoadConfig:
    """Configuration for loading a pretrained Trajectron++ model."""
    conf_path: str          # Path to the JSON hyperparameter config used in training
    model_dir: str          # Directory containing model_registrar-<epoch>.pt
    epoch: int              # Which checkpoint epoch to load
    device: str = "cpu"     # Device string, e.g., "cpu" or "cuda:0"


def load_hyperparams(conf_path: str, device: str) -> Dict[str, Any]:
    """Load hyperparameters from JSON and set device."""
    with open(conf_path, "r", encoding="utf-8") as f:
        hyperparams: Dict[str, Any] = json.load(f)

    # For inference we override device explicitly.
    hyperparams["device"] = device
    return hyperparams


def build_trajectron_from_checkpoint(cfg: TrajectronLoadConfig) -> Trajectron:
    """Instantiate Trajectron and load weights from a training checkpoint.

    This mirrors the construction pattern used in train_unified.py but
    skips any training-specific logic or distributed setup.
    """
    device_obj = torch.device(cfg.device)

    hyperparams = load_hyperparams(cfg.conf_path, cfg.device)

    # Match the training script's constructor usage
    model_registrar = ModelRegistrar(cfg.model_dir, hyperparams["device"])

    trajectron = Trajectron(
        model_registrar=model_registrar,
        hyperparams=hyperparams,
        log_writer=None,
        device=hyperparams["device"],
    )

    # Build the per-node models (MultimodalGenerativeCVAE instances)
    trajectron.set_environment()
    trajectron.set_annealing_params()

    ckpt_path = Path(cfg.model_dir) / f"model_registrar-{cfg.epoch}.pt"
    state = torch.load(ckpt_path, map_location=device_obj)

    if "model_state_dict" not in state:
        raise RuntimeError(
            f"Checkpoint at {ckpt_path} does not contain 'model_state_dict'. "
            f"Please check that it was produced by train_unified.py."
        )

    trajectron.load_state_dict(state["model_state_dict"])

    trajectron.to(device_obj)
    trajectron.eval()
    for p in trajectron.parameters():
        p.requires_grad_(False)

    return trajectron


def build_trajectron_adapter_from_checkpoint(
    load_cfg: "TrajectronLoadConfig",
    embedding_dim: int,
    mode: str = "dummy_mlp",
) -> TrajectronAdapter:
    """
    Load a frozen Trajectron base model from disk and wrap it
    in a TrajectronAdapter that produces per-agent embeddings.

    For now, the adapter still uses a simple MLP on curr_agent_state.
    The base model is stored and frozen for later use.
    """
    base_model = build_trajectron_from_checkpoint(load_cfg)

    adapter_cfg = TrajectronAdapterConfig(
        device=load_cfg.device,
        embedding_dim=embedding_dim,
        mode=mode,
        use_frozen_trajectron=True,
    )

    adapter = TrajectronAdapter(cfg=adapter_cfg, base_model=base_model)
    return adapter