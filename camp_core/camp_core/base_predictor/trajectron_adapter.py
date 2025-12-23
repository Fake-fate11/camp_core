from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from trajdata.data_structures.batch import AgentBatch

from trajectron.model.model_utils import ModeKeys


@dataclass
class TrajectronAdapterConfig:
    device: str = "cpu"
    embedding_dim: int = 16
    mode: str = "dummy_mlp"  # "dummy_mlp" or "encoder"
    use_frozen_trajectron: bool = False


class TrajectronAdapter(nn.Module):
    """
    Thin wrapper that turns a trajdata.AgentBatch into per-agent scene embeddings.

    Modes:
      - "dummy_mlp": use a small MLP on curr_agent_state only.
      - "encoder":   use a frozen Trajectron encoder (history + edges + map),
                     then project to embedding_dim; if that fails, fall back to MLP.
    """

    def __init__(
        self,
        cfg: TrajectronAdapterConfig,
        base_model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.embedding_dim = cfg.embedding_dim
        self.mode = cfg.mode
        self.base_model: Optional[nn.Module] = base_model

        # Simple MLP on curr_agent_state -> embedding_dim
        # Assumes curr_agent_state has shape [B, D_curr].
        self.fallback_encoder = nn.Sequential(
            nn.Linear(7, 64),  # 7 is typical for [x, y, vx, vy, ax, ay, heading]
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim),
        )

        # Linear projection on top of Trajectron encoder features.
        # Will be lazily initialized when we see the actual encoder feature dim.
        self.encoder_projection: Optional[nn.Linear] = None

        # If a Trajectron base model is provided, freeze it.
        if self.base_model is not None:
            self.base_model.to(self.device)
            self.base_model.eval()
            for p in self.base_model.parameters():
                p.requires_grad_(False)

        # Move adapter (including fallback encoder and projection) to device.
        self.to(self.device)

        print(
            f"TrajectronAdapter constructed "
            f"(mode={self.mode}, device={self.device}, "
            f"embedding_dim={self.embedding_dim}, "
            f"has_base_model={self.base_model is not None})."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_encoder_projection(self, in_dim: int) -> None:
        """Create or resize the encoder projection layer to match encoder feature dim."""
        if self.encoder_projection is None or self.encoder_projection.in_features != in_dim:
            self.encoder_projection = nn.Linear(in_dim, self.embedding_dim).to(self.device)
            print(
                f"TrajectronAdapter: initialized encoder_projection with in_dim={in_dim}, "
                f"out_dim={self.embedding_dim}."
            )

    def _ensure_dummy_maps(self, batch: AgentBatch, node_model: nn.Module) -> None:
        """
        If batch.maps is None but hyperparams expect map_encoding, create zero map crops
        so that the CNN map encoder can still run and keep dimensions consistent.
        """
        if getattr(batch, "maps", None) is not None:
            return

        if self.base_model is None:
            return

        hyper = getattr(self.base_model, "hyperparams", None)
        if hyper is None:
            return

        map_cfg_all: Dict[str, Dict] = hyper.get("map_encoder", {})
        node_type_str: Optional[str] = getattr(node_model, "node_type", None)
        if node_type_str is None:
            return

        me_params = map_cfg_all.get(node_type_str, None)
        if me_params is None:
            return

        patch_size = int(me_params.get("patch_size", 100))
        map_channels = int(me_params.get("map_channels", 3))

        B = batch.agent_hist.shape[0]
        dummy_maps = torch.zeros(
            (B, map_channels, patch_size, patch_size),
            device=self.device,
            dtype=batch.agent_hist.dtype,
        )
        batch.maps = dummy_maps  # type: ignore[attr-defined]
        print(
            "TrajectronAdapter: batch.maps is None; using zero map crops for map encoder."
        )

    def _get_embeddings_via_mlp(self, batch: AgentBatch) -> torch.Tensor:
        """
        Fallback path: simple MLP on curr_agent_state.
        """
        curr_state = batch.curr_agent_state

        if not isinstance(curr_state, torch.Tensor):
            curr_state = curr_state.to_tensor()  # type: ignore[attr-defined]

        curr_state = curr_state.to(self.device)

        in_dim = curr_state.shape[-1]
        fc0: nn.Linear = self.fallback_encoder[0]  # type: ignore[index]

        if fc0.in_features != in_dim:
            # Rebuild first layer to match current input dim.
            new_fc0 = nn.Linear(in_dim, fc0.out_features).to(self.device)
            with torch.no_grad():
                if fc0.weight.shape == new_fc0.weight.shape:
                    new_fc0.weight.copy_(fc0.weight)
                    if fc0.bias is not None and new_fc0.bias is not None:
                        new_fc0.bias.copy_(fc0.bias)
            self.fallback_encoder[0] = new_fc0  # type: ignore[index]
            print(
                f"TrajectronAdapter: adjusted fallback MLP input dim from "
                f"{fc0.in_features} to {in_dim}."
            )

        emb = self.fallback_encoder(curr_state)
        return emb

    def _get_embeddings_via_trajectron(self, batch: AgentBatch) -> torch.Tensor:
        """
        Encoder mode: call Trajectron's per-node encoder and project that feature.
        This currently uses the VEHICLE node model (if available).
        """
        if self.base_model is None:
            raise RuntimeError("TrajectronAdapter: base_model is None in encoder mode.")

        # 选一个 node model，优先 VEHICLE
        node_models = self.base_model.node_models_dict  # type: ignore[attr-defined]
        if "VEHICLE" in node_models:
            node_model: nn.Module = node_models["VEHICLE"]
        else:
            first_key = next(iter(node_models.keys()))
            node_model = node_models[first_key]
            print(
                f"TrajectronAdapter: VEHICLE node model not found; "
                f"falling back to node_type={first_key}."
            )

        # 就地把 batch 挪到同一个 device 上（AgentBatch.to 返回 None）
        batch.to(self.device)  # type: ignore[call-arg]

        # 确保有 maps（如果 config 开了 map_encoding 但 trajdata 这边没给地图）
        self._ensure_dummy_maps(batch, node_model)

        # Sanitize batch data to prevent NaNs in encoder
        if hasattr(batch, 'neigh_hist') and batch.neigh_hist is not None:
            if torch.isnan(batch.neigh_hist).any():
                # print(f"TrajectronAdapter: Found NaNs in neigh_hist, replacing with 0.")
                batch.neigh_hist = torch.nan_to_num(batch.neigh_hist, nan=0.0)
                
        if hasattr(batch, 'curr_agent_state') and batch.curr_agent_state is not None:
             # curr_agent_state might be a Tensor or StateTuple
            if isinstance(batch.curr_agent_state, torch.Tensor):
                if torch.isnan(batch.curr_agent_state).any():
                    # print(f"TrajectronAdapter: Found NaNs in curr_agent_state, replacing with 0.")
                    batch.curr_agent_state = torch.nan_to_num(batch.curr_agent_state, nan=0.0)

        # 从 Trajectron encoder 拿特征 enc
        enc, _, _, _, _ = node_model.obtain_encoded_tensors(  # type: ignore[call-arg]
            ModeKeys.PREDICT,
            batch,
        )

        # enc 可能是 [B, D] 或 [B, T, D]（adaptive）
        if enc.dim() == 3:
            enc_2d = enc[:, -1, :]
        else:
            enc_2d = enc

        if enc_2d.dim() != 2:
            raise RuntimeError(
                f"TrajectronAdapter: expected encoder feature to be 2D, got shape {enc_2d.shape}."
            )

        # DEBUG: Check enc_2d for NaNs
        if torch.isnan(enc_2d).any():
            print(f"TrajectronAdapter: NaN detected in enc_2d! Shape: {enc_2d.shape}")
            # Check inputs
            if hasattr(batch, 'agent_hist'):
                print(f"TrajectronAdapter: agent_hist has NaNs: {torch.isnan(batch.agent_hist).any()}")
            if hasattr(batch, 'neigh_hist') and batch.neigh_hist is not None:
                print(f"TrajectronAdapter: neigh_hist has NaNs: {torch.isnan(batch.neigh_hist).any()}")
            if hasattr(batch, 'maps') and batch.maps is not None:
                print(f"TrajectronAdapter: maps has NaNs: {torch.isnan(batch.maps).any()}")
            if hasattr(batch, 'robot_fut') and batch.robot_fut is not None:
                print(f"TrajectronAdapter: robot_fut has NaNs: {torch.isnan(batch.robot_fut).any()}")

        in_dim = enc_2d.shape[-1]
        self._ensure_encoder_projection(in_dim)

        emb = self.encoder_projection(enc_2d)  # type: ignore[operator]
        
        # Check for NaN values in embeddings and replace with zeros
        if torch.isnan(emb).any():
            nan_count = torch.isnan(emb).sum().item()
            total_elements = emb.numel()
            print(
                f"TrajectronAdapter: WARNING - Found {nan_count}/{total_elements} NaN values in embeddings. "
                f"Replacing with zeros."
            )
            emb = torch.nan_to_num(emb, nan=0.0)
        
        return emb


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_scene_embeddings(self, batch: AgentBatch) -> torch.Tensor:
        """
        Compute per-agent embeddings for a trajdata.AgentBatch.

        - If mode == "encoder" and a frozen Trajectron is present, use its encoder.
        - If that fails for any reason (e.g., device mismatch, missing fields),
          or if mode != "encoder", fall back to the dummy MLP on curr_agent_state.
        """
        if self.mode in ["encoder", "decoder"] and self.base_model is not None:
            try:
                return self._get_embeddings_via_trajectron(batch)
            except Exception as e:
                print(
                    "TrajectronAdapter: encoder mode failed with "
                    f"{type(e).__name__}: {e}. Falling back to dummy MLP."
                )

        return self._get_embeddings_via_mlp(batch)

    @torch.no_grad()
    def embed_batch(self, batch: AgentBatch) -> Dict[str, torch.Tensor]:
        """
        Convenience wrapper used by evaluation scripts.
        """
        emb = self.get_scene_embeddings(batch)
        return {"scene_embeddings": emb}
