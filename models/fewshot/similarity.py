"""
models/fewshot/similarity.py

Spatial cosine similarity between query features and support prototypes.

For each spatial location (i, j) in the query feature map, this module
computes how similar that feature vector is to the crack prototype and
to the background prototype. The result is a two-channel similarity map
that the decoder uses to produce the final segmentation mask.

Shapes (baseline: ResNet34, 256×256 input):
    query_features: B × 512 × 8 × 8
    proto_crack:    B × 512
    proto_bg:       B × 512
    sim_map:        B × 2   × 8 × 8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from config.base_config import SimilarityConfig


class SimilarityModule(nn.Module):
    """Computes spatial cosine similarity maps between query and prototypes.

    No learnable parameters — pure functional transformation.
    Produces a two-channel map: channel 0 = similarity to crack prototype,
    channel 1 = similarity to background prototype.

    Args:
        cfg: SimilarityConfig with temperature and normalize_query fields.

    Example:
        cfg = SimilarityConfig(temperature=1.0, normalize_query=True)
        module = SimilarityModule(cfg)

        sim_map = module(query_features, proto_crack, proto_bg)
        # sim_map: B × 2 × 8 × 8
    """

    def __init__(self, cfg: SimilarityConfig) -> None:
        super().__init__()
        self.temperature = cfg.temperature
        self.normalize_query = cfg.normalize_query

    def forward(
        self,
        query_features: torch.Tensor,
        proto_crack: torch.Tensor,
        proto_bg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute two-channel spatial similarity map.

        Args:
            query_features: Query feature map of shape B × C × H × W.
                            Typically layer4 output: B × 512 × 8 × 8.
            proto_crack:    Crack prototype of shape B × C.
            proto_bg:       Background prototype of shape B × C.

        Returns:
            Similarity map of shape B × 2 × H × W.
                Channel 0: cosine similarity to crack prototype.
                Channel 1: cosine similarity to background prototype.
        """
        # query_features: B × C × H × W
        _, _, h, w = query_features.shape

        if self.normalize_query:
            # L2-normalize along channel dim so cosine similarity is well-defined.
            # Shape unchanged: B × C × H × W
            query_features = F.normalize(query_features, p=2, dim=1)

        # sim_crack: B × 1 × H × W
        # sim_bg:    B × 1 × H × W
        sim_crack = self._cosine_similarity_map(query_features, proto_crack)
        sim_bg    = self._cosine_similarity_map(query_features, proto_bg)

        # Concatenate along channel dim to form the two-channel similarity map.
        # sim_map: B × 2 × H × W
        sim_map = torch.cat([sim_crack, sim_bg], dim=1)

        return sim_map * self.temperature


    def _cosine_similarity_map(
        self,
        query_features: torch.Tensor,
        prototype: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-location cosine similarity between query and a prototype.

        The prototype is broadcast across all spatial locations of the query.
        Equivalent to a dot product at each (i, j) since both are L2-normalized.

        Args:
            query_features: B × C × H × W — already L2-normalized if normalize_query.
            prototype:      B × C          — already L2-normalized from PrototypeModule.

        Returns:
            Similarity map of shape B × 1 × H × W, values in [-1, 1].
        """
        # Reshape prototype to broadcast across spatial dims.
        # B × C  →  B × C × 1 × 1
        proto_spatial = prototype.unsqueeze(-1).unsqueeze(-1)

        # F.cosine_similarity computes along dim=1 (channel dim).
        # Output shape: B × H × W
        sim = F.cosine_similarity(query_features, proto_spatial, dim=1)

        # Add channel dim to allow concatenation downstream.
        # B × H × W  →  B × 1 × H × W
        return sim.unsqueeze(1)
