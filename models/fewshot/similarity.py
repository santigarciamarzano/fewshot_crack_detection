"""
models/fewshot/similarity.py

Similitud coseno espacial entre las features de la query y los prototipos del support.

Para cada posición espacial (i, j) del mapa de features de la query, calcula
qué tan similar es ese vector a los prototipos de grieta y fondo.
El resultado es un mapa de similitud de 2 canales que el decoder usa para
producir la máscara de segmentación final.

Shapes (baseline: ResNet34, entrada 256×256):
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
    """Calcula mapas de similitud coseno espacial entre la query y los prototipos.

    No tiene parámetros aprendibles: transformación funcional pura.
    Produce un mapa de 2 canales: canal 0 = similitud a grieta, canal 1 = similitud a fondo.

    Args:
        cfg: SimilarityConfig con los campos temperature y normalize_query.
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
        """Devuelve el mapa de similitud de 2 canales: [sim_grieta, sim_fondo]."""
        _, _, h, w = query_features.shape

        if self.normalize_query:
            query_features = F.normalize(query_features, p=2, dim=1)

        sim_crack = self._cosine_similarity_map(query_features, proto_crack)  # B × 1 × H × W
        sim_bg    = self._cosine_similarity_map(query_features, proto_bg)      # B × 1 × H × W

        sim_map = torch.cat([sim_crack, sim_bg], dim=1)  # B × 2 × H × W
        return sim_map * self.temperature


    def _cosine_similarity_map(
        self,
        query_features: torch.Tensor,
        prototype: torch.Tensor,
    ) -> torch.Tensor:
        """Similitud coseno por posición entre la query y un prototipo. Devuelve B × 1 × H × W."""
        proto_spatial = prototype.unsqueeze(-1).unsqueeze(-1)  # B×C → B×C×1×1
        sim = F.cosine_similarity(query_features, proto_spatial, dim=1)  # B × H × W
        return sim.unsqueeze(1)  # B × 1 × H × W
