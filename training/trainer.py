"""
training/trainer.py

Loop de entrenamiento y validación para el modelo few-shot.

El Trainer coordina el modelo, los dataloaders, la loss, el optimizador
y el scheduler. No tiene conocimiento de los internos del modelo.

Métodos principales:
    - train_epoch(): un pase completo sobre el dataloader de entrenamiento
    - val_epoch():   un pase completo sobre el dataloader de validación
    - fit():         coordina épocas, scheduling, checkpointing y logging
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.base_config import FewShotConfig
from training.losses import DiceBCELoss
from training.metrics import binary_iou, binary_dice


class Trainer:
    """Trainer episódico para el modelo de segmentación few-shot.

    Gestiona el ciclo completo de entrenamiento: optimizador, scheduler,
    gradient clipping, checkpointing y logging.

    Args:
        model:        Instancia del FewShotModel a entrenar.
        cfg:          FewShotConfig raíz.
        train_loader: DataLoader de episodios de entrenamiento.
        val_loader:   DataLoader de episodios de validación. Opcional.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: FewShotConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        self.model        = model
        self.cfg          = cfg
        self.train_loader = train_loader
        self.val_loader   = val_loader

        self.device = torch.device(cfg.training.device)
        self.model.to(self.device)

        self.criterion = DiceBCELoss(cfg.loss)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.checkpoint_dir = Path(cfg.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.best_val_iou = 0.0
        self.global_step  = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Ejecuta el loop completo de entrenamiento. Guarda checkpoint cuando mejora el IoU de validación."""
        cfg = self.cfg.training

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            train_loss, train_iou, train_dice = self.train_epoch()

            log = (
                f"Epoch {epoch}/{cfg.epochs} "
                f"| train_loss {train_loss:.4f} "
                f"| train_iou {train_iou:.4f} "
                f"| train_dice {train_dice:.4f} "
                f"| {time.time() - t0:.1f}s"
            )

            if self.val_loader is not None:
                val_loss, val_iou, val_dice = self.val_epoch()
                log += (
                    f"  ||  val_loss {val_loss:.4f} "
                    f"| val_iou {val_iou:.4f} "
                    f"| val_dice {val_dice:.4f}"
                )

                if val_iou > self.best_val_iou:
                    self.best_val_iou = val_iou
                    self._save_checkpoint(epoch, val_iou)
                    log += "  ← best"

            print(log)

            if self.scheduler is not None:
                self.scheduler.step()

    def train_epoch(self) -> Tuple[float, float, float]:
        """Una época de entrenamiento. Devuelve (mean_loss, mean_iou, mean_dice)."""
        self.model.train()
        cfg = self.cfg.training

        total_loss = total_iou = total_dice = 0.0
        n_batches  = 0

        for batch in self.train_loader:
            support_imgs, support_masks, query_imgs, query_masks = self._to_device(batch)

            # Quitamos la dimensión K (k_shot=1 en baseline)
            support_imgs  = support_imgs.squeeze(1)   # B × 3 × H × W
            support_masks = support_masks.squeeze(1)  # B × 1 × H × W

            # logits: B × 1 × H × W
            logits = self.model(support_imgs, support_masks, query_imgs)
            loss   = self.criterion(logits, query_masks)

            self.optimizer.zero_grad()
            loss.backward()

            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            self.optimizer.step()
            self.global_step += 1

            with torch.no_grad():
                iou  = binary_iou(logits,  query_masks)
                dice = binary_dice(logits, query_masks)

            total_loss += loss.item()
            total_iou  += iou.item()
            total_dice += dice.item()
            n_batches  += 1

            if self.global_step % cfg.log_every_n_episodes == 0:
                print(
                    f"  step {self.global_step} "
                    f"| loss {loss.item():.4f} "
                    f"| iou {iou.item():.4f}"
                )

        return (
            total_loss / n_batches,
            total_iou  / n_batches,
            total_dice / n_batches,
        )

    def val_epoch(self) -> Tuple[float, float, float]:
        """Ejecuta una época de validación."""
        self.model.eval()

        total_loss = total_iou = total_dice = 0.0
        n_batches  = 0

        with torch.no_grad():
            for batch in self.val_loader:
                support_imgs, support_masks, query_imgs, query_masks = self._to_device(batch)

                support_imgs  = support_imgs.squeeze(1)
                support_masks = support_masks.squeeze(1)

                logits = self.model(support_imgs, support_masks, query_imgs)

                total_loss += self.criterion(logits, query_masks).item()
                total_iou  += binary_iou(logits,  query_masks).item()
                total_dice += binary_dice(logits, query_masks).item()
                n_batches  += 1

        return (
            total_loss / n_batches,
            total_iou  / n_batches,
            total_dice / n_batches,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_device(self, batch):
        """Move all tensors in a batch tuple to the configured device."""
        return [t.to(self.device) for t in batch]

    def _build_optimizer(self):
        """Crea el optimizador según la config."""
        name = self.cfg.training.optimizer
        params = dict(
            params=self.model.parameters(),
            lr=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
        )
 
        if name == "adamw":
            return torch.optim.AdamW(**params)
        elif name == "adam":
            return torch.optim.Adam(**params)
        else:
            raise ValueError(
                f"Unknown optimizer '{name}'. "
                f"Valid options: 'adam', 'adamw'."
            )

    def _build_scheduler(self):
        """Crea el scheduler de learning rate según la config. Devuelve None si está desactivado."""
        name = self.cfg.training.lr_scheduler

        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.training.epochs,
            )
        elif name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif name == "none":
            return None
        else:
            raise ValueError(
                f"Unknown lr_scheduler '{name}'. "
                f"Valid options: 'cosine', 'step', 'none'."
            )

    def _save_checkpoint(self, epoch: int, val_iou: float) -> None:
        """Guarda el checkpoint del mejor modelo en disco (sobreescribe el anterior)."""
        path = self.checkpoint_dir / "best_model.pt"

        torch.save(
            {
                "epoch":     epoch,
                "val_iou":   val_iou,
                "model":     self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config":    self.cfg,
            },
            path,
        )

        import json, dataclasses
        json_path = self.checkpoint_dir / "best_model.json"  # config legible al lado del .pt
        with open(json_path, "w") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=2)
