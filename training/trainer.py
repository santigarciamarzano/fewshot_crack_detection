"""
training/trainer.py

Training and validation loop for the few-shot segmentation model.

The Trainer coordinates the model, dataloader, loss, optimizer, and
scheduler into a complete training pipeline. It has no knowledge of
the model internals — it only calls forward() and receives logits.

Responsibilities:
    - train_epoch(): one full pass over the training dataloader
    - val_epoch():   one full pass over the validation dataloader
    - fit():         coordinates epochs, scheduling, checkpointing, logging
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
    """Episodic trainer for few-shot segmentation.

    Handles the full training lifecycle: optimizer setup, learning rate
    scheduling, gradient clipping, checkpointing, and logging.

    Args:
        model:        The FewShotModel instance to train.
        cfg:          Root FewShotConfig — all training settings read from here.
        train_loader: DataLoader yielding training episodes.
        val_loader:   DataLoader yielding validation episodes. Optional.

    Example:
        cfg   = get_baseline_config()
        model = FewShotModel(cfg)

        train_loader = DataLoader(EpisodicDataset(cfg.dataset, "train"), batch_size=4)
        val_loader   = DataLoader(EpisodicDataset(cfg.dataset, "val"),   batch_size=4)

        trainer = Trainer(model, cfg, train_loader, val_loader)
        trainer.fit()
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
        """Run the full training loop for cfg.training.epochs epochs.

        Trains for one epoch, validates if val_loader is provided,
        saves checkpoint if validation IoU improves.
        """
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
        """Run one training epoch over all episodes in train_loader.

        Returns:
            Tuple of (mean_loss, mean_iou, mean_dice) over the epoch.
        """
        self.model.train()
        cfg = self.cfg.training

        total_loss = total_iou = total_dice = 0.0
        n_batches  = 0

        for batch in self.train_loader:
            support_imgs, support_masks, query_imgs, query_masks = self._to_device(batch)
            # support_imgs:  B × K × 3 × H × W
            # support_masks: B × K × 1 × H × W
            # query_imgs:    B × 3 × H × W
            # query_masks:   B × 1 × H × W

            # Squeeze K dim — baseline k_shot=1.
            # For k_shot>1 this needs to be replaced with prototype averaging.
            # support_imgs:  B × 3 × H × W
            # support_masks: B × 1 × H × W
            support_imgs  = support_imgs.squeeze(1)
            support_masks = support_masks.squeeze(1)

            # logits: B × 1 × H × W
            logits = self.model(support_imgs, support_masks, query_imgs)
            loss   = self.criterion(logits, query_masks)

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping prevents exploding gradients early in training.
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
        """Run one validation epoch over all episodes in val_loader.

        No gradients computed. Model set to eval mode so BatchNorm
        and Dropout behave correctly for inference.

        Returns:
            Tuple of (mean_loss, mean_iou, mean_dice) over the epoch.
        """
        self.model.eval()

        total_loss = total_iou = total_dice = 0.0
        n_batches  = 0

        with torch.no_grad():
            for batch in self.val_loader:
                support_imgs, support_masks, query_imgs, query_masks = self._to_device(batch)

                # Squeeze K dim — baseline k_shot=1
                # support_imgs:  B × 3 × H × W
                # support_masks: B × 1 × H × W
                support_imgs  = support_imgs.squeeze(1)
                support_masks = support_masks.squeeze(1)

                # logits: B × 1 × H × W
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
        """Build the optimizer from config.
 
        AdamW is recommended over Adam — it applies weight decay correctly,
        decoupled from the adaptive gradient scaling.
 
        Returns:
            A torch.optim optimizer instance.
        """
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
        """Build the learning rate scheduler from config.

        Returns:
            A torch.optim.lr_scheduler instance, or None if disabled.
        """
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
        """Save model and optimizer state to disk.

        Saves only the best model — overwrites previous best.
        To save every epoch instead, change the filename to include epoch.

        Args:
            epoch:   Current epoch number.
            val_iou: Validation IoU that triggered this save.
        """
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

        # Guardar config legible como JSON al lado del checkpoint
        import json
        import dataclasses
        json_path = self.checkpoint_dir / "best_model.json"
        with open(json_path, "w") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=2)
