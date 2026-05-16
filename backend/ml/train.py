"""
Phase 5 — Production Trainer Class
====================================
A clean, self-contained Trainer that wraps the entire training loop.

Design goals:
  - Single class owns all training state (model, optimizer, scheduler, history)
  - Config-driven via dict (from config_loader.py)
  - Emits structured logs (TensorBoard + CSV + console)
  - Checkpoint manager: saves best + periodic + latest
  - GradScaler ready (mixed precision, no-op on CPU)
  - Debug-friendly: catches NaN loss and aborts gracefully
"""

import os
import csv
import json
import math
import time
import contextlib
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
try:
    from torch.amp import GradScaler, autocast  # PyTorch >= 2.0
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.model import CognitiveLSTM, LightweightGRU, create_model
from ml.dataset import create_dataloaders, FeatureScaler
from ml.config_loader import get_config


# Loss Functions
class HuberLoss(nn.Module):
    """Outlier-robust regression loss (δ controls linear/quadratic boundary)."""
    def __init__(self, delta: float = 0.1):
        super().__init__()
        self._loss = nn.HuberLoss(delta=delta, reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(pred.squeeze(-1), target.squeeze(-1))


# Metrics
def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    preds   = preds.flatten()
    targets = targets.flatten()
    mae     = float(np.mean(np.abs(preds - targets)))
    rmse    = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res  = float(np.sum((targets - preds) ** 2))
    ss_tot  = float(np.sum((targets - targets.mean()) ** 2))
    r2      = 1.0 - ss_res / (ss_tot + 1e-8)
    if len(preds) > 1 and preds.std() > 1e-8:
        corr = float(np.corrcoef(preds, targets)[0, 1])
    else:
        corr = 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson_r": corr}


# Early Stopping
class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = float("inf")
        self.counter    = 0
        self.best_epoch = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best       = val_loss
            self.counter    = 0
            self.best_epoch = epoch
            return False        # don't stop
        self.counter += 1
        return self.counter >= self.patience


# CSV Logger
class CSVLogger:
    def __init__(self, path: str):
        self.path   = path
        self._init  = False

    def log(self, row: dict):
        write_header = not self._init and not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._init = True


# Trainer
class Trainer:
    """
    Production-grade training engine.

    Usage:
        cfg = get_config()
        trainer = Trainer(cfg)
        trainer.train()
        trainer.evaluate_test()
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._setup_reproducibility()
        self._setup_device()
        self._setup_dirs()
        self._build_data()
        self._build_model()
        self._build_optimizer()
        self._build_scheduler()
        self._build_criterion()
        self._build_loggers()
        self.early_stop = EarlyStopping(
            patience=cfg["training"]["patience"],
            min_delta=cfg["training"]["min_delta"],
        )
        self.best_val_loss = float("inf")
        self.history: Dict[str, list] = {"train": [], "val": []}

    # Setup helpers
    def _setup_reproducibility(self):
        import random
        seed = self.cfg["training"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    def _setup_device(self):
        force_cpu = self.cfg["training"].get("force_cpu", False)
        if not force_cpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.scaler = GradScaler(
                "cuda",
                enabled=self.cfg["training"].get("mixed_precision", False)
            )
        elif not force_cpu and torch.backends.mps.is_available():
            # MPS has known instability with BiLSTM on small datasets.
            # Set force_cpu: true in config if you hit NaN losses.
            self.device = torch.device("mps")
            self.scaler = GradScaler("cpu", enabled=False)
        else:
            self.device = torch.device("cpu")
            self.scaler = GradScaler("cpu", enabled=False)
        print(f"[Trainer] Device: {self.device}")

    def _setup_dirs(self):
        base = Path(self.cfg["logging"]["checkpoint_dir"])
        exp  = self.cfg["experiment"]["name"]
        self.ckpt_dir = base / exp
        self.log_dir  = Path(self.cfg["logging"]["log_dir"]) / exp
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _build_data(self):
        dcfg = self.cfg["data"]
        tcfg = self.cfg["training"]
        self.train_loader, self.val_loader, self.test_loader, self.scaler_obj = \
            create_dataloaders(
                processed_dir  = dcfg["processed_dir"],
                dataset_dir    = dcfg.get("dataset_dir"),
                train_ratio    = dcfg["train_ratio"],
                val_ratio      = dcfg["val_ratio"],
                batch_size     = tcfg["batch_size"],
                label_smoothing= dcfg["label_smoothing"],
                use_weighted_sampler = dcfg["use_weighted_sampler"],
                seed           = dcfg["seed"],
            )
        print(f"[Trainer] Batches — train:{len(self.train_loader)} "
              f"val:{len(self.val_loader)} test:{len(self.test_loader)}")

    def _build_model(self):
        mcfg = self.cfg["model"]
        self.model = create_model(
            arch         = mcfg["arch"].replace("bilstm_attention", "lstm"),
            input_dim    = mcfg["input_dim"],
            hidden_dim   = mcfg["hidden_dim"],
            num_layers   = mcfg["num_layers"],
            dropout      = mcfg["dropout"],
            bidirectional= mcfg["bidirectional"],
            use_attention= mcfg["use_attention"],
        ).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Trainer] Model: {type(self.model).__name__} | {n_params:,} params")

    def _build_optimizer(self):
        tcfg = self.cfg["training"]
        self.optimizer = AdamW(
            self.model.parameters(),
            lr           = tcfg["lr"],
            weight_decay = tcfg["weight_decay"],
            betas        = tuple(tcfg["betas"]),
        )

    def _build_scheduler(self):
        tcfg       = self.cfg["training"]
        n_batches  = max(len(self.train_loader), 1)
        total      = tcfg["epochs"] * n_batches
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr           = tcfg["lr"],
            total_steps      = max(total, 1),
            pct_start        = tcfg.get("warmup_ratio", 0.1),
            anneal_strategy  = "cos",
            div_factor       = 25.0,
            final_div_factor = 1e4,
        )

    def _build_criterion(self):
        self.criterion = HuberLoss(delta=self.cfg["training"]["huber_delta"])

    def _build_loggers(self):
        self.csv_logger = CSVLogger(str(self.log_dir / "metrics.csv"))
        self.tb_writer  = None
        if TB_AVAILABLE and self.cfg["logging"].get("tensorboard", True):
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))
            print(f"[Trainer] TensorBoard → {self.log_dir}")

    # Core loops
    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []
        tcfg = self.cfg["training"]

        for X, Y in self.train_loader:
            X = X.to(self.device, non_blocking=True)
            Y = Y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            pred = self.model(X)
            loss = self.criterion(pred, Y)

            # NaN guard
            if not torch.isfinite(loss):
                print(f"[WARNING] NaN/Inf loss detected — skipping batch")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), tcfg["grad_clip"])
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            all_preds.append(pred.detach().cpu().float().numpy())
            all_targets.append(Y.detach().cpu().float().numpy())

        metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets))
        metrics["loss"] = total_loss / max(len(self.train_loader), 1)
        metrics["lr"]   = self.scheduler.get_last_lr()[0]
        return metrics

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for X, Y in loader:
            X = X.to(self.device, non_blocking=True)
            Y = Y.to(self.device, non_blocking=True)
            pred = self.model(X)
            loss = self.criterion(pred, Y)
            total_loss += loss.item()
            all_preds.append(pred.cpu().float().numpy())
            all_targets.append(Y.cpu().float().numpy())

        metrics = compute_metrics(np.concatenate(all_preds), np.concatenate(all_targets))
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    # Checkpoint
    def _save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool = False):
        state = {
            "epoch":        epoch,
            "model_state":  self.model.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "config":       self.cfg,
            "val_metrics":  val_metrics,
            "feature_columns": self.cfg["data"]["features"],
        }
        # Always overwrite "latest"
        torch.save(state, self.ckpt_dir / "latest.pt")

        # Periodic
        save_every = self.cfg["logging"].get("save_every_n_epochs", 10)
        if epoch % save_every == 0:
            torch.save(state, self.ckpt_dir / f"epoch_{epoch:04d}.pt")

        if is_best:
            torch.save(state, self.ckpt_dir / "best_model.pt")
            self.scaler_obj.save(str(self.ckpt_dir / "scaler.npz"))
            print(f"  ★ New best  val_loss={val_metrics['loss']:.5f}  "
                  f"MAE={val_metrics['mae']:.4f}  r={val_metrics['pearson_r']:.3f}")

    # Main train()
    def train(self):
        epochs = self.cfg["training"]["epochs"]
        exp    = self.cfg["experiment"]["name"]
        print(f"\n{'═'*60}")
        print(f"  Training: {exp}")
        print(f"  Epochs: {epochs}  BS: {self.cfg['training']['batch_size']}  "
              f"LR: {self.cfg['training']['lr']}")
        print(f"{'═'*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_m = self._train_epoch()
            val_m   = self._eval_epoch(self.val_loader)

            elapsed = time.time() - t0
            is_best = val_m["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_m["loss"]

            self.history["train"].append(train_m)
            self.history["val"].append(val_m)

            # Console
            best_marker = "★" if is_best else " "
            print(
                f"{best_marker} Ep {epoch:03d}/{epochs} [{elapsed:.1f}s]  "
                f"TrLoss={train_m['loss']:.4f}  TrMAE={train_m['mae']:.4f}  "
                f"VaLoss={val_m['loss']:.4f}  VaMAE={val_m['mae']:.4f}  "
                f"VaPearson={val_m['pearson_r']:.3f}  "
                f"LR={train_m['lr']:.2e}"
            )

            # Checkpoint
            self._save_checkpoint(epoch, val_m, is_best=is_best)

            # TensorBoard
            if self.tb_writer:
                for k, v in train_m.items():
                    self.tb_writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in val_m.items():
                    self.tb_writer.add_scalar(f"val/{k}", v, epoch)

            # CSV
            row = {"epoch": epoch, "elapsed_s": round(elapsed, 2)}
            row.update({f"train_{k}": round(v, 6) for k, v in train_m.items()})
            row.update({f"val_{k}":   round(v, 6) for k, v in val_m.items()})
            self.csv_logger.log(row)

            # Early stopping
            if self.early_stop.step(val_m["loss"], epoch):
                print(f"\n[EarlyStopping] Triggered at epoch {epoch}. "
                      f"Best was epoch {self.early_stop.best_epoch}.")
                break

        if self.tb_writer:
            self.tb_writer.close()

        self._save_history()
        print(f"\n[Train] Done. Best val_loss={self.best_val_loss:.5f}")

    def evaluate_test(self) -> Dict[str, float]:
        """Load best checkpoint and evaluate on held-out test set."""
        best_path = self.ckpt_dir / "best_model.pt"
        if best_path.exists():
            state = torch.load(str(best_path), map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model_state"])
            print(f"[Test] Loaded best checkpoint (epoch {state['epoch']})")

        test_m = self._eval_epoch(self.test_loader)
        print(f"\n{'═'*45}")
        print(f"  Test Results")
        print(f"{'═'*45}")
        print(f"  MAE:       {test_m['mae']:.4f}")
        print(f"  RMSE:      {test_m['rmse']:.4f}")
        print(f"  Pearson r: {test_m['pearson_r']:.4f}")
        print(f"  R²:        {test_m['r2']:.4f}")
        print(f"{'═'*45}")

        # Save test results
        with open(self.ckpt_dir / "test_results.json", "w") as f:
            json.dump(test_m, f, indent=2)

        return test_m

    def _save_history(self):
        history_path = self.ckpt_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump({
                "config":  self.cfg,
                "history": self.history,
                "best_val_loss": self.best_val_loss,
                "best_epoch": self.early_stop.best_epoch,
            }, f, indent=2)


# CLI Entry Point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, default=None, help="Experiment YAML path")
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--batch",   type=int,   default=None)
    parser.add_argument("--hidden",  type=int,   default=None)
    args = parser.parse_args()

    from ml.config_loader import load_config
    overrides = {}
    if args.lr:     overrides["training.lr"]           = args.lr
    if args.epochs: overrides["training.epochs"]       = args.epochs
    if args.batch:  overrides["training.batch_size"]   = args.batch
    if args.hidden: overrides["model.hidden_dim"]      = args.hidden

    cfg = load_config(args.config, overrides)

    trainer = Trainer(cfg)
    trainer.train()
    trainer.evaluate_test()
