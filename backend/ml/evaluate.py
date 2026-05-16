"""
Phase 5 — Evaluation Engine
=============================
Generates a full evaluation report with:
  - Regression metrics (MAE, RMSE, Pearson r, R²)
  - Temporal jitter analysis
  - Prediction vs ground truth plots
  - Residual distribution plot
  - Per-participant breakdown
  - Calibration curve
  - Exports JSON + PNG report
"""

import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.model import create_model
from ml.dataset import create_dataloaders, FeatureScaler, CognitiveLoadDataset

# Dark theme
plt.rcParams.update({
    "figure.facecolor": "#0d0d1a",
    "axes.facecolor":   "#131326",
    "axes.edgecolor":   "#3a3a5c",
    "axes.labelcolor":  "#c0c0e0",
    "xtick.color":      "#8080a0",
    "ytick.color":      "#8080a0",
    "text.color":       "#e0e0ff",
    "grid.color":       "#252540",
    "grid.alpha":       0.6,
    "font.family":      "monospace",
})
C_PRED    = "#7c3aed"   # violet
C_TRUE    = "#06b6d4"   # cyan
C_ERROR   = "#f43f5e"   # rose
C_ZERO    = "#22c55e"   # green


# Metrics
def regression_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    p, t = preds.flatten(), targets.flatten()
    mae  = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2   = 1.0 - ss_res / (ss_tot + 1e-8)
    corr = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-8 else 0.0
    # Temporal jitter (mean abs frame-to-frame change)
    jitter = float(np.mean(np.abs(np.diff(p * 100)))) if len(p) > 1 else 0.0
    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "pearson_r": corr, "jitter_pts": jitter,
        "n_samples": int(len(p)),
    }


# Plots
def plot_prediction_vs_truth(preds, targets, save_path: str, title: str = ""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Evaluation Report — {title}", fontsize=13, fontweight="bold", y=1.02)

    p, t = preds.flatten(), targets.flatten()
    m    = regression_metrics(p, t)

    # Scatter
    ax = axes[0]
    ax.scatter(t, p, s=12, alpha=0.6, color=C_PRED, edgecolors="none")
    lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
    ax.plot([lo, hi], [lo, hi], color=C_TRUE, linewidth=1.5, linestyle="--", label="Perfect")
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")
    ax.set_title(f"Pred vs Truth\nPearson r={m['pearson_r']:.3f}  R²={m['r2']:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Residuals histogram
    ax = axes[1]
    residuals = p - t
    ax.hist(residuals, bins=30, color=C_PRED, alpha=0.8, edgecolor="none")
    ax.axvline(0, color=C_ZERO, linewidth=1.5, linestyle="--", label="Zero error")
    ax.axvline(residuals.mean(), color=C_ERROR, linewidth=1.5, linestyle="-",
               label=f"Mean={residuals.mean():.3f}")
    ax.set_xlabel("Residual (Pred − Truth)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals\nMAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Calibration (binned)
    ax = axes[2]
    bins  = np.linspace(t.min(), t.max(), 8)
    bin_ids = np.digitize(t, bins) - 1
    bin_ids = np.clip(bin_ids, 0, len(bins) - 2)
    means_t, means_p = [], []
    for b in range(len(bins) - 1):
        mask = bin_ids == b
        if mask.sum() > 0:
            means_t.append(t[mask].mean())
            means_p.append(p[mask].mean())
    means_t, means_p = np.array(means_t), np.array(means_p)
    ax.plot(means_t, means_p, "o-", color=C_PRED, markersize=6, linewidth=1.5,
            label="Model")
    ax.plot(means_t, means_t, "--", color=C_TRUE, linewidth=1.5, label="Ideal")
    ax.set_xlabel("True Load (binned mean)")
    ax.set_ylabel("Predicted Load (mean)")
    ax.set_title(f"Calibration Curve")
    ax.legend(fontsize=8)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Eval] Plot saved → {save_path}")


def plot_sequence_prediction(preds, targets, save_path: str, n_seq: int = 5):
    """Show predicted vs true load over time for N sample sequences."""
    fig, axes = plt.subplots(n_seq, 1, figsize=(14, 3 * n_seq), sharex=False)
    if n_seq == 1:
        axes = [axes]

    N = len(preds)
    idxs = np.linspace(0, N - 1, n_seq, dtype=int)

    fig.suptitle("Temporal Prediction Samples", fontsize=13, fontweight="bold")
    for i, (ax, idx) in enumerate(zip(axes, idxs)):
        t_val = float(targets[idx].flatten()[0])
        p_val = float(preds[idx].flatten()[0])
        ax.axhline(t_val * 100, color=C_TRUE, linewidth=2,
                   label=f"True: {t_val*100:.1f}", linestyle="--")
        ax.axhline(p_val * 100, color=C_PRED, linewidth=2,
                   label=f"Pred: {p_val*100:.1f}")
        ax.fill_between([0, 1], [t_val * 100] * 2, [p_val * 100] * 2,
                        alpha=0.2, color=C_ERROR)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score (0–100)")
        ax.set_title(f"Sequence {idx}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Eval] Sequence plot → {save_path}")


# Main Evaluation Runner
@torch.no_grad()
def run_evaluation(
    checkpoint_dir: str,
    processed_dir: str,
    dataset_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """
    Loads best checkpoint, runs test-set evaluation, generates plots.
    """
    ckpt_dir = Path(checkpoint_dir)
    out_dir  = Path(output_dir) if output_dir else ckpt_dir / "eval_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path = ckpt_dir / "best_model.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {ckpt_dir}")

    # Load checkpoint
    state  = torch.load(str(best_path), map_location="cpu", weights_only=False)
    cfg    = state["config"]
    mcfg   = cfg["model"]

    model_kwargs = {
        "input_dim":    mcfg["input_dim"],
        "hidden_dim":   mcfg["hidden_dim"],
        "num_layers":   mcfg["num_layers"],
        "dropout":      mcfg["dropout"],
        "bidirectional":mcfg["bidirectional"],
        "use_attention":mcfg["use_attention"],
    }
    model = create_model(arch="lstm", **model_kwargs)
    model.load_state_dict(state["model_state"])
    model.eval()

    print(f"[Eval] Loaded epoch {state['epoch']}  val_loss={state['val_metrics']['loss']:.4f}")

    # Rebuild data loaders
    _, _, test_loader, scaler = create_dataloaders(
        processed_dir  = processed_dir,
        dataset_dir    = dataset_dir,
        batch_size     = 64,
        label_smoothing= 0.0,      # no smoothing at eval time
        seed           = cfg["data"]["seed"],
    )

    # Collect predictions
    all_preds, all_targets = [], []
    for X, Y in test_loader:
        pred = model(X)
        all_preds.append(pred.numpy())
        all_targets.append(Y.numpy())

    if not all_preds:
        print("[Eval] Test set empty — cannot evaluate.")
        return {}

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    metrics = regression_metrics(preds, targets)
    print(f"\n  MAE:       {metrics['mae']:.4f}")
    print(f"  RMSE:      {metrics['rmse']:.4f}")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  R²:        {metrics['r2']:.4f}")
    print(f"  Jitter:    {metrics['jitter_pts']:.2f} pts/window")

    # Plots
    exp_name = cfg["experiment"]["name"]
    plot_prediction_vs_truth(preds, targets,
                              str(out_dir / "prediction_report.png"), title=exp_name)
    plot_sequence_prediction(preds, targets,
                              str(out_dir / "sequence_samples.png"))

    # Save JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[Eval] Report saved → {out_dir}/")
    return metrics


if __name__ == "__main__":
    BASE = Path(__file__).parent.parent
    exp  = "phase5_bilstm_attention_v1"
    run_evaluation(
        checkpoint_dir = str(BASE / "ml" / "checkpoints" / exp),
        processed_dir  = str(BASE / "processed_data"),
        dataset_dir    = str(BASE / "dataset"),
    )
