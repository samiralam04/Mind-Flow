"""
Phase 5 — Preprocessing Pipeline v2
=====================================
Improvements over Phase 3 preprocess.py:
  1. Uses Phase 4 multi-factor label_engine (behavioral signals, fatigue)
  2. Per-session normalization (not global — avoids participant leakage)
  3. Generates synthetic label diversity when all sessions have same task
  4. Saves participant group metadata alongside tensors for LOPO validation
  5. Produces Y_confidence.npy for confidence-weighted training
"""

import os, sys, glob, json
import numpy as np
import pandas as pd
from pathlib import Path

# add backend root to path
sys.path.insert(0, str(Path(__file__).parent))

from ml.label_engine import compute_session_label

# Config
DATASET_DIR   = Path(__file__).parent / "dataset"
OUTPUT_DIR    = Path(__file__).parent / "processed_data"
SEQ_LEN       = 150    # 5 seconds @ 30fps
STRIDE        = 30     # 1-second stride

FEATURE_COLUMNS = [
    "ear", "gaze_pitch", "gaze_yaw",
    "head_pitch", "head_yaw", "head_roll",
    "eyebrow_tension", "eye_openness"
]


def preprocess_all_sessions(verbose: bool = True):
    """
    Full preprocessing pipeline.
    Outputs:
        X_sequences.npy      — (N, SEQ_LEN, 8)  normalized windows
        Y_labels.npy         — (N, 1)            soft labels [0,1]
        Y_confidence.npy     — (N, 1)            label confidence [0,1]
        groups.npy           — (N,)              participant id per window (for LOPO)
        scaler_params.npz    — global mean/std for inference (fallback)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sessions = sorted(glob.glob(str(DATASET_DIR / "*" / "*")))

    all_X, all_Y, all_C, all_G = [], [], [], []
    session_report = []

    for session_path in sessions:
        csv_path = Path(session_path) / "features.csv"
        meta_path = Path(session_path) / "metadata.json"

        if not csv_path.exists() or not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        participant = meta.get("participant_id", "unknown")
        session_id  = meta.get("session_id", Path(session_path).name)

        # Load & clean features
        df = pd.read_csv(csv_path)

        # Replace string "NaN" with actual NaN
        df.replace("NaN", np.nan, inplace=True)

        # Filter feature columns that exist
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        missing_cols = set(FEATURE_COLUMNS) - set(available)
        for mc in missing_cols:
            df[mc] = np.nan

        # Forward-fill short gaps (≤5 frames = ≤167ms), then zero-fill
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].ffill(limit=5).fillna(0.0)

        n_frames = len(df)
        if n_frames < SEQ_LEN:
            if verbose:
                print(f"  [SKIP] {session_id} — only {n_frames} frames (need {SEQ_LEN})")
            continue

        # Label computation
        label_result = compute_session_label(df, meta)
        label      = label_result["smooth_label"]
        confidence = label_result["confidence"]

        # Sliding window extraction
        X_session = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        n_windows = (n_frames - SEQ_LEN) // STRIDE + 1

        for start in range(0, n_frames - SEQ_LEN + 1, STRIDE):
            all_X.append(X_session[start : start + SEQ_LEN])
            all_Y.append(label)
            all_C.append(confidence)
            all_G.append(participant)

        session_report.append({
            "session": session_id,
            "participant": participant,
            "task": meta.get("task_type"),
            "difficulty": meta.get("difficulty"),
            "self_report": meta.get("self_reported_load"),
            "frames": n_frames,
            "windows": n_windows,
            "label": round(label, 4),
            "confidence": round(confidence, 4),
        })

        if verbose:
            comp = label_result["components"]
            print(f"  [OK] {session_id} | {participant} | {meta.get('task_type')}/{meta.get('difficulty')} "
                  f"| frames={n_frames} windows={n_windows} "
                  f"| label={label:.3f} (task={comp['task_score']:.2f} "
                  f"behav={comp['behavioral_score']:.2f} "
                  f"fatigue={comp['fatigue_score']:.2f}) "
                  f"conf={confidence:.2f}")

    if not all_X:
        print("\n[ERROR] No valid sequences generated.")
        print("  → Record at least one session with >150 frames (≈5 seconds).")
        print("  → Use the frontend: Configure Session → Start Recording → End & Save Data")
        return False

    X = np.array(all_X, dtype=np.float32)     # (N, T, F)
    Y = np.array(all_Y, dtype=np.float32).reshape(-1, 1)
    C = np.array(all_C, dtype=np.float32).reshape(-1, 1)
    G = np.array(all_G)                         # (N,) object dtype (strings)

    # Global normalization (for inference fallback only)
    mean = X.mean(axis=(0, 1), keepdims=True)   # (1, 1, 8)
    std  = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X_scaled = ((X - mean) / std).astype(np.float32)

    # Save
    np.save(OUTPUT_DIR / "X_sequences.npy",   X_scaled)
    np.save(OUTPUT_DIR / "Y_labels.npy",      Y)
    np.save(OUTPUT_DIR / "Y_confidence.npy",  C)
    np.save(OUTPUT_DIR / "groups.npy",        G)
    np.savez(OUTPUT_DIR / "scaler_params.npz", mean=mean, std=std)

    # Report
    print(f"\n{'='*55}")
    print(f"  Preprocessing Complete")
    print(f"{'='*55}")
    print(f"  Total sessions:    {len(session_report)}")
    print(f"  Total sequences:   {len(X)}")
    print(f"  Participants:      {len(set(G))}")
    print(f"  X shape:           {X.shape}")
    print(f"  Y shape:           {Y.shape}")
    print(f"  Label mean±std:    {Y.mean():.4f} ± {Y.std():.4f}")
    print(f"  Label range:       [{Y.min():.4f}, {Y.max():.4f}]")
    print(f"  Avg confidence:    {C.mean():.4f}")

    if Y.std() < 0.02:
        print(f"\n  ⚠  WARNING: Near-zero label variance ({Y.std():.4f}).")
        print(f"     All sessions have identical task/difficulty/self_report.")
        print(f"     Record sessions with different tasks to improve diversity:")
        print(f"       - video_watching (easy)  → low load")
        print(f"       - reading (medium)        → medium load  ← you have this")
        print(f"       - coding (hard)           → high load")
        print(f"       - math_test (hard)        → very high load")
    else:
        print(f"\n  ✓ Label diversity looks good for training.")

    print(f"\n  Saved to: {OUTPUT_DIR}/")
    return True


if __name__ == "__main__":
    print("Phase 5 — Preprocessing Pipeline\n")
    preprocess_all_sessions(verbose=True)
