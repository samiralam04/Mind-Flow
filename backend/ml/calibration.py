"""
Phase 6 — Personalized Baseline Calibration System
====================================================
Science:
  Inter-person variability in EAR, head pose, and gaze is enormous —
  a "neutral" EAR of 0.28 for one person is "tired" for another with
  naturally smaller eyes. Without personalization, a generic threshold
  treats everyone the same, causing systematic false positives/negatives.

Architecture:
  1. CalibrationCapture — collects ~300 frames of relaxed-state features
  2. BaselineProfile — stores per-user mean/std + confidence score
  3. ProfileStore — JSON file persistence in user_profiles/
  4. Normalizer — z-scores live features against personal baseline

Calibration phases:
  relaxed (10s) → "sit normally, look at the screen"
  This gives us the user's neutral behavioral fingerprint.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from collections import deque

FEATURE_NAMES = [
    "ear", "gaze_pitch", "gaze_yaw",
    "head_pitch", "head_yaw", "head_roll",
    "eyebrow_tension", "eye_openness",
]

PROFILES_DIR = Path(__file__).parent.parent / "user_profiles"
PROFILES_DIR.mkdir(exist_ok=True)

MIN_CALIBRATION_FRAMES = 150   # 5 seconds @ 30fps (minimum)
TARGET_FRAMES          = 300   # 10 seconds @ 30fps (ideal)


@dataclass
class BaselineProfile:
    """Per-user behavioral fingerprint captured during relaxed calibration."""
    user_id:          str
    created_at:       float
    updated_at:       float
    n_frames:         int
    mean:             Dict[str, float]   # feature → personal mean
    std:              Dict[str, float]   # feature → personal std
    confidence:       float              # 0–1, based on n_frames + std stability
    calibration_count: int              # how many times recalibrated
    session_count:    int               # how many sessions since calibration

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BaselineProfile":
        return cls(**d)


class CalibrationCapture:
    """
    Collects features during a calibration phase.
    Thread-safe. Call .add_frame() per webcam frame, .finish() when done.
    """

    def __init__(self, user_id: str):
        self.user_id   = user_id
        self.started   = time.time()
        self._buffers: Dict[str, List[float]] = {f: [] for f in FEATURE_NAMES}
        self._n        = 0
        self.done      = False

    @property
    def progress(self) -> float:
        """0.0 → 1.0 calibration progress."""
        return min(1.0, self._n / TARGET_FRAMES)

    @property
    def seconds_remaining(self) -> float:
        elapsed = time.time() - self.started
        target_s = TARGET_FRAMES / 30.0
        return max(0.0, target_s - elapsed)

    def add_frame(self, metrics: dict) -> bool:
        """
        Ingest one frame of metrics. Returns True when calibration is complete.
        Only uses frames where face_confidence >= 0.8 (clean captures only).
        """
        if self.done:
            return True

        face_conf = float(metrics.get("face_confidence", 0.0))
        if face_conf < 0.8:
            return False  # don't count low-quality frames

        for feat in FEATURE_NAMES:
            val = metrics.get(feat)
            if val is not None and val != "NaN" and not (isinstance(val, float) and np.isnan(val)):
                self._buffers[feat].append(float(val))

        self._n += 1
        if self._n >= TARGET_FRAMES:
            self.done = True
        return self.done

    def finish(self, min_frames: int = MIN_CALIBRATION_FRAMES) -> Optional[BaselineProfile]:
        """
        Build BaselineProfile from captured frames.
        Returns None if too few clean frames were captured.
        """
        usable = min(len(v) for v in self._buffers.values() if v)
        if usable < min_frames:
            return None

        mean, std = {}, {}
        for feat in FEATURE_NAMES:
            vals = np.array(self._buffers[feat])
            if len(vals) == 0:
                mean[feat] = 0.0
                std[feat]  = 1.0
                continue
            # Robust stats: trim 5% outliers each side
            lo, hi = np.percentile(vals, 5), np.percentile(vals, 95)
            trimmed = vals[(vals >= lo) & (vals <= hi)]
            mean[feat] = float(trimmed.mean()) if len(trimmed) > 0 else float(vals.mean())
            std[feat]  = float(trimmed.std())  if len(trimmed) > 0 else float(vals.std())
            std[feat]  = max(std[feat], 1e-4)  # avoid division by zero

        # Confidence = based on frame count + std stability
        frame_conf = min(1.0, usable / TARGET_FRAMES)
        # Low std variance in EAR = stable capture = higher confidence
        ear_cv = std.get("ear", 0.1) / (abs(mean.get("ear", 0.3)) + 1e-4)
        stability_conf = float(np.clip(1.0 - ear_cv * 2, 0.3, 1.0))
        confidence = float(0.6 * frame_conf + 0.4 * stability_conf)

        now = time.time()
        existing = ProfileStore.load(self.user_id)
        calib_count = (existing.calibration_count + 1) if existing else 1
        sess_count  = existing.session_count if existing else 0

        return BaselineProfile(
            user_id          = self.user_id,
            created_at       = existing.created_at if existing else now,
            updated_at       = now,
            n_frames         = usable,
            mean             = mean,
            std              = std,
            confidence       = confidence,
            calibration_count= calib_count,
            session_count    = sess_count,
        )


class ProfileStore:
    """JSON-backed per-user profile storage."""

    @staticmethod
    def path(user_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in user_id)
        return PROFILES_DIR / f"{safe}.json"

    @classmethod
    def save(cls, profile: BaselineProfile):
        with open(cls.path(profile.user_id), "w") as f:
            json.dump(profile.to_dict(), f, indent=2)

    @classmethod
    def load(cls, user_id: str) -> Optional[BaselineProfile]:
        p = cls.path(user_id)
        if not p.exists():
            return None
        try:
            with open(p) as f:
                return BaselineProfile.from_dict(json.load(f))
        except Exception:
            return None

    @classmethod
    def list_all(cls) -> List[str]:
        return [p.stem for p in PROFILES_DIR.glob("*.json")]

    @classmethod
    def delete(cls, user_id: str):
        p = cls.path(user_id)
        if p.exists():
            p.unlink()

    @classmethod
    def increment_session_count(cls, user_id: str):
        profile = cls.load(user_id)
        if profile:
            profile.session_count += 1
            profile.updated_at = time.time()
            cls.save(profile)


class PersonalNormalizer:
    """
    Normalizes raw feature vectors using per-user baseline.
    Falls back to global z-score if no profile exists.

    Science note: We normalize to (x - user_mean) / user_std
    so the model sees deviations from the user's own baseline,
    not absolute values. This is analogous to within-subject
    normalization in psychophysiology research.
    """

    def __init__(self, profile: Optional[BaselineProfile], global_scaler=None):
        self.profile       = profile
        self.global_scaler = global_scaler   # FeatureScaler from Phase 5

    def normalize(self, feature_vec: np.ndarray) -> np.ndarray:
        """
        Normalize feature vector (shape: [F] or [T, F]).
        If profile exists: personal z-score.
        Else: global z-score (from training data).
        """
        if self.profile and self.profile.confidence > 0.4:
            mean = np.array([self.profile.mean.get(f, 0.0) for f in FEATURE_NAMES])
            std  = np.array([self.profile.std.get(f, 1.0)  for f in FEATURE_NAMES])
            return ((feature_vec - mean) / std).astype(np.float32)
        elif self.global_scaler is not None and self.global_scaler.mean is not None:
            m = self.global_scaler.mean[0]   # shape (1, 8) → (8,)
            s = self.global_scaler.std[0]
            return ((feature_vec - m) / s).astype(np.float32)
        else:
            # Cold start: return as-is (model handles unnormalized reasonably)
            return feature_vec.astype(np.float32)


# Global calibration session registry (in-memory)
_active_calibrations: Dict[str, CalibrationCapture] = {}


def start_calibration(user_id: str) -> CalibrationCapture:
    cap = CalibrationCapture(user_id)
    _active_calibrations[user_id] = cap
    return cap


def get_calibration(user_id: str) -> Optional[CalibrationCapture]:
    return _active_calibrations.get(user_id)


def finish_calibration(user_id: str) -> Optional[BaselineProfile]:
    cap = _active_calibrations.pop(user_id, None)
    if cap is None:
        return None
    profile = cap.finish()
    if profile:
        ProfileStore.save(profile)
        print(f"[Calibration] {user_id}: profile saved "
              f"(n={profile.n_frames}, conf={profile.confidence:.2f})")
    return profile
