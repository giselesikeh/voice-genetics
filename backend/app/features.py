
from typing import Any, Dict, Optional

import librosa
import numpy as np


def _safe_round(value: Optional[float], digits: int = 5):
    if value is None:
        return None
    return round(float(value), digits)


def extract_basic_features(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Extract basic acoustic features for the first working version.
    """
    if len(audio) == 0:
        return {
            "error": "No audio samples available for feature extraction."
        }

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)

    pitch_mean = None
    pitch_min = None
    pitch_max = None

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )

        valid_f0 = f0[~np.isnan(f0)]

        if len(valid_f0) > 0:
            pitch_mean = float(np.mean(valid_f0))
            pitch_min = float(np.min(valid_f0))
            pitch_max = float(np.max(valid_f0))

    except Exception:
        pass

    features = {
        "duration_seconds": round(float(len(audio) / sample_rate), 3),
        "mfcc": {
            "mean": np.mean(mfcc, axis=1).round(5).tolist(),
            "std": np.std(mfcc, axis=1).round(5).tolist(),
        },
        "spectral_centroid": {
            "mean": round(float(np.mean(spectral_centroid)), 5),
            "std": round(float(np.std(spectral_centroid)), 5),
        },
        "pitch": {
            "mean_hz": _safe_round(pitch_mean, 3),
            "min_hz": _safe_round(pitch_min, 3),
            "max_hz": _safe_round(pitch_max, 3),
        },
    }

    return features
