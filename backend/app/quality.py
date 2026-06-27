from typing import Any, Dict, List, Tuple

import numpy as np

from app.config import LOW_RMS_THRESHOLD, MIN_DURATION_SECONDS


def compute_quality_metrics(audio: np.ndarray, sample_rate: int) -> Tuple[Dict[str, Any], List[str]]:
    """
    Compute basic audio quality metrics.

    Returns:
        quality_metrics: dictionary with duration, sample rate, RMS energy, peak amplitude, clipping rate.
        warnings: list of warning messages about audio quality.
    """
    warnings: List[str] = []

    duration_seconds = float(len(audio) / sample_rate) if sample_rate > 0 else 0.0
    rms_energy = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0
    peak_amplitude = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0

    clipping_threshold = 0.99
    clipping_rate = float(np.mean(np.abs(audio) >= clipping_threshold)) if len(audio) > 0 else 0.0

    if duration_seconds < MIN_DURATION_SECONDS:
        warnings.append("Audio is too short for reliable feature extraction.")

    if rms_energy < LOW_RMS_THRESHOLD:
        warnings.append("Audio may be silent or too quiet.")

    if clipping_rate > 0.01:
        warnings.append("Audio may contain clipping or distortion.")

    quality_metrics = {
        "duration_seconds": round(duration_seconds, 3),
        "sample_rate": sample_rate,
        "rms_energy": round(rms_energy, 6),
        "peak_amplitude": round(peak_amplitude, 6),
        "clipping_rate": round(clipping_rate, 6),
    }

    return quality_metrics, warnings
