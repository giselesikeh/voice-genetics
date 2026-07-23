from __future__ import annotations

from typing import Any

import librosa
import numpy as np
from scipy.signal import find_peaks


EPS = 1e-10


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return None
        return round(x, 6)
    except Exception:
        return None


def _summary(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "mean": _safe_float(np.mean(values)),
        "median": _safe_float(np.median(values)),
        "std": _safe_float(np.std(values)),
        "min": _safe_float(np.min(values)),
        "max": _safe_float(np.max(values)),
    }


def estimate_formants_lpc(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    lpc_order: int = 12,
) -> dict[str, Any]:
    """
    Estimate F1/F2/F3 using LPC roots.

    This is an approximate acoustic estimate, not Praat-level clinical formant tracking.
    It works best on voiced speech and may be unstable on noisy or music-heavy audio.
    """
    y = np.asarray(y, dtype=float)

    if y.size < frame_length or np.max(np.abs(y)) < EPS:
        return {
            "method": "lpc_roots",
            "available": False,
            "reason": "audio_too_short_or_silent",
        }

    # Pre-emphasis improves LPC formant estimation.
    y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])

    frames = librosa.util.frame(y_pre, frame_length=frame_length, hop_length=hop_length).T

    f1_values = []
    f2_values = []
    f3_values = []

    for frame in frames:
        frame = frame * np.hamming(len(frame))

        if np.sqrt(np.mean(frame**2)) < 1e-4:
            continue

        try:
            # librosa.lpc signature is librosa.lpc(y, order=...)
            a = librosa.lpc(frame, order=lpc_order)
        except Exception:
            continue

        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0.01]

        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * (sr / (2 * np.pi))

        # Bandwidth filter removes unstable/wide resonances.
        bandwidths = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(roots) + EPS)

        formants = []
        for freq, bw in zip(freqs, bandwidths):
            if 90 <= freq <= 5500 and bw < 700:
                formants.append(freq)

        formants = sorted(formants)

        if len(formants) >= 1:
            f1_values.append(formants[0])
        if len(formants) >= 2:
            f2_values.append(formants[1])
        if len(formants) >= 3:
            f3_values.append(formants[2])

    return {
        "method": "lpc_roots",
        "available": bool(f1_values or f2_values or f3_values),
        "note": "Approximate LPC-based formant estimates. More reliable formants would require Praat/parselmouth.",
        "F1_hz": _summary(np.array(f1_values)),
        "F2_hz": _summary(np.array(f2_values)),
        "F3_hz": _summary(np.array(f3_values)),
        "frames_used": int(max(len(f1_values), len(f2_values), len(f3_values))),
    }


def estimate_jitter_shimmer(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> dict[str, Any]:
    """
    Approximate jitter and shimmer from frame-level f0 and RMS.

    Jitter is estimated from frame-to-frame pitch-period variation.
    Shimmer is estimated from frame-to-frame voiced RMS variation.
    """
    y = np.asarray(y, dtype=float)

    if y.size < frame_length or np.max(np.abs(y)) < EPS:
        return {
            "available": False,
            "reason": "audio_too_short_or_silent",
        }

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    except Exception as e:
        return {
            "available": False,
            "reason": f"pyin_failed: {e}",
        }

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    voiced = np.isfinite(f0) & voiced_flag
    f0_voiced = f0[voiced]
    rms_voiced = rms[voiced]

    if f0_voiced.size < 3:
        return {
            "available": False,
            "reason": "not_enough_voiced_frames",
            "voiced_frame_count": int(f0_voiced.size),
        }

    periods = 1.0 / np.maximum(f0_voiced, EPS)

    jitter_local = np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + EPS)

    shimmer_local = np.mean(np.abs(np.diff(rms_voiced))) / (np.mean(rms_voiced) + EPS)

    return {
        "available": True,
        "method": "frame_level_f0_rms_proxy",
        "note": "Approximate jitter/shimmer from frame-level f0 and RMS, not cycle-exact Praat measurements.",
        "voiced_frame_count": int(f0_voiced.size),
        "jitter_local": _safe_float(jitter_local),
        "jitter_local_percent": _safe_float(jitter_local * 100.0),
        "shimmer_local": _safe_float(shimmer_local),
        "shimmer_local_percent": _safe_float(shimmer_local * 100.0),
    }


def estimate_hnr(y: np.ndarray, sr: int) -> dict[str, Any]:
    """
    Estimate harmonic-to-noise ratio using harmonic source separation.

    HNR = 10 log10(harmonic energy / residual noise energy)
    """
    y = np.asarray(y, dtype=float)

    if y.size < sr * 0.25 or np.max(np.abs(y)) < EPS:
        return {
            "available": False,
            "reason": "audio_too_short_or_silent",
        }

    try:
        harmonic = librosa.effects.harmonic(y)
        noise = y - harmonic

        harmonic_energy = float(np.sum(harmonic**2))
        noise_energy = float(np.sum(noise**2))

        hnr_db = 10.0 * np.log10((harmonic_energy + EPS) / (noise_energy + EPS))

        return {
            "available": True,
            "method": "harmonic_residual_energy_ratio",
            "hnr_db": _safe_float(hnr_db),
            "harmonic_energy": _safe_float(harmonic_energy),
            "noise_residual_energy": _safe_float(noise_energy),
            "note": "Approximate HNR proxy based on harmonic separation, not Praat HNR.",
        }
    except Exception as e:
        return {
            "available": False,
            "reason": f"hnr_failed: {e}",
        }


def estimate_speaking_rate_proxy(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> dict[str, Any]:
    """
    Estimate speaking rate without transcript.

    Since we do not have words or syllable labels, this estimates syllable-like nuclei
    from RMS envelope peaks. It should be reported as a proxy, not true words per minute.
    """
    y = np.asarray(y, dtype=float)

    if y.size < sr * 0.5 or np.max(np.abs(y)) < EPS:
        return {
            "available": False,
            "reason": "audio_too_short_or_silent",
        }

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    if rms.size < 3:
        return {
            "available": False,
            "reason": "not_enough_rms_frames",
        }

    rms_norm = rms / (np.max(rms) + EPS)

    # Peaks in loudness envelope roughly approximate syllable nuclei.
    peaks, _ = find_peaks(
        rms_norm,
        height=0.15,
        distance=max(1, int(0.18 * sr / hop_length)),
        prominence=0.05,
    )

    duration_seconds = len(y) / sr
    rate_per_second = len(peaks) / max(duration_seconds, EPS)
    rate_per_minute = rate_per_second * 60.0

    return {
        "available": True,
        "method": "rms_envelope_peak_proxy",
        "note": "Speaking rate proxy from syllable-like RMS peaks. True speaking rate requires transcript or syllable labels.",
        "estimated_syllable_nuclei_count": int(len(peaks)),
        "estimated_syllable_rate_per_second": _safe_float(rate_per_second),
        "estimated_syllable_rate_per_minute": _safe_float(rate_per_minute),
    }


def estimate_voice_onset_time_proxy(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> dict[str, Any]:
    """
    Estimate a weak proxy for voice onset time.

    True VOT requires phoneme/stop-consonant alignment. Without transcript/phoneme labels,
    we can only estimate the time from detected energy onset to first reliable voiced f0.
    """
    y = np.asarray(y, dtype=float)

    if y.size < sr * 0.5 or np.max(np.abs(y)) < EPS:
        return {
            "available": False,
            "reason": "audio_too_short_or_silent",
        }

    try:
        intervals = librosa.effects.split(y, top_db=30)

        if len(intervals) == 0:
            return {
                "available": False,
                "reason": "no_detected_speech_intervals",
            }

        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            hop_length=hop_length,
        )

        voiced_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
        voiced_mask = np.isfinite(f0) & voiced_flag

        onset_delays_ms = []

        for start_sample, end_sample in intervals[:50]:
            start_time = start_sample / sr
            end_time = end_sample / sr

            inside = (voiced_times >= start_time) & (voiced_times <= end_time) & voiced_mask

            if np.any(inside):
                first_voiced_time = voiced_times[np.where(inside)[0][0]]
                delay_ms = (first_voiced_time - start_time) * 1000.0

                if 0 <= delay_ms <= 500:
                    onset_delays_ms.append(delay_ms)

        return {
            "available": len(onset_delays_ms) > 0,
            "method": "energy_onset_to_first_voiced_frame_proxy",
            "note": "This is not true phonetic VOT. True VOT needs stop-consonant boundaries or forced alignment.",
            "estimated_vot_ms": _summary(np.array(onset_delays_ms)),
            "events_used": int(len(onset_delays_ms)),
        }

    except Exception as e:
        return {
            "available": False,
            "reason": f"vot_proxy_failed: {e}",
        }


def extract_advanced_acoustic_features(y: np.ndarray, sr: int) -> dict[str, Any]:
    """
    Final-stage acoustic feature bundle.
    """
    return {
        "formants": estimate_formants_lpc(y, sr),
        "jitter_shimmer": estimate_jitter_shimmer(y, sr),
        "harmonic_to_noise_ratio": estimate_hnr(y, sr),
        "speaking_rate": estimate_speaking_rate_proxy(y, sr),
        "voice_onset_time": estimate_voice_onset_time_proxy(y, sr),
    }