from typing import Any, Dict, List, Tuple

import librosa
import numpy as np


def _safe_rms(audio: np.ndarray) -> float:
    if audio is None or len(audio) == 0:
        return 0.0

    audio = np.asarray(audio, dtype=np.float32)

    return float(np.sqrt(np.mean(audio ** 2)))


def _choose_adaptive_min_rms(
    global_rms: float,
    requested_min_rms: float,
) -> Tuple[float, str]:
    """
    Choose a safer VAD RMS threshold from the global audio loudness.

    The goal is:
    - avoid missing quiet speech in low-energy recordings;
    - keep the normal threshold for normal/loud recordings.
    """
    if requested_min_rms <= 0:
        return 0.0, "requested_min_rms_disabled"

    if global_rms < 0.025:
        return min(requested_min_rms, 0.005), (
            "quiet_audio_detected_global_rms_below_0.025_using_min_rms_0.005"
        )

    if global_rms < 0.040:
        return min(requested_min_rms, 0.010), (
            "moderately_quiet_audio_detected_global_rms_below_0.040_using_min_rms_0.010"
        )

    return requested_min_rms, "normal_audio_using_requested_min_rms"


def _format_region(
    audio: np.ndarray,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
) -> Dict[str, Any]:
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)

    start_seconds = start_sample / sample_rate
    end_seconds = end_sample / sample_rate
    duration_seconds = max(0.0, end_seconds - start_seconds)

    region_audio = audio[start_sample:end_sample]

    return {
        "start": round(start_seconds, 3),
        "end": round(end_seconds, 3),
        "duration": round(duration_seconds, 3),
        "rms": round(_safe_rms(region_audio), 6),
    }


def _merge_close_regions(
    audio: np.ndarray,
    sample_rate: int,
    regions: List[Dict[str, Any]],
    merge_gap_seconds: float,
) -> List[Dict[str, Any]]:
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda item: float(item["start"]))
    merged: List[Dict[str, Any]] = [sorted_regions[0].copy()]

    for region in sorted_regions[1:]:
        previous = merged[-1]
        gap = float(region["start"]) - float(previous["end"])

        if gap <= merge_gap_seconds:
            start_sample = int(float(previous["start"]) * sample_rate)
            end_sample = int(float(region["end"]) * sample_rate)
            merged[-1] = _format_region(audio, sample_rate, start_sample, end_sample)
        else:
            merged.append(region.copy())

    return merged


def _build_gap_segments(
    speech_regions: List[Dict[str, Any]],
    audio_duration_seconds: float,
) -> List[Dict[str, Any]]:
    removed_segments: List[Dict[str, Any]] = []

    if not speech_regions:
        return [
            {
                "start": 0.0,
                "end": round(audio_duration_seconds, 3),
                "duration": round(audio_duration_seconds, 3),
                "reason": "no_speech_detected",
            }
        ]

    sorted_regions = sorted(speech_regions, key=lambda item: float(item["start"]))

    first_start = float(sorted_regions[0]["start"])

    if first_start > 0:
        removed_segments.append(
            {
                "start": 0.0,
                "end": round(first_start, 3),
                "duration": round(first_start, 3),
                "reason": "leading_non_speech",
            }
        )

    for previous_region, next_region in zip(sorted_regions, sorted_regions[1:]):
        gap_start = float(previous_region["end"])
        gap_end = float(next_region["start"])

        if gap_end > gap_start:
            removed_segments.append(
                {
                    "start": round(gap_start, 3),
                    "end": round(gap_end, 3),
                    "duration": round(gap_end - gap_start, 3),
                    "reason": "vad_gap_or_pause",
                }
            )

    last_end = float(sorted_regions[-1]["end"])

    if last_end < audio_duration_seconds:
        removed_segments.append(
            {
                "start": round(last_end, 3),
                "end": round(audio_duration_seconds, 3),
                "duration": round(audio_duration_seconds - last_end, 3),
                "reason": "trailing_non_speech",
            }
        )

    return removed_segments


def detect_speech_regions(
    audio: np.ndarray,
    sample_rate: int,
    top_db: float = 30.0,
    min_rms: float = 0.015,
    min_region_duration_seconds: float = 0.25,
    merge_gap_seconds: float = 0.8,
    adaptive_min_rms: bool = False,
) -> Dict[str, Any]:
    """
    Central VAD function.

    It uses librosa.effects.split to propose speech-like regions, then filters
    them using duration and RMS energy. When adaptive_min_rms=True, quiet audio
    automatically receives a lower RMS threshold.
    """
    audio = np.asarray(audio, dtype=np.float32)

    audio_duration_seconds = len(audio) / sample_rate if sample_rate > 0 else 0.0
    global_rms = _safe_rms(audio)
    requested_min_rms = float(min_rms)

    if adaptive_min_rms:
        effective_min_rms, adaptive_reason = _choose_adaptive_min_rms(
            global_rms=global_rms,
            requested_min_rms=requested_min_rms,
        )
    else:
        effective_min_rms = requested_min_rms
        adaptive_reason = "adaptive_vad_disabled_using_requested_min_rms"

    if len(audio) == 0:
        return {
            "enabled": True,
            "method": "librosa_split_rms_guard_adaptive",
            "adaptive_vad_enabled": adaptive_min_rms,
            "adaptive_reason": "empty_audio",
            "speech_regions": [],
            "removed_non_speech_segments": [],
            "speech_region_count": 0,
            "removed_non_speech_count": 0,
            "speech_duration_seconds": 0.0,
            "non_speech_duration_seconds": 0.0,
            "speech_coverage_ratio": 0.0,
            "audio_duration_seconds": 0.0,
            "top_db": top_db,
            "requested_min_rms": requested_min_rms,
            "effective_min_rms": effective_min_rms,
            "min_rms": effective_min_rms,
            "min_region_duration_seconds": min_region_duration_seconds,
            "merge_gap_seconds": merge_gap_seconds,
            "global_rms": global_rms,
        }

    raw_intervals = librosa.effects.split(
        audio,
        top_db=top_db,
    )

    candidate_regions: List[Dict[str, Any]] = []
    filtered_segments: List[Dict[str, Any]] = []

    for start_sample, end_sample in raw_intervals:
        region = _format_region(audio, sample_rate, int(start_sample), int(end_sample))

        duration = float(region["duration"])
        region_rms = float(region["rms"])

        if duration < min_region_duration_seconds:
            filtered_segments.append(
                {
                    **region,
                    "reason": "region_too_short",
                }
            )
            continue

        if region_rms < effective_min_rms:
            filtered_segments.append(
                {
                    **region,
                    "reason": "region_rms_below_threshold",
                }
            )
            continue

        candidate_regions.append(region)

    speech_regions = _merge_close_regions(
        audio=audio,
        sample_rate=sample_rate,
        regions=candidate_regions,
        merge_gap_seconds=merge_gap_seconds,
    )

    gap_segments = _build_gap_segments(
        speech_regions=speech_regions,
        audio_duration_seconds=audio_duration_seconds,
    )

    removed_segments = gap_segments + filtered_segments

    speech_duration_seconds = round(
        sum(float(region["duration"]) for region in speech_regions),
        3,
    )

    non_speech_duration_seconds = round(
        max(0.0, audio_duration_seconds - speech_duration_seconds),
        3,
    )

    speech_coverage_ratio = (
        round(speech_duration_seconds / audio_duration_seconds, 4)
        if audio_duration_seconds > 0
        else 0.0
    )

    return {
        "enabled": True,
        "method": "librosa_split_rms_guard_adaptive",
        "adaptive_vad_enabled": adaptive_min_rms,
        "adaptive_reason": adaptive_reason,
        "speech_regions": speech_regions,
        "removed_non_speech_segments": removed_segments,
        "speech_region_count": len(speech_regions),
        "removed_non_speech_count": len(removed_segments),
        "speech_duration_seconds": speech_duration_seconds,
        "non_speech_duration_seconds": non_speech_duration_seconds,
        "speech_coverage_ratio": speech_coverage_ratio,
        "audio_duration_seconds": round(audio_duration_seconds, 3),
        "top_db": top_db,
        "requested_min_rms": requested_min_rms,
        "effective_min_rms": effective_min_rms,
        "min_rms": effective_min_rms,
        "min_region_duration_seconds": min_region_duration_seconds,
        "merge_gap_seconds": merge_gap_seconds,
        "global_rms": round(global_rms, 6),
    }


def concatenate_speech_regions(
    audio: np.ndarray,
    sample_rate: int,
    speech_regions: List[Dict[str, Any]],
) -> np.ndarray:
    if not speech_regions:
        return np.array([], dtype=np.float32)

    clips = []

    for region in speech_regions:
        start_sample = int(float(region["start"]) * sample_rate)
        end_sample = int(float(region["end"]) * sample_rate)

        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        if end_sample > start_sample:
            clips.append(audio[start_sample:end_sample].astype(np.float32))

    if not clips:
        return np.array([], dtype=np.float32)

    return np.concatenate(clips).astype(np.float32)


def remove_silence(
    audio: np.ndarray,
    sample_rate: int,
    top_db: float = 30.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Backward-compatible silence removal function.

    It returns:
    - processed speech-only audio
    - preprocessing metrics
    """
    audio = np.asarray(audio, dtype=np.float32)

    original_duration_seconds = len(audio) / sample_rate if sample_rate > 0 else 0.0

    vad_result = detect_speech_regions(
        audio=audio,
        sample_rate=sample_rate,
        top_db=top_db,
        min_rms=0.005,
        min_region_duration_seconds=0.25,
        merge_gap_seconds=0.2,
        adaptive_min_rms=False,
    )

    processed_audio = concatenate_speech_regions(
        audio=audio,
        sample_rate=sample_rate,
        speech_regions=vad_result["speech_regions"],
    )

    if len(processed_audio) == 0:
        processed_audio = audio

    processed_duration_seconds = len(processed_audio) / sample_rate if sample_rate > 0 else 0.0
    removed_silence_seconds = max(0.0, original_duration_seconds - processed_duration_seconds)

    metrics = {
        "original_duration_seconds": round(original_duration_seconds, 3),
        "processed_duration_seconds": round(processed_duration_seconds, 3),
        "removed_silence_seconds": round(removed_silence_seconds, 3),
        "removed_silence_percentage": (
            round((removed_silence_seconds / original_duration_seconds) * 100.0, 2)
            if original_duration_seconds > 0
            else 0.0
        ),
        "speech_region_count": vad_result["speech_region_count"],
        "removed_non_speech_count": vad_result["removed_non_speech_count"],
        "speech_regions": vad_result["speech_regions"],
        "removed_non_speech_segments": vad_result["removed_non_speech_segments"],
    }

    return processed_audio.astype(np.float32), metrics
