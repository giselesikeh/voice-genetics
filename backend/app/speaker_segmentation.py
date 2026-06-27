
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler


_ECAPA_CLASSIFIER = None
_ECAPA_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
_ECAPA_MODEL_SAVEDIR = (
    Path(__file__).resolve().parents[1]
    / "pretrained_models"
    / "spkrec-ecapa-voxceleb"
)

_WAVLM_FEATURE_EXTRACTOR = None
_WAVLM_MODEL = None
_WAVLM_MODEL_SOURCE = "microsoft/wavlm-base-plus"
_WAVLM_MODEL_SAVEDIR = (
    Path(__file__).resolve().parents[1]
    / "pretrained_models"
    / "wavlm-base-plus"
)


def speaker_segmentation_placeholder() -> Dict[str, Any]:
    return {
        "enabled": False,
        "method": "not_selected",
        "message": "No speaker segmentation method was selected.",
        "segmentation_source": "none",
        "automatic_method_used": False,
        "detected_speakers": None,
        "speaker_segments": [],
    }


# -------------------------------------------------------------------------
# Method 1: Manual timestamp-based speaker labeling
# -------------------------------------------------------------------------

def validate_manual_segments(
    segments: List[Dict[str, Any]],
    audio_duration_seconds: float,
) -> List[Dict[str, Any]]:
    validated_segments: List[Dict[str, Any]] = []

    for index, segment in enumerate(segments):
        if "speaker" not in segment:
            raise ValueError(f"Segment {index} is missing 'speaker'.")

        if "start" not in segment:
            raise ValueError(f"Segment {index} is missing 'start' time.")

        if "end" not in segment:
            raise ValueError(f"Segment {index} is missing 'end' time.")

        speaker = str(segment["speaker"])
        start = float(segment["start"])
        end = float(segment["end"])

        if start < 0:
            raise ValueError(f"Segment {index} has a negative start time.")

        if end <= start:
            raise ValueError(f"Segment {index} has an end time before or equal to start time.")

        if end > audio_duration_seconds:
            raise ValueError(
                f"Segment {index} ends after the audio duration. "
                f"Audio duration is {audio_duration_seconds:.3f} seconds."
            )

        validated_segments.append(
            {
                "speaker": speaker,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
            }
        )

    return validated_segments


def extract_speaker_audio_segments(
    audio: np.ndarray,
    sample_rate: int,
    segments: List[Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    speaker_audio: Dict[str, List[np.ndarray]] = {}

    for segment in segments:
        speaker = str(segment["speaker"])
        start_sample = int(float(segment["start"]) * sample_rate)
        end_sample = int(float(segment["end"]) * sample_rate)

        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        clip = audio[start_sample:end_sample]

        if len(clip) == 0:
            continue

        if speaker not in speaker_audio:
            speaker_audio[speaker] = []

        speaker_audio[speaker].append(clip.astype(np.float32))

    joined_speaker_audio: Dict[str, np.ndarray] = {}

    for speaker, clips in speaker_audio.items():
        if len(clips) > 0:
            joined_speaker_audio[speaker] = np.concatenate(clips).astype(np.float32)

    return joined_speaker_audio


def compute_speaker_segmentation_summary(
    validated_segments: List[Dict[str, Any]],
    method_name: str = "manual_speaker_labels",
    message: str = "Manual speaker labels were provided and processed.",
) -> Dict[str, Any]:
    speaker_durations: Dict[str, float] = {}
    speaker_segment_counts: Dict[str, int] = {}

    for segment in validated_segments:
        speaker = segment["speaker"]
        duration = float(segment["duration"])

        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration
        speaker_segment_counts[speaker] = speaker_segment_counts.get(speaker, 0) + 1

    speaker_durations = {
        speaker: round(duration, 3)
        for speaker, duration in speaker_durations.items()
    }

    is_manual = method_name == "manual_speaker_labels"

    return {
        "enabled": True,
        "method": method_name,
        "message": message,
        "segmentation_source": "manual_reference" if is_manual else "automatic",
        "automatic_method_used": not is_manual,
        "detected_speakers": len(speaker_durations),
        "speaker_speech_duration_seconds": speaker_durations,
        "speaker_segment_count": speaker_segment_counts,
        "overlap_warning": "handled_in_v2_pipeline",
        "low_confidence_warning": False,
        "speaker_segments": validated_segments,
    }


# -------------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------------

def _rms(audio: np.ndarray) -> float:
    if audio is None or len(audio) == 0:
        return 0.0

    audio = np.asarray(audio, dtype=np.float32)

    return float(np.sqrt(np.mean(audio ** 2)))


def _l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    return matrix / np.maximum(norms, eps)


def _stable_speaker_names(labels: np.ndarray) -> List[str]:
    label_to_speaker: Dict[int, str] = {}
    speaker_names: List[str] = []

    for label in labels:
        label_int = int(label)

        if label_int not in label_to_speaker:
            label_to_speaker[label_int] = f"speaker_{len(label_to_speaker) + 1}"

        speaker_names.append(label_to_speaker[label_int])

    return speaker_names


def _merge_adjacent_segments(
    segments: List[Dict[str, Any]],
    max_gap_seconds: float = 0.0,
) -> List[Dict[str, Any]]:
    if not segments:
        return []

    sorted_segments = sorted(
        segments,
        key=lambda item: (float(item["start"]), float(item["end"])),
    )

    merged: List[Dict[str, Any]] = [sorted_segments[0].copy()]

    for segment in sorted_segments[1:]:
        last = merged[-1]
        gap = float(segment["start"]) - float(last["end"])

        if last["speaker"] == segment["speaker"] and gap <= max_gap_seconds + 1e-6:
            old_duration = float(last["duration"])
            new_duration = float(segment["duration"])
            total_duration = old_duration + new_duration

            last["end"] = segment["end"]
            last["duration"] = round(float(last["end"]) - float(last["start"]), 3)

            if "rms" in last and "rms" in segment and total_duration > 0:
                last["rms"] = round(
                    float(
                        (
                            float(last["rms"]) * old_duration
                            + float(segment["rms"]) * new_duration
                        )
                        / total_duration
                    ),
                    6,
                )

            if "confidence" in last and "confidence" in segment and total_duration > 0:
                last["confidence"] = round(
                    float(
                        (
                            float(last["confidence"]) * old_duration
                            + float(segment["confidence"]) * new_duration
                        )
                        / total_duration
                    ),
                    4,
                )
        else:
            merged.append(segment.copy())

    return merged


def _build_removed_segments_from_speech_regions(
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
                "reason": "no_speech_region_from_central_vad",
            }
        ]

    regions = sorted(speech_regions, key=lambda item: float(item["start"]))

    first_start = float(regions[0]["start"])

    if first_start > 0:
        removed_segments.append(
            {
                "start": 0.0,
                "end": round(first_start, 3),
                "duration": round(first_start, 3),
                "reason": "leading_non_speech_from_central_vad",
            }
        )

    for previous_region, next_region in zip(regions, regions[1:]):
        gap_start = float(previous_region["end"])
        gap_end = float(next_region["start"])

        if gap_end > gap_start:
            removed_segments.append(
                {
                    "start": round(gap_start, 3),
                    "end": round(gap_end, 3),
                    "duration": round(gap_end - gap_start, 3),
                    "reason": "central_vad_gap_or_pause",
                }
            )

    last_end = float(regions[-1]["end"])

    if last_end < audio_duration_seconds:
        removed_segments.append(
            {
                "start": round(last_end, 3),
                "end": round(audio_duration_seconds, 3),
                "duration": round(audio_duration_seconds - last_end, 3),
                "reason": "trailing_non_speech_from_central_vad",
            }
        )

    return removed_segments


def _default_full_audio_region(
    audio: np.ndarray,
    sample_rate: int,
) -> List[Dict[str, Any]]:
    duration = len(audio) / sample_rate

    return [
        {
            "start": 0.0,
            "end": round(duration, 3),
            "duration": round(duration, 3),
            "rms": round(_rms(audio), 6),
        }
    ]


def _collect_non_overlapping_chunks_from_regions(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration_seconds: float,
    min_rms: float,
    speech_regions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]]]:
    chunk_size = int(chunk_duration_seconds * sample_rate)

    if chunk_size <= 0:
        raise ValueError("chunk_duration_seconds must be greater than zero.")

    audio_duration_seconds = len(audio) / sample_rate

    regions = speech_regions if speech_regions else _default_full_audio_region(audio, sample_rate)

    chunks: List[np.ndarray] = []
    chunk_segments: List[Dict[str, Any]] = []
    removed_segments: List[Dict[str, Any]] = []

    if speech_regions is not None:
        removed_segments.extend(
            _build_removed_segments_from_speech_regions(
                speech_regions,
                audio_duration_seconds,
            )
        )

    for region_index, region in enumerate(regions):
        region_start_sample = int(float(region["start"]) * sample_rate)
        region_end_sample = int(float(region["end"]) * sample_rate)

        region_start_sample = max(0, region_start_sample)
        region_end_sample = min(len(audio), region_end_sample)

        start_sample = region_start_sample

        while start_sample < region_end_sample:
            end_sample = min(start_sample + chunk_size, region_end_sample)
            chunk = audio[start_sample:end_sample]

            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            duration = end_time - start_time

            if duration < 1.0:
                removed_segments.append(
                    {
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        "duration": round(duration, 3),
                        "reason": "chunk_shorter_than_one_second",
                        "rms": round(_rms(chunk), 6),
                    }
                )
                break

            chunk_rms = _rms(chunk)

            if chunk_rms < min_rms:
                removed_segments.append(
                    {
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        "duration": round(duration, 3),
                        "reason": "chunk_rms_below_threshold",
                        "rms": round(chunk_rms, 6),
                    }
                )
            else:
                chunks.append(chunk.astype(np.float32))

                chunk_segments.append(
                    {
                        "region_index": region_index,
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                        "duration": round(duration, 3),
                        "rms": round(chunk_rms, 6),
                    }
                )

            start_sample = end_sample

    return chunks, chunk_segments, removed_segments


def _collect_overlapping_chunks_from_regions(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration_seconds: float,
    chunk_hop_seconds: float,
    min_chunk_duration_seconds: float,
    min_rms: float,
    speech_regions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    ECAPA V2 chunk collector.

    Important fix:
    short but valid speech regions are no longer dropped.
    If a VAD speech region is shorter than the ECAPA chunk size, it is kept with
    its original timestamp and padded later only for embedding extraction.
    """
    chunk_samples = int(chunk_duration_seconds * sample_rate)
    hop_samples = int(chunk_hop_seconds * sample_rate)
    min_chunk_samples = int(min_chunk_duration_seconds * sample_rate)

    if chunk_samples <= 0:
        raise ValueError("chunk_duration_seconds must be greater than zero.")

    if hop_samples <= 0:
        raise ValueError("chunk_hop_seconds must be greater than zero.")

    audio_duration_seconds = len(audio) / sample_rate

    regions = speech_regions if speech_regions else _default_full_audio_region(audio, sample_rate)

    chunks: List[np.ndarray] = []
    chunk_segments: List[Dict[str, Any]] = []
    removed_segments: List[Dict[str, Any]] = []

    if speech_regions is not None:
        removed_segments.extend(
            _build_removed_segments_from_speech_regions(
                speech_regions,
                audio_duration_seconds,
            )
        )

    for region_index, region in enumerate(regions):
        region_start_sample = int(float(region["start"]) * sample_rate)
        region_end_sample = int(float(region["end"]) * sample_rate)

        region_start_sample = max(0, region_start_sample)
        region_end_sample = min(len(audio), region_end_sample)

        region_length = region_end_sample - region_start_sample
        region_duration = region_length / sample_rate if sample_rate > 0 else 0.0
        region_audio = audio[region_start_sample:region_end_sample]
        region_rms = _rms(region_audio)

        if region_length <= 0:
            continue

        if region_length < min_chunk_samples:
            removed_segments.append(
                {
                    "start": round(region_start_sample / sample_rate, 3),
                    "end": round(region_end_sample / sample_rate, 3),
                    "duration": round(region_duration, 3),
                    "reason": "speech_region_shorter_than_min_allowed_duration",
                    "rms": round(region_rms, 6),
                }
            )
            continue

        if region_rms < min_rms:
            removed_segments.append(
                {
                    "start": round(region_start_sample / sample_rate, 3),
                    "end": round(region_end_sample / sample_rate, 3),
                    "duration": round(region_duration, 3),
                    "reason": "speech_region_rms_below_threshold",
                    "rms": round(region_rms, 6),
                }
            )
            continue

        if region_length <= chunk_samples:
            start_positions = [region_start_sample]
        else:
            start_positions = []
            current_start = region_start_sample

            while current_start + chunk_samples <= region_end_sample:
                start_positions.append(current_start)
                current_start += hop_samples

            final_start = region_end_sample - chunk_samples

            if not start_positions or final_start > start_positions[-1]:
                start_positions.append(final_start)

        seen_positions = set()

        for start_sample in start_positions:
            if start_sample in seen_positions:
                continue

            seen_positions.add(start_sample)

            end_sample = min(start_sample + chunk_samples, region_end_sample)
            chunk = audio[start_sample:end_sample]

            chunk_duration = (end_sample - start_sample) / sample_rate
            chunk_rms = _rms(chunk)

            if len(chunk) < min_chunk_samples:
                removed_segments.append(
                    {
                        "start": round(start_sample / sample_rate, 3),
                        "end": round(end_sample / sample_rate, 3),
                        "duration": round(chunk_duration, 3),
                        "reason": "chunk_shorter_than_min_allowed_duration",
                        "rms": round(chunk_rms, 6),
                    }
                )
                continue

            if chunk_rms < min_rms:
                removed_segments.append(
                    {
                        "start": round(start_sample / sample_rate, 3),
                        "end": round(end_sample / sample_rate, 3),
                        "duration": round(chunk_duration, 3),
                        "reason": "chunk_rms_below_threshold",
                        "rms": round(chunk_rms, 6),
                    }
                )
                continue

            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate

            chunks.append(chunk.astype(np.float32))

            chunk_segments.append(
                {
                    "region_index": region_index,
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                    "center": round((start_time + end_time) / 2.0, 3),
                    "duration": round(end_time - start_time, 3),
                    "rms": round(chunk_rms, 6),
                    "padded_for_embedding": bool(len(chunk) < chunk_samples),
                }
            )

    return chunks, chunk_segments, removed_segments


# -------------------------------------------------------------------------
# Method 2: DSP feature clustering baseline
# -------------------------------------------------------------------------

def _extract_chunk_features(chunk: np.ndarray, sample_rate: int) -> np.ndarray:
    chunk_rms = _rms(chunk)

    spectral_centroid = librosa.feature.spectral_centroid(y=chunk, sr=sample_rate)
    spectral_centroid_mean = float(np.mean(spectral_centroid))

    mfcc = librosa.feature.mfcc(y=chunk, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    pitch_mean = 0.0

    try:
        f0, _, _ = librosa.pyin(
            chunk,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )

        valid_f0 = f0[~np.isnan(f0)]

        if len(valid_f0) > 0:
            pitch_mean = float(np.mean(valid_f0))

    except Exception:
        pitch_mean = 0.0

    return np.array(
        [chunk_rms, spectral_centroid_mean, pitch_mean, *mfcc_mean],
        dtype=np.float32,
    )


def automatic_vad_speaker_clustering(
    audio: np.ndarray,
    sample_rate: int,
    expected_speakers: int = 2,
    chunk_duration_seconds: float = 2.0,
    min_rms: float = 0.005,
    speech_regions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    audio_duration_seconds = len(audio) / sample_rate

    chunks, chunk_segments, removed_segments = _collect_non_overlapping_chunks_from_regions(
        audio=audio,
        sample_rate=sample_rate,
        chunk_duration_seconds=chunk_duration_seconds,
        min_rms=min_rms,
        speech_regions=speech_regions,
    )

    if len(chunk_segments) == 0:
        raise ValueError("No speech-like chunks were detected for clustering.")

    if len(chunk_segments) < expected_speakers:
        raise ValueError(
            f"Not enough speech chunks for {expected_speakers} speakers. "
            f"Detected only {len(chunk_segments)} usable chunks."
        )

    feature_matrix = np.vstack(
        [_extract_chunk_features(chunk, sample_rate) for chunk in chunks]
    )

    scaled_features = StandardScaler().fit_transform(feature_matrix)

    kmeans = KMeans(
        n_clusters=expected_speakers,
        random_state=42,
        n_init=10,
    )

    labels = kmeans.fit_predict(scaled_features)
    speaker_names = _stable_speaker_names(labels)

    automatic_segments: List[Dict[str, Any]] = []

    for segment, speaker_name in zip(chunk_segments, speaker_names):
        automatic_segments.append(
            {
                "speaker": speaker_name,
                "start": segment["start"],
                "end": segment["end"],
                "duration": segment["duration"],
                "rms": segment["rms"],
            }
        )

    merged_segments = _merge_adjacent_segments(automatic_segments)

    automatic_summary = compute_speaker_segmentation_summary(
        merged_segments,
        method_name="vad_simple_speaker_clustering",
        message=(
            "Speech chunks were grouped using handcrafted DSP features and K-Means. "
            "When central VAD is available, chunks are created only inside detected speech regions."
        ),
    )

    automatic_summary["chunk_duration_seconds"] = chunk_duration_seconds
    automatic_summary["expected_speakers"] = expected_speakers
    automatic_summary["usable_speech_chunks"] = len(chunk_segments)
    automatic_summary["removed_pause_chunks"] = len(removed_segments)
    automatic_summary["removed_pause_segments"] = removed_segments
    automatic_summary["audio_duration_seconds"] = round(audio_duration_seconds, 3)
    automatic_summary["uses_central_vad_regions"] = speech_regions is not None
    automatic_summary["clustering_backend"] = "kmeans"
    automatic_summary["embedding_type"] = "handcrafted_dsp_features"

    return merged_segments, automatic_summary


# -------------------------------------------------------------------------
# Method 3: ECAPA-TDNN embeddings
# -------------------------------------------------------------------------

def _get_ecapa_classifier():
    global _ECAPA_CLASSIFIER

    if _ECAPA_CLASSIFIER is None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception as exc:
            raise ValueError(
                "SpeechBrain is not available. Install it with: pip install speechbrain"
            ) from exc

        try:
            _ECAPA_CLASSIFIER = EncoderClassifier.from_hparams(
                source=_ECAPA_MODEL_SOURCE,
                savedir=str(_ECAPA_MODEL_SAVEDIR),
                run_opts={"device": "cpu"},
            )
        except Exception as exc:
            raise ValueError(
                "Could not load ECAPA-TDNN model. "
                "Check internet connection for first-time model download, "
                "or confirm that the model exists in backend/pretrained_models."
            ) from exc

    return _ECAPA_CLASSIFIER


def _extract_ecapa_embedding(chunk: np.ndarray) -> np.ndarray:
    try:
        import torch
    except Exception as exc:
        raise ValueError(
            "PyTorch is not available. Install it with: pip install torch torchaudio"
        ) from exc

    classifier = _get_ecapa_classifier()

    chunk = np.asarray(chunk, dtype=np.float32)

    if chunk.ndim != 1:
        chunk = np.mean(chunk, axis=0).astype(np.float32)

    waveform = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        embedding = classifier.encode_batch(waveform)

    embedding_np = embedding.squeeze().detach().cpu().numpy().astype(np.float32)

    return embedding_np


def _extract_ecapa_embeddings_batched(
    chunks: List[np.ndarray],
    target_chunk_samples: int,
    batch_size: int = 8,
) -> List[np.ndarray]:
    try:
        import torch
    except Exception as exc:
        raise ValueError(
            "PyTorch is not available. Install it with: pip install torch torchaudio"
        ) from exc

    classifier = _get_ecapa_classifier()
    prepared_chunks: List[np.ndarray] = []

    for chunk in chunks:
        chunk = np.asarray(chunk, dtype=np.float32)

        if chunk.ndim != 1:
            chunk = np.mean(chunk, axis=0).astype(np.float32)

        if len(chunk) < target_chunk_samples:
            chunk = np.pad(chunk, (0, target_chunk_samples - len(chunk)))
        elif len(chunk) > target_chunk_samples:
            chunk = chunk[:target_chunk_samples]

        prepared_chunks.append(chunk.astype(np.float32))

    embeddings: List[np.ndarray] = []

    for start_index in range(0, len(prepared_chunks), batch_size):
        batch = prepared_chunks[start_index:start_index + batch_size]
        waveform = torch.tensor(np.stack(batch), dtype=torch.float32)

        with torch.no_grad():
            batch_embeddings = classifier.encode_batch(waveform)

        batch_embeddings_np = batch_embeddings.squeeze().detach().cpu().numpy().astype(np.float32)

        if batch_embeddings_np.ndim == 1:
            batch_embeddings_np = batch_embeddings_np.reshape(1, -1)

        for embedding in batch_embeddings_np:
            embeddings.append(embedding.astype(np.float32))

    return embeddings


def ecapa_speaker_embedding_clustering(
    audio: np.ndarray,
    sample_rate: int,
    expected_speakers: int = 2,
    chunk_duration_seconds: float = 2.0,
    min_rms: float = 0.005,
    speech_regions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if sample_rate != 16000:
        raise ValueError(
            f"ECAPA method expects 16 kHz audio. Current sample rate is {sample_rate}."
        )

    audio_duration_seconds = len(audio) / sample_rate

    chunks, chunk_segments, removed_segments = _collect_non_overlapping_chunks_from_regions(
        audio=audio,
        sample_rate=sample_rate,
        chunk_duration_seconds=chunk_duration_seconds,
        min_rms=min_rms,
        speech_regions=speech_regions,
    )

    if len(chunk_segments) == 0:
        raise ValueError("No speech-like chunks were detected for ECAPA clustering.")

    if len(chunk_segments) < expected_speakers:
        raise ValueError(
            f"Not enough speech chunks for {expected_speakers} speakers. "
            f"Detected only {len(chunk_segments)} usable chunks."
        )

    embeddings = [_extract_ecapa_embedding(chunk) for chunk in chunks]

    embedding_matrix = np.vstack(embeddings)
    scaled_embeddings = StandardScaler().fit_transform(embedding_matrix)

    kmeans = KMeans(
        n_clusters=expected_speakers,
        random_state=42,
        n_init=10,
    )

    labels = kmeans.fit_predict(scaled_embeddings)
    speaker_names = _stable_speaker_names(labels)

    automatic_segments: List[Dict[str, Any]] = []

    for segment, speaker_name in zip(chunk_segments, speaker_names):
        automatic_segments.append(
            {
                "speaker": speaker_name,
                "start": segment["start"],
                "end": segment["end"],
                "duration": segment["duration"],
                "rms": segment["rms"],
            }
        )

    merged_segments = _merge_adjacent_segments(automatic_segments)

    ecapa_summary = compute_speaker_segmentation_summary(
        merged_segments,
        method_name="ecapa_tdnn_speaker_embedding_clustering",
        message=(
            "Speech chunks were grouped using ECAPA-TDNN speaker embeddings and K-Means. "
            "When central VAD is available, chunks are created only inside detected speech regions."
        ),
    )

    ecapa_summary["chunk_duration_seconds"] = chunk_duration_seconds
    ecapa_summary["expected_speakers"] = expected_speakers
    ecapa_summary["usable_speech_chunks"] = len(chunk_segments)
    ecapa_summary["removed_pause_chunks"] = len(removed_segments)
    ecapa_summary["removed_pause_segments"] = removed_segments
    ecapa_summary["audio_duration_seconds"] = round(audio_duration_seconds, 3)
    ecapa_summary["embedding_model"] = _ECAPA_MODEL_SOURCE
    ecapa_summary["embedding_type"] = "ECAPA-TDNN speaker embedding"
    ecapa_summary["uses_central_vad_regions"] = speech_regions is not None
    ecapa_summary["clustering_backend"] = "kmeans"
    ecapa_summary["embedding_scaling"] = "standard_scaler"
    ecapa_summary["important_note"] = (
        "This is the baseline ECAPA method. For the improved full pipeline, use segmentation_method=ecapa_v2."
    )

    return merged_segments, ecapa_summary


# -------------------------------------------------------------------------
# Method 3B: ECAPA V2 full-pipeline diarization
# -------------------------------------------------------------------------

def _cluster_embeddings_agglomerative_cosine(
    embedding_matrix: np.ndarray,
    expected_speakers: int,
) -> np.ndarray:
    if len(embedding_matrix) < expected_speakers:
        raise ValueError(
            f"Not enough embeddings for {expected_speakers} speakers. "
            f"Detected only {len(embedding_matrix)} usable embeddings."
        )

    normalized_embeddings = _l2_normalize(embedding_matrix)

    try:
        clustering = AgglomerativeClustering(
            n_clusters=expected_speakers,
            metric="cosine",
            linkage="average",
        )
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=expected_speakers,
            affinity="cosine",
            linkage="average",
        )

    labels = clustering.fit_predict(normalized_embeddings)

    return labels.astype(int)


def _smooth_isolated_labels(labels: np.ndarray, passes: int = 1) -> np.ndarray:
    if len(labels) < 3:
        return labels.astype(int)

    smoothed = labels.astype(int).copy()

    for _ in range(max(passes, 0)):
        updated = smoothed.copy()

        for index in range(1, len(smoothed) - 1):
            previous_label = smoothed[index - 1]
            current_label = smoothed[index]
            next_label = smoothed[index + 1]

            if previous_label == next_label and current_label != previous_label:
                updated[index] = previous_label

        smoothed = updated

    return smoothed.astype(int)


def _compute_cluster_confidences(
    normalized_embeddings: np.ndarray,
    labels: np.ndarray,
) -> List[float]:
    centroids: Dict[int, np.ndarray] = {}

    for label in sorted(set(int(label) for label in labels)):
        members = normalized_embeddings[labels == label]
        centroid = np.mean(members, axis=0, keepdims=True)
        centroid = _l2_normalize(centroid)[0]
        centroids[int(label)] = centroid

    confidences: List[float] = []

    for embedding, label in zip(normalized_embeddings, labels):
        centroid = centroids[int(label)]
        cosine_similarity = float(np.dot(embedding, centroid))
        confidence = max(0.0, min(1.0, (cosine_similarity + 1.0) / 2.0))
        confidences.append(confidence)

    return confidences


def _chunks_to_non_overlapping_segments(
    chunk_segments: List[Dict[str, Any]],
    speaker_names: List[str],
    confidences: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    if not chunk_segments:
        return []

    if confidences is None:
        confidences = [1.0] * len(chunk_segments)

    packed: List[Dict[str, Any]] = []

    for segment, speaker, confidence in zip(chunk_segments, speaker_names, confidences):
        packed.append(
            {
                **segment,
                "speaker": speaker,
                "confidence": float(confidence),
            }
        )

    packed.sort(key=lambda item: (int(item["region_index"]), float(item.get("center", item["start"]))))

    final_segments: List[Dict[str, Any]] = []

    current_region_index = int(packed[0]["region_index"])
    region_items: List[Dict[str, Any]] = []

    def flush_region(items: List[Dict[str, Any]]) -> None:
        if not items:
            return

        items.sort(key=lambda item: float(item.get("center", item["start"])))

        if len(items) == 1:
            item = items[0]

            final_segments.append(
                {
                    "speaker": item["speaker"],
                    "start": round(float(item["start"]), 3),
                    "end": round(float(item["end"]), 3),
                    "duration": round(float(item["end"]) - float(item["start"]), 3),
                    "confidence": round(float(item["confidence"]), 4),
                    "rms": round(float(item["rms"]), 6),
                    "padded_for_embedding": bool(item.get("padded_for_embedding", False)),
                }
            )
            return

        boundaries: List[float] = [float(items[0]["start"])]

        for left_item, right_item in zip(items, items[1:]):
            left_center = float(left_item.get("center", left_item["start"]))
            right_center = float(right_item.get("center", right_item["start"]))
            boundaries.append((left_center + right_center) / 2.0)

        boundaries.append(float(items[-1]["end"]))

        active_speaker = items[0]["speaker"]
        active_start = boundaries[0]
        active_confidences = [float(items[0]["confidence"])]
        active_rms_values = [float(items[0]["rms"])]
        active_padded = bool(items[0].get("padded_for_embedding", False))

        for index in range(1, len(items)):
            item = items[index]
            item_speaker = item["speaker"]
            boundary = boundaries[index]

            if item_speaker == active_speaker:
                active_confidences.append(float(item["confidence"]))
                active_rms_values.append(float(item["rms"]))
                active_padded = active_padded or bool(item.get("padded_for_embedding", False))
                continue

            if boundary > active_start:
                final_segments.append(
                    {
                        "speaker": active_speaker,
                        "start": round(active_start, 3),
                        "end": round(boundary, 3),
                        "duration": round(boundary - active_start, 3),
                        "confidence": round(float(np.mean(active_confidences)), 4),
                        "rms": round(float(np.mean(active_rms_values)), 6),
                        "padded_for_embedding": active_padded,
                    }
                )

            active_speaker = item_speaker
            active_start = boundary
            active_confidences = [float(item["confidence"])]
            active_rms_values = [float(item["rms"])]
            active_padded = bool(item.get("padded_for_embedding", False))

        final_end = boundaries[-1]

        if final_end > active_start:
            final_segments.append(
                {
                    "speaker": active_speaker,
                    "start": round(active_start, 3),
                    "end": round(final_end, 3),
                    "duration": round(final_end - active_start, 3),
                    "confidence": round(float(np.mean(active_confidences)), 4),
                    "rms": round(float(np.mean(active_rms_values)), 6),
                    "padded_for_embedding": active_padded,
                }
            )

    for item in packed:
        item_region_index = int(item["region_index"])

        if item_region_index != current_region_index:
            flush_region(region_items)
            region_items = []
            current_region_index = item_region_index

        region_items.append(item)

    flush_region(region_items)

    return _merge_adjacent_segments(final_segments, max_gap_seconds=0.15)


def ecapa_v2_speaker_embedding_clustering(
    audio: np.ndarray,
    sample_rate: int,
    expected_speakers: int = 2,
    chunk_duration_seconds: float = 2.0,
    chunk_hop_seconds: Optional[float] = None,
    min_rms: float = 0.02,
    speech_regions: Optional[List[Dict[str, Any]]] = None,
    smoothing_passes: int = 1,
    embedding_batch_size: int = 8,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if sample_rate != 16000:
        raise ValueError(
            f"ECAPA V2 method expects 16 kHz audio. Current sample rate is {sample_rate}."
        )

    if expected_speakers < 1:
        raise ValueError("expected_speakers must be at least 1.")

    if chunk_duration_seconds <= 0:
        raise ValueError("chunk_duration_seconds must be greater than zero.")

    if chunk_hop_seconds is None:
        chunk_hop_seconds = chunk_duration_seconds / 2.0

    if chunk_hop_seconds <= 0:
        raise ValueError("chunk_hop_seconds must be greater than zero.")

    audio_duration_seconds = len(audio) / sample_rate

    chunks, chunk_segments, removed_segments = _collect_overlapping_chunks_from_regions(
        audio=audio,
        sample_rate=sample_rate,
        chunk_duration_seconds=chunk_duration_seconds,
        chunk_hop_seconds=chunk_hop_seconds,
        min_chunk_duration_seconds=0.25,
        min_rms=min_rms,
        speech_regions=speech_regions,
    )

    if len(chunk_segments) == 0:
        raise ValueError("No usable speech chunks were detected for ECAPA V2 clustering.")

    if len(chunk_segments) < expected_speakers:
        raise ValueError(
            f"Not enough usable chunks for {expected_speakers} speakers. "
            f"Detected only {len(chunk_segments)} usable chunks."
        )

    target_chunk_samples = int(chunk_duration_seconds * sample_rate)

    embeddings = _extract_ecapa_embeddings_batched(
        chunks=chunks,
        target_chunk_samples=target_chunk_samples,
        batch_size=embedding_batch_size,
    )

    embedding_matrix = np.vstack(embeddings)
    normalized_embeddings = _l2_normalize(embedding_matrix)

    raw_labels = _cluster_embeddings_agglomerative_cosine(
        normalized_embeddings,
        expected_speakers=expected_speakers,
    )

    smoothed_labels = _smooth_isolated_labels(
        raw_labels,
        passes=smoothing_passes,
    )

    speaker_names = _stable_speaker_names(smoothed_labels)

    confidences = _compute_cluster_confidences(
        normalized_embeddings=normalized_embeddings,
        labels=smoothed_labels,
    )

    automatic_segments = _chunks_to_non_overlapping_segments(
        chunk_segments=chunk_segments,
        speaker_names=speaker_names,
        confidences=confidences,
    )

    if len(automatic_segments) == 0:
        raise ValueError("ECAPA V2 could not produce final speaker segments.")

    ecapa_summary = compute_speaker_segmentation_summary(
        automatic_segments,
        method_name="ecapa_v2_full_pipeline",
        message=(
            "Central VAD speech regions were converted into overlapping chunks, "
            "short valid speech regions were kept and padded internally for ECAPA embeddings, "
            "then embeddings were clustered using L2-normalized cosine agglomerative clustering."
        ),
    )

    if speech_regions:
        speech_duration = sum(float(region["duration"]) for region in speech_regions)
        speech_region_count = len(speech_regions)
    else:
        speech_duration = audio_duration_seconds
        speech_region_count = 1

    mean_confidence = float(np.mean(confidences)) if len(confidences) > 0 else 0.0
    min_confidence = float(np.min(confidences)) if len(confidences) > 0 else 0.0
    padded_chunk_count = sum(1 for segment in chunk_segments if segment.get("padded_for_embedding", False))

    ecapa_summary["diarization_pipeline_version"] = "ecapa_v2_full_pipeline_keep_short_regions"
    ecapa_summary["chunk_duration_seconds"] = chunk_duration_seconds
    ecapa_summary["chunk_hop_seconds"] = round(chunk_hop_seconds, 3)
    ecapa_summary["expected_speakers"] = expected_speakers
    ecapa_summary["usable_speech_chunks"] = len(chunk_segments)
    ecapa_summary["padded_short_chunks"] = padded_chunk_count
    ecapa_summary["short_region_padding_enabled"] = True
    ecapa_summary["minimum_kept_region_seconds"] = 0.25
    ecapa_summary["removed_pause_chunks"] = len(removed_segments)
    ecapa_summary["removed_pause_segments"] = removed_segments
    ecapa_summary["audio_duration_seconds"] = round(audio_duration_seconds, 3)
    ecapa_summary["uses_central_vad_regions"] = speech_regions is not None
    ecapa_summary["central_vad_speech_region_count"] = speech_region_count
    ecapa_summary["central_vad_speech_duration_seconds"] = round(speech_duration, 3)
    ecapa_summary["central_vad_speech_coverage_ratio"] = (
        round(speech_duration / audio_duration_seconds, 4)
        if audio_duration_seconds > 0
        else 0.0
    )
    ecapa_summary["embedding_model"] = _ECAPA_MODEL_SOURCE
    ecapa_summary["embedding_type"] = "ECAPA-TDNN speaker embedding"
    ecapa_summary["embedding_normalization"] = "l2"
    ecapa_summary["clustering_backend"] = "agglomerative_cosine_average_linkage"
    ecapa_summary["smoothing_passes"] = smoothing_passes
    ecapa_summary["mean_cluster_confidence"] = round(mean_confidence, 4)
    ecapa_summary["min_cluster_confidence"] = round(min_confidence, 4)
    ecapa_summary["low_confidence_warning"] = bool(mean_confidence < 0.60 or min_confidence < 0.50)
    ecapa_summary["important_note"] = (
        "Speaker names are anonymous cluster labels. Compare predicted speakers "
        "with manual labels using label mapping, not by assuming speaker_1 is always the same person."
    )

    return automatic_segments, ecapa_summary


# -------------------------------------------------------------------------
# Method 4: WavLM embedding clustering
# -------------------------------------------------------------------------

def _get_wavlm_model():
    global _WAVLM_FEATURE_EXTRACTOR
    global _WAVLM_MODEL

    if _WAVLM_FEATURE_EXTRACTOR is None or _WAVLM_MODEL is None:
        try:
            from transformers import AutoFeatureExtractor, AutoModel
        except Exception as exc:
            raise ValueError(
                "WavLM dependencies are not available. Install them with: pip install transformers torch torchaudio"
            ) from exc

        try:
            model_location = (
                str(_WAVLM_MODEL_SAVEDIR)
                if _WAVLM_MODEL_SAVEDIR.exists()
                else _WAVLM_MODEL_SOURCE
            )

            _WAVLM_FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(
                model_location,
                local_files_only=_WAVLM_MODEL_SAVEDIR.exists(),
            )

            _WAVLM_MODEL = AutoModel.from_pretrained(
                model_location,
                local_files_only=_WAVLM_MODEL_SAVEDIR.exists(),
            )

            _WAVLM_MODEL.eval()
            _WAVLM_MODEL.to("cpu")

        except Exception as exc:
            raise ValueError(
                f"Could not load WavLM model from {_WAVLM_MODEL_SAVEDIR}. "
                f"Original error: {str(exc)}"
            ) from exc

    return _WAVLM_FEATURE_EXTRACTOR, _WAVLM_MODEL


def _extract_wavlm_embedding(chunk: np.ndarray, sample_rate: int) -> np.ndarray:
    try:
        import torch
    except Exception as exc:
        raise ValueError(
            "PyTorch is not available. Install it with: pip install torch torchaudio"
        ) from exc

    feature_extractor, model = _get_wavlm_model()

    chunk = np.asarray(chunk, dtype=np.float32)

    if chunk.ndim != 1:
        chunk = np.mean(chunk, axis=0).astype(np.float32)

    inputs = feature_extractor(
        chunk,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )

    inputs = {key: value.to("cpu") for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state
    embedding = hidden_states.mean(dim=1).squeeze().detach().cpu().numpy()

    return embedding.astype(np.float32)


def wavlm_speaker_embedding_clustering(
    audio: np.ndarray,
    sample_rate: int,
    expected_speakers: int = 2,
    chunk_duration_seconds: float = 2.0,
    min_rms: float = 0.005,
    speech_regions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if sample_rate != 16000:
        raise ValueError(
            f"WavLM method expects 16 kHz audio. Current sample rate is {sample_rate}."
        )

    audio_duration_seconds = len(audio) / sample_rate

    chunks, chunk_segments, removed_segments = _collect_non_overlapping_chunks_from_regions(
        audio=audio,
        sample_rate=sample_rate,
        chunk_duration_seconds=chunk_duration_seconds,
        min_rms=min_rms,
        speech_regions=speech_regions,
    )

    if len(chunk_segments) == 0:
        raise ValueError("No speech-like chunks were detected for WavLM clustering.")

    if len(chunk_segments) < expected_speakers:
        raise ValueError(
            f"Not enough speech chunks for {expected_speakers} speakers. "
            f"Detected only {len(chunk_segments)} usable chunks."
        )

    embeddings = [_extract_wavlm_embedding(chunk, sample_rate) for chunk in chunks]

    embedding_matrix = np.vstack(embeddings)
    scaled_embeddings = StandardScaler().fit_transform(embedding_matrix)

    kmeans = KMeans(
        n_clusters=expected_speakers,
        random_state=42,
        n_init=10,
    )

    labels = kmeans.fit_predict(scaled_embeddings)
    speaker_names = _stable_speaker_names(labels)

    automatic_segments: List[Dict[str, Any]] = []

    for segment, speaker_name in zip(chunk_segments, speaker_names):
        automatic_segments.append(
            {
                "speaker": speaker_name,
                "start": segment["start"],
                "end": segment["end"],
                "duration": segment["duration"],
                "rms": segment["rms"],
            }
        )

    merged_segments = _merge_adjacent_segments(automatic_segments)

    wavlm_summary = compute_speaker_segmentation_summary(
        merged_segments,
        method_name="wavlm_speaker_embedding_clustering",
        message=(
            "Speech chunks were grouped using WavLM speech embeddings and K-Means. "
            "When central VAD is available, chunks are created only inside detected speech regions."
        ),
    )

    wavlm_summary["chunk_duration_seconds"] = chunk_duration_seconds
    wavlm_summary["expected_speakers"] = expected_speakers
    wavlm_summary["usable_speech_chunks"] = len(chunk_segments)
    wavlm_summary["removed_pause_chunks"] = len(removed_segments)
    wavlm_summary["removed_pause_segments"] = removed_segments
    wavlm_summary["audio_duration_seconds"] = round(audio_duration_seconds, 3)
    wavlm_summary["embedding_model"] = _WAVLM_MODEL_SOURCE
    wavlm_summary["embedding_type"] = "WavLM pooled speech embedding"
    wavlm_summary["uses_central_vad_regions"] = speech_regions is not None
    wavlm_summary["clustering_backend"] = "kmeans"
    wavlm_summary["embedding_scaling"] = "standard_scaler"

    return merged_segments, wavlm_summary
