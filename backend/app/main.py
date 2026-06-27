import json
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.audio_io import load_audio_file
from app.evaluation import evaluate_diarization
from app.features import extract_basic_features
from app.privacy import get_privacy_status
from app.quality import compute_quality_metrics
from app.schemas import ExtractResponse
from app.speaker_segmentation import (
    automatic_vad_speaker_clustering,
    compute_speaker_segmentation_summary,
    ecapa_speaker_embedding_clustering,
    ecapa_v2_speaker_embedding_clustering,
    extract_speaker_audio_segments,
    speaker_segmentation_placeholder,
    validate_manual_segments,
    wavlm_speaker_embedding_clustering,
)
from app.vad import concatenate_speech_regions, detect_speech_regions, remove_silence


openapi_tags = [
    {
        "name": "System",
        "description": "Basic API health and service information.",
    },
    {
        "name": "Acoustic Analysis",
        "description": "Upload audio or video files and extract acoustic and speaker-level features.",
    },
]


app = FastAPI(
    title="Voice Genetics Acoustic Feature Extraction API",
    description=(
        "Privacy-compliant acoustic feature extraction API. "
        "The API accepts voice/audio files, and can also accept video files "
        "when the audio track is available. It returns quality metrics, "
        "voice activity information, acoustic features, optional speaker segmentation, "
        "speaker-level features, diarization evaluation metrics, warnings, and privacy status."
    ),
    version="0.10.0",
    openapi_tags=openapi_tags,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/", tags=["System"])
def root():
    return {
        "project": "Voice Genetics",
        "message": "Privacy-compliant acoustic feature extraction API",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health", tags=["System"], summary="Health Check")
def health_check():
    return {
        "status": "ok",
        "service": "voice-genetics-api",
        "version": "0.10.0",
        "message": "Voice Genetics API is running",
    }


def _clean_optional_json_text(value: Optional[str]) -> str:
    if value is None:
        return ""

    cleaned = value.strip()

    if cleaned.lower() in {"", "string", "none", "null"}:
        return ""

    return cleaned


def _extract_features_for_speakers(
    speaker_audio_segments,
    sample_rate: int,
):
    speaker_features = {}

    for speaker, speaker_audio in speaker_audio_segments.items():
        trimmed_speaker_audio, speaker_preprocessing = remove_silence(
            speaker_audio,
            sample_rate,
        )

        if len(trimmed_speaker_audio) == 0:
            trimmed_speaker_audio = np.asarray(speaker_audio, dtype=np.float32)

        speaker_features[speaker] = {
            "preprocessing_metrics": speaker_preprocessing,
            "features": extract_basic_features(
                trimmed_speaker_audio,
                sample_rate,
            ),
        }

    return speaker_features


def _empty_evaluation_status(message: str = "No reference_segments_json was provided.") -> dict:
    return {
        "enabled": False,
        "message": message,
    }


@app.post(
    "/extract",
    tags=["Acoustic Analysis"],
    summary="Extract Audio Features",
    response_model=ExtractResponse,
)
async def extract_features(
    file: UploadFile = File(
        ...,
        description=(
            "Upload a .wav, .mp3, .m4a, .mp4, or .mov file. "
            "For video files, only the audio track is processed."
        ),
    ),
    segments_json: Optional[str] = Form(
        default=None,
        description=(
            "Manual speaker segments as a JSON list. "
            "If provided, this overrides segmentation_method and runs Method 1 manual segmentation."
        ),
    ),
    reference_segments_json: Optional[str] = Form(
        default=None,
        description=(
            "Manual reference speaker segments as a JSON list for evaluation only. "
            "This does NOT override automatic segmentation."
        ),
    ),
    segmentation_method: Optional[str] = Form(
        default="none",
        description=(
            "Speaker segmentation method: none, auto, ecapa, ecapa_v2, or wavlm. "
            "Use ecapa_v2 for the improved full-pipeline diarization method."
        ),
    ),
    expected_speakers: int = Form(
        default=2,
        ge=1,
        le=10,
        description="Expected number of speakers in the recording.",
    ),
    chunk_duration_seconds: float = Form(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Chunk duration in seconds for automatic speaker segmentation.",
    ),
    vad_mode: str = Form(
        default="adaptive",
        description="VAD mode: adaptive or fixed. Adaptive lowers RMS threshold for quiet recordings.",
    ),
    vad_top_db: float = Form(
        default=30.0,
        ge=5.0,
        le=60.0,
        description="Central VAD top_db.",
    ),
    vad_min_rms: float = Form(
        default=0.015,
        ge=0.0,
        le=1.0,
        description=(
            "Requested central VAD minimum RMS. "
            "In adaptive mode, quiet recordings may use a lower effective_min_rms."
        ),
    ),
    vad_min_region_duration_seconds: float = Form(
        default=0.25,
        ge=0.1,
        le=5.0,
        description="Minimum duration for a speech-like region.",
    ),
    vad_merge_gap_seconds: float = Form(
        default=0.8,
        ge=0.0,
        le=3.0,
        description="Merge speech regions separated by gaps shorter than this value.",
    ),
    ecapa_chunk_hop_seconds: Optional[float] = Form(
        default=None,
        ge=0.25,
        le=10.0,
        description=(
            "Hop size for ecapa_v2 overlapping chunks. "
            "If empty, it defaults to half of chunk_duration_seconds."
        ),
    ),
    ecapa_smoothing_passes: int = Form(
        default=1,
        ge=0,
        le=5,
        description="Number of isolated-label smoothing passes for ecapa_v2.",
    ),
    evaluation_frame_step_seconds: float = Form(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Frame step used for diarization evaluation metrics.",
    ),
):
    request_start_time = time.perf_counter()
    speaker_processing_seconds = 0.0
    evaluation_processing_seconds = 0.0

    audio, sample_rate, loading_warnings = await load_audio_file(file)

    audio_duration_seconds = len(audio) / sample_rate

    quality_metrics, quality_warnings = compute_quality_metrics(audio, sample_rate)

    normalized_vad_mode = (vad_mode or "adaptive").strip().lower()

    if normalized_vad_mode not in {"adaptive", "fixed"}:
        raise HTTPException(
            status_code=400,
            detail="vad_mode must be either 'adaptive' or 'fixed'.",
        )

    adaptive_vad_enabled = normalized_vad_mode == "adaptive"

    voice_activity = detect_speech_regions(
        audio=audio,
        sample_rate=sample_rate,
        top_db=vad_top_db,
        min_rms=vad_min_rms,
        min_region_duration_seconds=vad_min_region_duration_seconds,
        merge_gap_seconds=vad_merge_gap_seconds,
        adaptive_min_rms=adaptive_vad_enabled,
    )

    speech_regions = voice_activity.get("speech_regions", [])

    effective_vad_min_rms = float(
        voice_activity.get(
            "effective_min_rms",
            voice_activity.get("min_rms", vad_min_rms),
        )
    )

    if speech_regions:
        speech_only_audio = concatenate_speech_regions(
            audio,
            sample_rate,
            speech_regions,
        )
    else:
        speech_only_audio = np.array([], dtype=np.float32)

    if len(speech_only_audio) == 0:
        speech_only_audio = audio
        voice_activity["warning"] = (
            "No speech-only audio could be created, so global features were extracted from the full decoded audio."
        )

    processed_audio, preprocessing_metrics = remove_silence(
        speech_only_audio,
        sample_rate,
    )

    if len(processed_audio) == 0:
        processed_audio = speech_only_audio

    features = extract_basic_features(processed_audio, sample_rate)

    warnings = loading_warnings + quality_warnings

    speaker_status = speaker_segmentation_placeholder()
    speaker_features = {}
    diarization_evaluation = _empty_evaluation_status()

    predicted_segments_for_evaluation = []

    method = (segmentation_method or "none").strip().lower()
    manual_segments_text = _clean_optional_json_text(segments_json)
    reference_segments_text = _clean_optional_json_text(reference_segments_json)

    try:
        speaker_start_time = time.perf_counter()

        if manual_segments_text:
            manual_segments = json.loads(manual_segments_text)

            if not isinstance(manual_segments, list):
                raise ValueError("segments_json must be a list of speaker segments.")

            validated_segments = validate_manual_segments(
                manual_segments,
                audio_duration_seconds,
            )

            speaker_audio_segments = extract_speaker_audio_segments(
                audio,
                sample_rate,
                validated_segments,
            )

            speaker_status = compute_speaker_segmentation_summary(
                validated_segments,
                method_name="manual_speaker_labels",
                message=(
                    "Manual speaker labels were provided and processed. "
                    "No automatic speaker diarization method was used."
                ),
            )

            speaker_status["manual_segments_override"] = True
            speaker_status["requested_segmentation_method"] = method
            speaker_status["important_note"] = (
                "Because segments_json was provided, the API used manual timestamps. "
                "This result must not be reported as automatic ECAPA/WavLM diarization."
            )

            speaker_features = _extract_features_for_speakers(
                speaker_audio_segments,
                sample_rate,
            )

            predicted_segments_for_evaluation = validated_segments

        elif method == "auto":
            automatic_segments, speaker_status = automatic_vad_speaker_clustering(
                audio=audio,
                sample_rate=sample_rate,
                expected_speakers=expected_speakers,
                chunk_duration_seconds=chunk_duration_seconds,
                min_rms=effective_vad_min_rms,
                speech_regions=speech_regions,
            )

            speaker_audio_segments = extract_speaker_audio_segments(
                audio,
                sample_rate,
                automatic_segments,
            )

            speaker_features = _extract_features_for_speakers(
                speaker_audio_segments,
                sample_rate,
            )

            predicted_segments_for_evaluation = automatic_segments

        elif method == "ecapa":
            ecapa_segments, speaker_status = ecapa_speaker_embedding_clustering(
                audio=audio,
                sample_rate=sample_rate,
                expected_speakers=expected_speakers,
                chunk_duration_seconds=chunk_duration_seconds,
                min_rms=effective_vad_min_rms,
                speech_regions=speech_regions,
            )

            speaker_audio_segments = extract_speaker_audio_segments(
                audio,
                sample_rate,
                ecapa_segments,
            )

            speaker_features = _extract_features_for_speakers(
                speaker_audio_segments,
                sample_rate,
            )

            predicted_segments_for_evaluation = ecapa_segments

        elif method == "ecapa_v2":
            ecapa_v2_segments, speaker_status = ecapa_v2_speaker_embedding_clustering(
                audio=audio,
                sample_rate=sample_rate,
                expected_speakers=expected_speakers,
                chunk_duration_seconds=chunk_duration_seconds,
                chunk_hop_seconds=ecapa_chunk_hop_seconds,
                min_rms=effective_vad_min_rms,
                speech_regions=speech_regions,
                smoothing_passes=ecapa_smoothing_passes,
            )

            speaker_audio_segments = extract_speaker_audio_segments(
                audio,
                sample_rate,
                ecapa_v2_segments,
            )

            speaker_features = _extract_features_for_speakers(
                speaker_audio_segments,
                sample_rate,
            )

            predicted_segments_for_evaluation = ecapa_v2_segments

        elif method == "wavlm":
            wavlm_segments, speaker_status = wavlm_speaker_embedding_clustering(
                audio=audio,
                sample_rate=sample_rate,
                expected_speakers=expected_speakers,
                chunk_duration_seconds=chunk_duration_seconds,
                min_rms=effective_vad_min_rms,
                speech_regions=speech_regions,
            )

            speaker_audio_segments = extract_speaker_audio_segments(
                audio,
                sample_rate,
                wavlm_segments,
            )

            speaker_features = _extract_features_for_speakers(
                speaker_audio_segments,
                sample_rate,
            )

            predicted_segments_for_evaluation = wavlm_segments

        elif method not in ["none", ""]:
            raise ValueError(
                "Unknown segmentation_method. Use one of: none, auto, ecapa, ecapa_v2, wavlm."
            )

        speaker_processing_seconds = time.perf_counter() - speaker_start_time

        if reference_segments_text:
            evaluation_start_time = time.perf_counter()

            reference_segments = json.loads(reference_segments_text)

            if not isinstance(reference_segments, list):
                raise ValueError("reference_segments_json must be a list of speaker segments.")

            validated_reference_segments = validate_manual_segments(
                reference_segments,
                audio_duration_seconds,
            )

            if not predicted_segments_for_evaluation:
                diarization_evaluation = _empty_evaluation_status(
                    "reference_segments_json was provided, but no predicted speaker segments were available."
                )
            else:
                diarization_evaluation = evaluate_diarization(
                    reference_segments=validated_reference_segments,
                    predicted_segments=predicted_segments_for_evaluation,
                    audio_duration_seconds=audio_duration_seconds,
                    frame_step_seconds=evaluation_frame_step_seconds,
                )

            evaluation_processing_seconds = time.perf_counter() - evaluation_start_time

    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "segments_json or reference_segments_json is not valid JSON. "
                "Leave segments_json empty for automatic methods and use reference_segments_json only for scoring."
            ),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    total_processing_seconds = time.perf_counter() - request_start_time

    runtime_metrics = {
        "total_processing_seconds": round(total_processing_seconds, 4),
        "speaker_processing_seconds": round(speaker_processing_seconds, 4),
        "evaluation_processing_seconds": round(evaluation_processing_seconds, 4),
    }

    return {
        "filename": file.filename,
        "quality_metrics": quality_metrics,
        "voice_activity": voice_activity,
        "preprocessing_metrics": preprocessing_metrics,
        "speaker_segmentation": speaker_status,
        "diarization_evaluation": diarization_evaluation,
        "runtime_metrics": runtime_metrics,
        "features": features,
        "speaker_features": speaker_features,
        "warnings": warnings,
        "privacy_status": get_privacy_status(),
    }
