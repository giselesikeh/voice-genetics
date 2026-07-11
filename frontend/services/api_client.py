import requests
from typing import Any, Dict, Optional


class VoiceGeneticsAPIError(Exception):
    """Custom exception for Voice Genetics API errors."""


def check_health(backend_url: str) -> Dict[str, Any]:
    """
    Check whether the FastAPI backend is running.
    """
    url = backend_url.rstrip("/") + "/health"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise VoiceGeneticsAPIError(
            f"Could not connect to backend at {url}. "
            "Make sure FastAPI is running on port 8000."
        ) from exc


def extract_audio_features(
    backend_url: str,
    uploaded_file,
    segmentation_method: str,
    expected_speakers: int,
    chunk_duration_seconds: float,
    vad_mode: str,
    vad_top_db: int,
    vad_min_rms: float,
    vad_min_region_duration_seconds: float,
    vad_merge_gap_seconds: float,
    segments_json: Optional[str] = None,
    reference_segments_json: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send an uploaded audio file to the FastAPI /extract endpoint.
    """
    url = backend_url.rstrip("/") + "/extract"

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }

    data = {
        "segmentation_method": segmentation_method,
        "expected_speakers": str(expected_speakers),
        "chunk_duration_seconds": str(chunk_duration_seconds),
        "vad_mode": vad_mode,
        "vad_top_db": str(vad_top_db),
        "vad_min_rms": str(vad_min_rms),
        "vad_min_region_duration_seconds": str(vad_min_region_duration_seconds),
        "vad_merge_gap_seconds": str(vad_merge_gap_seconds),
        "evaluation_frame_step_seconds": "0.1",
    }

    if segments_json and segments_json.strip():
        data["segments_json"] = segments_json.strip()

    if reference_segments_json and reference_segments_json.strip():
        data["reference_segments_json"] = reference_segments_json.strip()

    try:
        response = requests.post(url, files=files, data=data, timeout=600)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as exc:
        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text

        raise VoiceGeneticsAPIError(
            f"Backend returned an error: {error_detail}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise VoiceGeneticsAPIError(
            f"Could not send request to backend at {url}. "
            "Check that FastAPI is running and reachable."
        ) from exc
