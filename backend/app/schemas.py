
from typing import Any, Dict, List

from pydantic import BaseModel


class PrivacyStatus(BaseModel):
    raw_audio_stored: bool
    temporary_files_deleted: bool
    output_contains_raw_audio: bool


class ExtractResponse(BaseModel):
    filename: str
    quality_metrics: Dict[str, Any]
    voice_activity: Dict[str, Any]
    preprocessing_metrics: Dict[str, Any]
    speaker_segmentation: Dict[str, Any]
    diarization_evaluation: Dict[str, Any]
    runtime_metrics: Dict[str, Any]
    features: Dict[str, Any]
    speaker_features: Dict[str, Any]
    warnings: List[str]
    privacy_status: PrivacyStatus
