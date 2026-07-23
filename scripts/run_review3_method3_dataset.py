import csv
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_URL = "http://127.0.0.1:8000/extract"
OUTPUT_DIR = PROJECT_ROOT / "results" / "review3_method3_dataset"

SELECTED_SAMPLES = {
    "afjiv": 5,
    "akthc": 2,
    "ampme": 3,
    "asxwr": 3,
    "aufkn": 3,
}


def safe_get(data: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def find_audio_file(file_id: str) -> Optional[Path]:
    candidates = []
    for ext in [".wav", ".m4a", ".mp3"]:
        candidates.extend(PROJECT_ROOT.rglob(f"{file_id}{ext}"))
    candidates = [
        path for path in candidates
        if ".venv" not in str(path) and "__pycache__" not in str(path)
    ]
    return candidates[0] if candidates else None


def find_rttm_file(file_id: str) -> Optional[Path]:
    candidates = list(PROJECT_ROOT.rglob(f"{file_id}.rttm"))
    candidates = [
        path for path in candidates
        if ".venv" not in str(path) and "__pycache__" not in str(path)
    ]
    return candidates[0] if candidates else None


def rttm_to_reference_segments(rttm_path: Path) -> List[Dict[str, Any]]:
    segments = []

    with rttm_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            # RTTM format:
            # SPEAKER file channel start duration ... speaker_id ...
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            end = start + duration

            segments.append(
                {
                    "speaker": str(speaker),
                    "start": round(start, 3),
                    "end": round(end, 3),
                }
            )

    return segments


def run_sample(file_id: str, expected_speakers: int) -> Dict[str, Any]:
    audio_path = find_audio_file(file_id)
    if audio_path is None:
        raise FileNotFoundError(
            f"Could not find audio file for {file_id}. Expected something like {file_id}.wav"
        )

    rttm_path = find_rttm_file(file_id)
    reference_segments_json = ""

    if rttm_path is not None:
        reference_segments = rttm_to_reference_segments(rttm_path)
        reference_segments_json = json.dumps(reference_segments)

    print(f"\n=== Running Method 3 ECAPA on {file_id} ===")
    print(f"Audio: {audio_path}")
    print(f"Expected speakers: {expected_speakers}")
    print(f"RTTM reference: {rttm_path if rttm_path else 'not found'}")

    data = {
        "segmentation_method": "ecapa",
        "expected_speakers": str(expected_speakers),
        "chunk_duration_seconds": "2.0",
        "vad_mode": "adaptive",
        "vad_top_db": "30",
        "vad_min_rms": "0.015",
        "vad_min_region_duration_seconds": "0.25",
        "vad_merge_gap_seconds": "0.8",
        "ecapa_chunk_hop_seconds": "1.0",
        "ecapa_smoothing_passes": "1",
        "evaluation_frame_step_seconds": "0.1",
    }

    if reference_segments_json:
        data["reference_segments_json"] = reference_segments_json

    with audio_path.open("rb") as audio_file:
        files = {
            "file": (
                audio_path.name,
                audio_file,
                "audio/wav",
            )
        }

        response = requests.post(
            BACKEND_URL,
            files=files,
            data=data,
            timeout=1800,
        )

    if response.status_code != 200:
        print(response.text)
        response.raise_for_status()

    result = response.json()

    output_path = OUTPUT_DIR / f"{file_id}_method3_ecapa.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Saved: {output_path}")

    return result


def build_summary_row(file_id: str, expected_speakers: int, result: Dict[str, Any]) -> Dict[str, Any]:
    clustering = safe_get(result, ["speaker_segmentation", "clustering_metrics"], {}) or {}
    quality = result.get("quality_metrics", {}) or {}
    voice_activity = result.get("voice_activity", {}) or {}
    features = result.get("features", {}) or {}
    pitch = features.get("pitch", {}) or {}
    spectral = features.get("spectral_centroid", {}) or {}

    diar_eval = (
        result.get("diarization_evaluation")
        or result.get("evaluation")
        or result.get("diarization_metrics")
        or {}
    )

    return {
        "file_id": file_id,
        "expected_speakers": expected_speakers,
        "filename": result.get("filename"),
        "duration_seconds": quality.get("duration_seconds"),
        "sample_rate": quality.get("sample_rate"),
        "rms_energy": quality.get("rms_energy"),
        "speech_coverage_ratio": voice_activity.get("speech_coverage_ratio"),
        "method": safe_get(result, ["speaker_segmentation", "method"]),
        "detected_speakers": safe_get(result, ["speaker_segmentation", "detected_speakers"]),
        "speaker_durations": json.dumps(
            safe_get(result, ["speaker_segmentation", "speaker_speech_duration_seconds"], {}),
            ensure_ascii=False,
        ),
        "speaker_segment_counts": json.dumps(
            safe_get(result, ["speaker_segmentation", "speaker_segment_count"], {}),
            ensure_ascii=False,
        ),
        "silhouette_score": clustering.get("silhouette_score"),
        "cluster_balance_ratio": clustering.get("cluster_balance_ratio"),
        "speaker_switches": clustering.get("speaker_switches"),
        "segment_smoothness": clustering.get("segment_smoothness"),
        "mean_segment_duration_seconds": clustering.get("mean_segment_duration_seconds"),
        "number_of_chunks_clustered": clustering.get("number_of_chunks_clustered"),
        "pitch_mean_hz": pitch.get("mean_hz"),
        "pitch_min_hz": pitch.get("min_hz"),
        "pitch_max_hz": pitch.get("max_hz"),
        "spectral_centroid_mean": spectral.get("mean"),
        "spectral_centroid_std": spectral.get("std"),
        "der_percentage": (
            diar_eval.get("der_percentage")
            or diar_eval.get("DER_percentage")
            or diar_eval.get("diarization_error_rate_percentage")
        ),
        "missed_speech_seconds": diar_eval.get("missed_speech_seconds"),
        "false_alarm_seconds": diar_eval.get("false_alarm_seconds"),
        "speaker_confusion_seconds": diar_eval.get("speaker_confusion_seconds"),
        "total_processing_seconds": safe_get(result, ["runtime_metrics", "total_processing_seconds"]),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check backend health first
    health_url = BACKEND_URL.replace("/extract", "/health")
    print(f"Checking backend: {health_url}")
    health = requests.get(health_url, timeout=10)
    health.raise_for_status()
    print("Backend health:", health.json())

    summary_rows = []

    for file_id, expected_speakers in SELECTED_SAMPLES.items():
        result = run_sample(file_id, expected_speakers)
        summary_rows.append(build_summary_row(file_id, expected_speakers, result))

    summary_path = OUTPUT_DIR / "summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved summary CSV: {summary_path}")

    zip_path = PROJECT_ROOT / "results" / "review3_method3_dataset.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path in OUTPUT_DIR.rglob("*"):
            zipf.write(file_path, file_path.relative_to(PROJECT_ROOT))

    print(f"Saved ZIP: {zip_path}")

    print("\nDone. Send me this ZIP or the JSON/CSV files:")
    print(zip_path)


if __name__ == "__main__":
    main()
