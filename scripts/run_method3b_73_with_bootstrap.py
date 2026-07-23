from __future__ import annotations

import csv
import json
import random
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8000/extract"

SUBSET_ROOT = ROOT / "data" / "samples" / "voxconverse_short_2_3_speakers"
SUMMARY_CSV = SUBSET_ROOT / "summary_10min.csv"
AUDIO_DIR = SUBSET_ROOT / "audio"
RTTM_DIR = SUBSET_ROOT / "rttm"

OUT = ROOT / "results" / "method3b_73_bootstrap"
JSON_OUT = OUT / "json"
PER_FILE_CSV = OUT / "per_file_metrics.csv"
BOOTSTRAP_CSV = OUT / "bootstrap_summary.csv"
BOOTSTRAP_DRAWS_CSV = OUT / "bootstrap_draws.csv"

BOOTSTRAP_ROUNDS = 1000
RANDOM_SEED = 42


def read_subset_rows() -> list[dict[str, str]]:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing subset summary CSV: {SUMMARY_CSV}")

    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cleaned = []
    for row in rows:
        file_id = row["file_id"]
        speaker_count = int(row["speaker_count"])
        duration = float(row["audio_duration_seconds"])

        if speaker_count in {2, 3} and duration <= 600:
            cleaned.append(row)

    return cleaned


def find_audio(file_id: str) -> Path:
    candidates = [
        AUDIO_DIR / f"{file_id}.wav",
        AUDIO_DIR / file_id,
        AUDIO_DIR / f"{file_id}.mp3",
        AUDIO_DIR / f"{file_id}.m4a",
    ]

    for p in candidates:
        if p.exists():
            return p

    hits = list(AUDIO_DIR.rglob(f"{file_id}*"))
    hits = [p for p in hits if p.is_file()]

    if not hits:
        raise FileNotFoundError(f"No audio found for {file_id}")

    return hits[0]


def find_rttm(file_id: str) -> Path:
    candidates = [
        RTTM_DIR / f"{file_id}.rttm",
        RTTM_DIR / file_id,
    ]

    for p in candidates:
        if p.exists():
            return p

    hits = list(RTTM_DIR.rglob(f"{file_id}*"))
    hits = [p for p in hits if p.is_file()]

    if not hits:
        raise FileNotFoundError(f"No RTTM found for {file_id}")

    return hits[0]


def rttm_to_reference_json(path: Path) -> str:
    segments = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue

        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        end = start + duration

        segments.append(
            {
                "speaker": speaker,
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )

    return json.dumps(segments)


def get(d: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = d

    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]

    return cur


def run_one(file_id: str, expected_speakers: int) -> dict[str, Any]:
    JSON_OUT.mkdir(parents=True, exist_ok=True)

    out_json = JSON_OUT / f"{file_id}_method3b.json"

    if out_json.exists():
        print(f"SKIP existing JSON: {file_id}")
        return json.loads(out_json.read_text(encoding="utf-8"))

    audio = find_audio(file_id)
    rttm = find_rttm(file_id)

    print(f"\n=== Running Method 3B: {file_id} | speakers={expected_speakers} ===")
    print("Audio:", audio)
    print("RTTM:", rttm)

    data = {
        "segmentation_method": "ecapa_v2",
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
        "auto_detect_speakers": "false",
        "min_speakers": "2",
        "max_speakers": "4",
        "reference_segments_json": rttm_to_reference_json(rttm),
    }

    with audio.open("rb") as f:
        response = requests.post(
            URL,
            files={"file": (audio.name, f, "audio/wav")},
            data=data,
            timeout=3600,
        )

    if response.status_code != 200:
        print("ERROR:", response.text)
        response.raise_for_status()

    result = response.json()
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved:", out_json)

    return result


def result_to_row(file_id: str, expected_speakers: int, result: dict[str, Any]) -> dict[str, Any]:
    quality = result.get("quality_metrics", {})
    vad = result.get("voice_activity", {})
    seg = result.get("speaker_segmentation", {})
    cm = get(result, ["speaker_segmentation", "clustering_metrics"], {}) or {}
    ev = result.get("diarization_evaluation", {}) or {}
    rates = ev.get("rate_metrics", {}) or {}
    times = ev.get("time_metrics_seconds", {}) or {}
    features = result.get("features", {}) or {}
    pitch = features.get("pitch", {}) or {}
    centroid = features.get("spectral_centroid", {}) or {}
    adv = features.get("advanced_acoustic_features", {}) or {}

    formants = adv.get("formants", {}) or {}
    f1 = get(formants, ["F1_hz", "median"])
    f2 = get(formants, ["F2_hz", "median"])
    f3 = get(formants, ["F3_hz", "median"])

    jitter_shimmer = adv.get("jitter_shimmer", {}) or {}
    hnr = adv.get("harmonic_to_noise_ratio", {}) or {}
    speaking_rate = adv.get("speaking_rate", {}) or {}
    vot = adv.get("voice_onset_time", {}) or {}

    return {
        "file_id": file_id,
        "expected_speakers": expected_speakers,
        "duration_seconds": quality.get("duration_seconds"),
        "global_rms": quality.get("rms_energy"),
        "speech_coverage": vad.get("speech_coverage_ratio"),
        "detected_speakers": seg.get("detected_speakers"),

        "DER_percent": rates.get("der_percent"),
        "speech_precision": rates.get("speech_precision"),
        "speech_recall": rates.get("speech_recall"),
        "speaker_accuracy_overlap": rates.get("speaker_assignment_accuracy_on_overlap"),
        "speaker_accuracy_reference": rates.get("speaker_assignment_accuracy_on_reference"),
        "pause_detection_accuracy": rates.get("pause_detection_accuracy"),

        "missed_speech_seconds": times.get("missed_speech"),
        "false_alarm_seconds": times.get("false_alarm"),
        "speaker_confusion_seconds": times.get("speaker_confusion"),
        "diarization_error_seconds": times.get("diarization_error"),

        "silhouette_score": cm.get("silhouette_score"),
        "cluster_balance_ratio": cm.get("cluster_balance_ratio"),
        "speaker_switches": cm.get("speaker_switches"),
        "segment_smoothness": cm.get("segment_smoothness"),
        "mean_segment_duration_seconds": cm.get("mean_segment_duration_seconds"),

        "pitch_mean_hz": pitch.get("mean_hz"),
        "pitch_min_hz": pitch.get("min_hz"),
        "pitch_max_hz": pitch.get("max_hz"),
        "spectral_centroid_mean": centroid.get("mean"),
        "spectral_centroid_std": centroid.get("std"),

        "F1_median_hz": f1,
        "F2_median_hz": f2,
        "F3_median_hz": f3,
        "jitter_local_percent": jitter_shimmer.get("jitter_local_percent"),
        "shimmer_local_percent": jitter_shimmer.get("shimmer_local_percent"),
        "HNR_db": hnr.get("hnr_db"),
        "speaking_rate_proxy_per_minute": speaking_rate.get("estimated_syllable_rate_per_minute"),
        "voice_onset_time_proxy_median_ms": get(vot, ["estimated_vot_ms", "median"]),

        "runtime_seconds": get(result, ["runtime_metrics", "total_processing_seconds"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_columns(rows: list[dict[str, Any]]) -> list[str]:
    cols = []

    for key in rows[0].keys():
        if key == "file_id":
            continue

        values = []
        for row in rows:
            value = row.get(key)
            try:
                if value is not None and value != "":
                    values.append(float(value))
            except Exception:
                pass

        if values:
            cols.append(key)

    return cols


def bootstrap_summary(rows: list[dict[str, Any]], label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(RANDOM_SEED)
    cols = numeric_columns(rows)
    n = len(rows)

    draw_rows = []

    for b in range(BOOTSTRAP_ROUNDS):
        sampled = [rows[rng.randrange(n)] for _ in range(n)]

        draw = {
            "group": label,
            "bootstrap_round": b + 1,
            "n_files": n,
        }

        for col in cols:
            values = []
            for row in sampled:
                try:
                    value = row.get(col)
                    if value is not None and value != "":
                        values.append(float(value))
                except Exception:
                    pass

            draw[f"mean_{col}"] = float(np.mean(values)) if values else None

        draw_rows.append(draw)

    summary_rows = []

    for col in cols:
        boot_values = np.array(
            [
                d.get(f"mean_{col}")
                for d in draw_rows
                if d.get(f"mean_{col}") is not None
            ],
            dtype=float,
        )

        if boot_values.size == 0:
            continue

        original_values = np.array(
            [
                float(row[col])
                for row in rows
                if row.get(col) is not None and row.get(col) != ""
            ],
            dtype=float,
        )

        summary_rows.append(
            {
                "group": label,
                "metric": col,
                "n_files": n,
                "original_mean_over_files": round(float(np.mean(original_values)), 6),
                "bootstrap_mean_of_averages": round(float(np.mean(boot_values)), 6),
                "bootstrap_std_of_averages": round(float(np.std(boot_values, ddof=1)), 6),
                "bootstrap_ci_2_5": round(float(np.percentile(boot_values, 2.5)), 6),
                "bootstrap_ci_97_5": round(float(np.percentile(boot_values, 97.5)), 6),
            }
        )

    return summary_rows, draw_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.mkdir(parents=True, exist_ok=True)

    print("Checking backend...")
    requests.get("http://127.0.0.1:8000/health", timeout=10).raise_for_status()

    subset = read_subset_rows()
    print(f"Selected files to run: {len(subset)}")

    rows = []

    for item in subset:
        file_id = item["file_id"]
        expected_speakers = int(item["speaker_count"])

        result = run_one(file_id, expected_speakers)
        rows.append(result_to_row(file_id, expected_speakers, result))

        write_csv(PER_FILE_CSV, rows)

    print("\nWriting final per-file CSV...")
    write_csv(PER_FILE_CSV, rows)

    print("Bootstrapping...")
    all_summary, all_draws = bootstrap_summary(rows, "all")

    rows_2 = [r for r in rows if int(r["expected_speakers"]) == 2]
    rows_3 = [r for r in rows if int(r["expected_speakers"]) == 3]

    if len(rows_2) >= 2:
        s2, d2 = bootstrap_summary(rows_2, "2_speakers")
        all_summary.extend(s2)
        all_draws.extend(d2)

    if len(rows_3) >= 2:
        s3, d3 = bootstrap_summary(rows_3, "3_speakers")
        all_summary.extend(s3)
        all_draws.extend(d3)

    write_csv(BOOTSTRAP_CSV, all_summary)
    write_csv(BOOTSTRAP_DRAWS_CSV, all_draws)

    zip_path = OUT.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob("*"):
            z.write(p, p.relative_to(ROOT))

    print("\nDONE")
    print("Per-file metrics:", PER_FILE_CSV)
    print("Bootstrap summary:", BOOTSTRAP_CSV)
    print("Bootstrap draws:", BOOTSTRAP_DRAWS_CSV)
    print("ZIP:", zip_path)


if __name__ == "__main__":
    main()