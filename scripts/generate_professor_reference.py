from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from pyannote.audio import Pipeline


# ============================================================
# SETTINGS
# ============================================================

AUDIO_PATH = Path("data/samples/professor_test.mp3")
OUTPUT_DIR = Path("results/professor_reference")

MODEL_ID = "pyannote/speaker-diarization-community-1"

# Your previous Method 3B run used two expected speakers.
# Keep 2 when you know there are exactly two speakers.
# Change to None when the speaker count is unknown.
KNOWN_SPEAKERS: int | None = 2

TARGET_SAMPLE_RATE = 16_000

# Gaps shorter than this are not reported as non-speech.
MIN_GAP_SECONDS = 0.25

# Silence/music/noise classification window.
WINDOW_SECONDS = 1.0

# Audio below this dBFS value is treated as silence.
SILENCE_DBFS = -45.0


# ============================================================
# SPEAKER DIARIZATION
# ============================================================

def load_pipeline(token: str) -> Pipeline:
    """Load the independent pyannote diarization pipeline."""

    try:
        pipeline = Pipeline.from_pretrained(
            MODEL_ID,
            token=token,
        )
    except TypeError:
        # Compatibility with older pyannote versions.
        pipeline = Pipeline.from_pretrained(
            MODEL_ID,
            use_auth_token=token,
        )

    if pipeline is None:
        raise RuntimeError(
            "Could not load pyannote. Check your HF_TOKEN and "
            "confirm that you accepted the model conditions."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    pipeline.to(device)

    print(f"pyannote device: {device}")
    return pipeline


def run_diarization(
    pipeline: Pipeline,
    audio: np.ndarray,
    sample_rate: int,
) -> Any:
    """Run diarization from the in-memory mono waveform."""

    waveform = torch.from_numpy(audio).unsqueeze(0).float()

    audio_input = {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }

    if KNOWN_SPEAKERS is None:
        result = pipeline(
            audio_input,
            min_speakers=1,
            max_speakers=8,
        )
    else:
        result = pipeline(
            audio_input,
            num_speakers=KNOWN_SPEAKERS,
        )

    # Newer pipelines may wrap the annotation in an output object.
    annotation = getattr(
        result,
        "speaker_diarization",
        result,
    )

    if not hasattr(annotation, "itertracks"):
        raise RuntimeError(
            "Unexpected pyannote output: no diarization "
            "annotation was returned."
        )

    return annotation


def extract_speaker_segments(
    annotation: Any,
) -> list[dict[str, Any]]:
    """Convert pyannote labels to speaker_1, speaker_2, etc."""

    turns: list[tuple[float, float, str]] = []

    for turn, _, raw_label in annotation.itertracks(
        yield_label=True
    ):
        start = float(turn.start)
        end = float(turn.end)

        if end > start:
            turns.append(
                (start, end, str(raw_label))
            )

    turns.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        )
    )

    label_map: dict[str, str] = {}
    segments: list[dict[str, Any]] = []

    for start, end, raw_label in turns:
        if raw_label not in label_map:
            label_map[raw_label] = (
                f"speaker_{len(label_map) + 1}"
            )

        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "type": "speech",
                "speaker": label_map[raw_label],
                "raw_pyannote_label": raw_label,
            }
        )

    return segments


# ============================================================
# FIND NON-SPEECH REGIONS
# ============================================================

def merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping speaker intervals."""

    merged: list[list[float]] = []

    for start, end in sorted(intervals):
        if end <= start:
            continue

        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(
                merged[-1][1],
                end,
            )

    return [
        (start, end)
        for start, end in merged
    ]


def find_non_speech_gaps(
    speaker_segments: list[dict[str, Any]],
    total_duration: float,
) -> list[tuple[float, float]]:
    """Find regions outside all diarized speaker turns."""

    occupied = merge_intervals(
        [
            (
                float(segment["start"]),
                float(segment["end"]),
            )
            for segment in speaker_segments
        ]
    )

    gaps: list[tuple[float, float]] = []
    cursor = 0.0

    for start, end in occupied:
        if start - cursor >= MIN_GAP_SECONDS:
            gaps.append((cursor, start))

        cursor = max(cursor, end)

    if total_duration - cursor >= MIN_GAP_SECONDS:
        gaps.append(
            (cursor, total_duration)
        )

    return gaps


# ============================================================
# SILENCE / MUSIC / NOISE CLASSIFICATION
# ============================================================

def classify_non_speech(
    chunk: np.ndarray,
    sample_rate: int,
) -> tuple[str, float]:
    """Classify a region already considered non-speech."""

    if chunk.size == 0:
        return "silence", 1.0

    if chunk.size < 2048:
        chunk = np.pad(
            chunk,
            (0, 2048 - chunk.size),
        )

    rms = float(
        np.sqrt(
            np.mean(np.square(chunk))
            + 1e-12
        )
    )

    dbfs = float(
        20.0
        * math.log10(
            max(rms, 1e-12)
        )
    )

    if dbfs <= SILENCE_DBFS:
        return "silence", 0.95

    flatness = float(
        np.mean(
            librosa.feature.spectral_flatness(
                y=chunk
            )
        )
    )

    zero_crossing_rate = float(
        np.mean(
            librosa.feature.zero_crossing_rate(
                y=chunk
            )
        )
    )

    harmonic = librosa.effects.harmonic(
        chunk
    )

    harmonic_ratio = float(
        np.sum(np.square(harmonic))
        / (
            np.sum(np.square(chunk))
            + 1e-12
        )
    )

    try:
        chroma = librosa.feature.chroma_stft(
            y=harmonic,
            sr=sample_rate,
        )

        chroma_strength = (
            float(
                np.mean(
                    np.max(chroma, axis=0)
                )
            )
            if chroma.size
            else 0.0
        )

    except Exception:
        chroma_strength = 0.0

    music_score = (
        0.50
        * np.clip(
            (harmonic_ratio - 0.40) / 0.50,
            0.0,
            1.0,
        )
        + 0.25
        * np.clip(
            (0.20 - flatness) / 0.20,
            0.0,
            1.0,
        )
        + 0.25
        * np.clip(
            (chroma_strength - 0.25) / 0.55,
            0.0,
            1.0,
        )
    )

    noise_score = (
        0.60
        * np.clip(
            (flatness - 0.08) / 0.42,
            0.0,
            1.0,
        )
        + 0.40
        * np.clip(
            (zero_crossing_rate - 0.04) / 0.30,
            0.0,
            1.0,
        )
    )

    if (
        music_score >= 0.58
        and music_score > noise_score
    ):
        return (
            "music",
            round(float(music_score), 4),
        )

    return (
        "noise",
        round(
            max(0.35, float(noise_score)),
            4,
        ),
    )


def detect_non_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    speaker_segments: list[dict[str, Any]],
    total_duration: float,
) -> list[dict[str, Any]]:
    """Classify gaps as silence, music, or noise."""

    gaps = find_non_speech_gaps(
        speaker_segments,
        total_duration,
    )

    results: list[dict[str, Any]] = []

    for gap_start, gap_end in gaps:
        start = gap_start

        while start < gap_end:
            end = min(
                gap_end,
                start + WINDOW_SECONDS,
            )

            first_sample = int(
                round(start * sample_rate)
            )
            last_sample = int(
                round(end * sample_rate)
            )

            chunk = audio[
                first_sample:last_sample
            ]

            label, confidence = classify_non_speech(
                chunk,
                sample_rate,
            )

            current = {
                "start": round(start, 3),
                "end": round(end, 3),
                "type": label,
                "confidence": confidence,
            }

            # Merge adjacent windows with the same label.
            if (
                results
                and results[-1]["type"] == label
                and abs(
                    float(results[-1]["end"])
                    - start
                )
                <= 0.05
            ):
                results[-1]["end"] = round(
                    end,
                    3,
                )
            else:
                results.append(current)

            start = end

    return results


# ============================================================
# OUTPUT FILES
# ============================================================

def write_rttm(
    path: Path,
    audio_uri: str,
    speaker_segments: list[dict[str, Any]],
) -> None:
    """Write standard speaker diarization RTTM."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for segment in speaker_segments:
            start = float(
                segment["start"]
            )
            end = float(
                segment["end"]
            )
            duration = end - start

            if duration <= 0:
                continue

            file.write(
                f"SPEAKER {audio_uri} 1 "
                f"{start:.3f} "
                f"{duration:.3f} "
                f"<NA> <NA> "
                f"{segment['speaker']} "
                f"<NA> <NA>\n"
            )


def write_review_csv(
    path: Path,
    timeline: list[dict[str, Any]],
) -> None:
    """Create a CSV for manual listening and correction."""

    columns = [
        "start",
        "end",
        "type",
        "speaker",
        "confidence",
        "corrected_label",
        "corrected_start",
        "corrected_end",
        "notes",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        writer.writeheader()

        for segment in timeline:
            writer.writerow(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "type": segment["type"],
                    "speaker": segment.get(
                        "speaker",
                        "",
                    ),
                    "confidence": segment.get(
                        "confidence",
                        "",
                    ),
                    "corrected_label": "",
                    "corrected_start": "",
                    "corrected_end": "",
                    "notes": "",
                }
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio not found: "
            f"{AUDIO_PATH.resolve()}"
        )

    token = os.getenv(
        "HF_TOKEN",
        "",
    ).strip()

    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Run:\n"
            'export HF_TOKEN="hf_your_token"'
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading {AUDIO_PATH}...")

    # Convert the professor audio to mono 16 kHz in memory.
    audio, sample_rate = librosa.load(
        AUDIO_PATH,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
    )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    total_duration = (
        len(audio) / sample_rate
    )

    print(
        f"Duration: "
        f"{total_duration:.3f} seconds"
    )
    print(
        f"Processing format: "
        f"{sample_rate} Hz mono"
    )

    pipeline = load_pipeline(token)

    print(
        "Running independent "
        "speaker diarization..."
    )

    annotation = run_diarization(
        pipeline,
        audio,
        sample_rate,
    )

    speaker_segments = (
        extract_speaker_segments(
            annotation
        )
    )

    print(
        "Detecting silence, "
        "music, and noise..."
    )

    non_speech_segments = (
        detect_non_speech_segments(
            audio,
            sample_rate,
            speaker_segments,
            total_duration,
        )
    )

    timeline = sorted(
        speaker_segments
        + non_speech_segments,
        key=lambda segment: (
            float(segment["start"]),
            (
                0
                if segment["type"] == "speech"
                else 1
            ),
            float(segment["end"]),
        ),
    )

    reference_segments = [
        {
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
        }
        for segment in speaker_segments
    ]

    audio_uri = AUDIO_PATH.stem

    rttm_path = (
        OUTPUT_DIR
        / f"{audio_uri}.rttm"
    )

    reference_path = (
        OUTPUT_DIR
        / "reference_segments.json"
    )

    timeline_path = (
        OUTPUT_DIR
        / "full_timeline.json"
    )

    review_path = (
        OUTPUT_DIR
        / "review_timeline.csv"
    )

    write_rttm(
        rttm_path,
        audio_uri,
        speaker_segments,
    )

    reference_path.write_text(
        json.dumps(
            reference_segments,
            indent=2,
        ),
        encoding="utf-8",
    )

    full_output = {
        "audio": {
            "filename": AUDIO_PATH.name,
            "duration_seconds": round(
                total_duration,
                3,
            ),
            "sample_rate": sample_rate,
        },
        "annotation_status": {
            "is_ground_truth": False,
            "is_pseudo_reference": True,
            "requires_manual_review": True,
        },
        "methods": {
            "speaker_diarization": MODEL_ID,
            "non_speech_classification": (
                "spectral heuristic applied only "
                "outside diarized speech"
            ),
        },
        "speaker_segments": speaker_segments,
        "non_speech_segments": non_speech_segments,
        "timeline": timeline,
        "reference_segments_json": (
            reference_segments
        ),
        "limitations": [
            (
                "This is an automatically generated "
                "pseudo-reference."
            ),
            (
                "Music and noise labels must be "
                "confirmed by listening."
            ),
            (
                "Music or noise underneath speech "
                "may not be detected."
            ),
            (
                "Speaker labels are anonymous and "
                "do not identify people."
            ),
        ],
    }

    timeline_path.write_text(
        json.dumps(
            full_output,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_review_csv(
        review_path,
        timeline,
    )

    speakers = sorted(
        {
            segment["speaker"]
            for segment in speaker_segments
        }
    )

    print("\nFinished.")
    print(
        f"Detected speakers: "
        f"{len(speakers)}"
    )
    print(
        f"RTTM:               "
        f"{rttm_path}"
    )
    print(
        f"Reference JSON:     "
        f"{reference_path}"
    )
    print(
        f"Full timeline JSON: "
        f"{timeline_path}"
    )
    print(
        f"Review CSV:         "
        f"{review_path}"
    )

    print(
        "\nImportant: verify these timestamps "
        "by listening before DER evaluation."
    )


if __name__ == "__main__":
    main()