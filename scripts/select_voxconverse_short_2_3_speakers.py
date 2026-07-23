from pathlib import Path
import csv
import shutil
import wave


PROJECT_ROOT = Path("/mnt/f/Innopolis 3 year/Industrial Project/Voice genetics/voice-genetics")

SOURCE_AUDIO_DIR = Path("/mnt/f/Innopolis 3 year/Industrial Project/Voice genetics/new/voxconverse_dev_wav/audio")
SOURCE_RTTM_DIR = Path("/mnt/f/Innopolis 3 year/Industrial Project/Voice genetics/new/labels/dev")

# Keep the same saved subset folder and add more files into it
OUTPUT_ROOT = PROJECT_ROOT / "data" / "samples" / "voxconverse_short_2_3_speakers"
OUTPUT_AUDIO_DIR = OUTPUT_ROOT / "audio"
OUTPUT_RTTM_DIR = OUTPUT_ROOT / "rttm"

OUTPUT_SUMMARY = OUTPUT_ROOT / "summary_10min.csv"
OUTPUT_NEW_ADDED = OUTPUT_ROOT / "newly_added_10min.csv"

MAX_DURATION_SECONDS = 10 * 60
ALLOWED_SPEAKER_COUNTS = {2, 3}


def read_rttm_speakers_and_duration(rttm_path: Path):
    speakers = set()
    total_speech_duration = 0.0
    latest_end = 0.0

    with rttm_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue

            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]

            speakers.add(speaker)
            total_speech_duration += duration
            latest_end = max(latest_end, start + duration)

    return speakers, total_speech_duration, latest_end


def get_wav_duration_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        return frames / float(rate)


def find_matching_audio(file_id: str) -> Path | None:
    candidates = [
        SOURCE_AUDIO_DIR / f"{file_id}.wav",
        SOURCE_AUDIO_DIR / file_id,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    matches = list(SOURCE_AUDIO_DIR.rglob(f"{file_id}*"))

    for match in matches:
        if match.is_file():
            return match

    return None


def write_csv(path: Path, rows: list[dict]):
    fieldnames = [
        "file_id",
        "speaker_count",
        "audio_duration_seconds",
        "reference_speech_duration_seconds",
        "latest_reference_end_seconds",
        "already_existed_before_run",
        "audio_path",
        "rttm_path",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_RTTM_DIR.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    newly_added_rows = []
    skipped_rows = []

    existing_audio_names_before = {p.name for p in OUTPUT_AUDIO_DIR.iterdir() if p.is_file()}

    rttm_files = sorted(SOURCE_RTTM_DIR.glob("*.rttm"))

    print(f"Found {len(rttm_files)} RTTM files.")
    print(f"Selecting samples with 2 or 3 speakers and duration <= {MAX_DURATION_SECONDS} seconds.")
    print(f"Output folder: {OUTPUT_ROOT}")

    for rttm_path in rttm_files:
        file_id = rttm_path.stem

        speakers, reference_speech_duration, latest_reference_end = read_rttm_speakers_and_duration(rttm_path)
        speaker_count = len(speakers)

        if speaker_count not in ALLOWED_SPEAKER_COUNTS:
            skipped_rows.append((file_id, speaker_count, "speaker_count_not_2_or_3"))
            continue

        audio_path = find_matching_audio(file_id)

        if audio_path is None:
            skipped_rows.append((file_id, speaker_count, "matching_audio_not_found"))
            continue

        try:
            audio_duration = get_wav_duration_seconds(audio_path)
        except Exception as e:
            skipped_rows.append((file_id, speaker_count, f"duration_error: {e}"))
            continue

        if audio_duration > MAX_DURATION_SECONDS:
            skipped_rows.append((file_id, speaker_count, f"too_long_{audio_duration:.2f}s"))
            continue

        output_audio_path = OUTPUT_AUDIO_DIR / audio_path.name
        output_rttm_path = OUTPUT_RTTM_DIR / rttm_path.name

        already_existed = audio_path.name in existing_audio_names_before

        shutil.copy2(audio_path, output_audio_path)
        shutil.copy2(rttm_path, output_rttm_path)

        row = {
            "file_id": file_id,
            "speaker_count": speaker_count,
            "audio_duration_seconds": round(audio_duration, 3),
            "reference_speech_duration_seconds": round(reference_speech_duration, 3),
            "latest_reference_end_seconds": round(latest_reference_end, 3),
            "already_existed_before_run": already_existed,
            "audio_path": str(output_audio_path),
            "rttm_path": str(output_rttm_path),
        }

        selected_rows.append(row)

        if not already_existed:
            newly_added_rows.append(row)
            print(f"NEW: {file_id} | speakers={speaker_count} | duration={audio_duration:.2f}s")
        else:
            print(f"EXISTS: {file_id} | speakers={speaker_count} | duration={audio_duration:.2f}s")

    write_csv(OUTPUT_SUMMARY, selected_rows)
    write_csv(OUTPUT_NEW_ADDED, newly_added_rows)

    print("\nDone.")
    print(f"Total selected under 10 minutes: {len(selected_rows)}")
    print(f"Newly added this run: {len(newly_added_rows)}")
    print(f"Skipped files: {len(skipped_rows)}")
    print(f"Audio folder: {OUTPUT_AUDIO_DIR}")
    print(f"RTTM folder: {OUTPUT_RTTM_DIR}")
    print(f"Full summary: {OUTPUT_SUMMARY}")
    print(f"Newly added summary: {OUTPUT_NEW_ADDED}")


if __name__ == "__main__":
    main()