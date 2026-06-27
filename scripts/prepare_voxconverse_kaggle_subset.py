import json
import shutil
import wave
from pathlib import Path


KAGGLE_ROOT = Path("/mnt/f/Innopolis 3 year/Industrial Project/Voice genetics/new")

OUT_ROOT = Path("data/voxconverse_kaggle")
OUT_AUDIO = OUT_ROOT / "audio"
OUT_REFERENCE = OUT_ROOT / "reference_json"
OUT_LIST = OUT_ROOT / "eval_subset.txt"

MAX_SPEAKERS = 8
MAX_DURATION_SECONDS = 240
MIN_DURATION_SECONDS = 30
NUM_FILES = 5


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def parse_rttm(rttm_path: Path, audio_duration: float):
    segments = []

    for line in rttm_path.read_text(errors="ignore").splitlines():
        parts = line.strip().split()

        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue

        start = float(parts[3])
        duration = float(parts[4])
        end = start + duration
        speaker = parts[7]

        if start >= audio_duration:
            continue

        end = min(end, audio_duration)

        if end <= start:
            continue

        segments.append({
            "speaker": speaker,
            "start": round(start, 3),
            "end": round(end, 3),
        })

    segments.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    return segments


def find_matching_wav(file_id: str):
    candidates = list(KAGGLE_ROOT.rglob(f"{file_id}.wav"))

    if not candidates:
        # Sometimes Windows hides extensions, but WSL still sees them.
        candidates = [
            path for path in KAGGLE_ROOT.rglob("*")
            if path.is_file() and path.stem == file_id and path.suffix.lower() == ".wav"
        ]

    if not candidates:
        return None

    return candidates[0]


def main():
    OUT_AUDIO.mkdir(parents=True, exist_ok=True)
    OUT_REFERENCE.mkdir(parents=True, exist_ok=True)

    selected = []

    rttm_files = sorted(KAGGLE_ROOT.rglob("*.rttm"))

    print(f"Found RTTM files: {len(rttm_files)}")

    for rttm_path in rttm_files:
        file_id = rttm_path.stem
        wav_path = find_matching_wav(file_id)

        if wav_path is None:
            print(f"SKIP {file_id}: no matching WAV")
            continue

        try:
            duration = wav_duration_seconds(wav_path)
        except Exception as exc:
            print(f"SKIP {file_id}: cannot read wav duration: {exc}")
            continue

        if duration < MIN_DURATION_SECONDS:
            print(f"SKIP {file_id}: duration {duration:.1f}s < {MIN_DURATION_SECONDS}s")
            continue

        if duration > MAX_DURATION_SECONDS:
            print(f"SKIP {file_id}: duration {duration:.1f}s > {MAX_DURATION_SECONDS}s")
            continue

        segments = parse_rttm(rttm_path, duration)
        speakers = sorted(set(segment["speaker"] for segment in segments))

        if len(speakers) < 2:
            print(f"SKIP {file_id}: only {len(speakers)} speaker")
            continue

        if len(speakers) > MAX_SPEAKERS:
            print(f"SKIP {file_id}: {len(speakers)} speakers > {MAX_SPEAKERS}")
            continue

        out_wav = OUT_AUDIO / f"{file_id}.wav"
        out_json = OUT_REFERENCE / f"{file_id}.json"

        shutil.copy2(wav_path, out_wav)
        out_json.write_text(json.dumps(segments, indent=2), encoding="utf-8")

        selected.append(file_id)

        print(
            f"SELECT {file_id}: "
            f"duration={duration:.1f}s, speakers={len(speakers)}, segments={len(segments)}"
        )

        if len(selected) >= NUM_FILES:
            break

    OUT_LIST.write_text("\n".join(selected) + "\n", encoding="utf-8")

    print("\nSelected files:")
    for file_id in selected:
        print(file_id)

    print("\nSaved list:", OUT_LIST)


if __name__ == "__main__":
    main()
