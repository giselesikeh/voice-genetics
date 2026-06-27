import json
import sys
from pathlib import Path


def rttm_to_segments(rttm_path: Path):
    segments = []

    with rttm_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 8:
                continue

            # RTTM SPEAKER format:
            # SPEAKER file_id channel start duration <NA> <NA> speaker_id <NA> <NA>
            record_type = parts[0]

            if record_type != "SPEAKER":
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

    segments.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))

    return segments


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/rttm_to_reference_json.py input.rttm output.json")
        sys.exit(1)

    rttm_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not rttm_path.exists():
        raise FileNotFoundError(f"RTTM file not found: {rttm_path}")

    segments = rttm_to_segments(rttm_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(segments, indent=2), encoding="utf-8")

    print(f"Converted {len(segments)} segments")
    print(f"Input: {rttm_path}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
