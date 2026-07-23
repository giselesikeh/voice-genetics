import csv, json, zipfile
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8000/extract"
OUT = ROOT / "results" / "review4_method3b_dataset"

SAMPLES = {
    "afjiv": 5,
    "akthc": 2,
    "ampme": 3,
    "asxwr": 3,
    "aufkn": 3,
}

def find_file(file_id, exts):
    for ext in exts:
        hits = list(ROOT.rglob(f"{file_id}{ext}"))
        hits = [p for p in hits if ".venv" not in str(p)]
        if hits:
            return hits[0]
    return None

def rttm_to_reference_json(path):
    segs = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 8:
            start = float(p[3])
            dur = float(p[4])
            segs.append({"speaker": p[7], "start": round(start, 3), "end": round(start + dur, 3)})
    return json.dumps(segs)

def get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def run_one(file_id, speakers):
    audio = find_file(file_id, [".wav", ".m4a", ".mp3"])
    rttm = find_file(file_id, [".rttm"])

    print(f"\n=== Method 3B ECAPA V2: {file_id} ===")
    print("Audio:", audio)
    print("RTTM:", rttm)

    data = {
        "segmentation_method": "ecapa_v2",
        "expected_speakers": str(speakers),
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
        "max_speakers": "6",
    }

    if rttm:
        data["reference_segments_json"] = rttm_to_reference_json(rttm)

    with audio.open("rb") as f:
        res = requests.post(URL, files={"file": (audio.name, f, "audio/wav")}, data=data, timeout=2400)

    if res.status_code != 200:
        print(res.text)
        res.raise_for_status()

    result = res.json()
    out_path = OUT / f"{file_id}_method3b_ecapa_v2.json"
    out_path.write_text(json.dumps(result, indent=2))
    print("Saved:", out_path)
    return result

def row(file_id, speakers, r):
    cm = get(r, ["speaker_segmentation", "clustering_metrics"], {}) or {}
    ev = get(r, ["diarization_evaluation"], {}) or {}
    rates = ev.get("rate_metrics", {})
    times = ev.get("time_metrics_seconds", {})
    q = r.get("quality_metrics", {})
    va = r.get("voice_activity", {})
    feat = r.get("features", {})
    pitch = feat.get("pitch", {})
    spec = feat.get("spectral_centroid", {})

    return {
        "file_id": file_id,
        "expected_speakers": speakers,
        "duration_seconds": q.get("duration_seconds"),
        "global_rms": q.get("rms_energy"),
        "speech_coverage": va.get("speech_coverage_ratio"),
        "effective_min_rms": va.get("effective_min_rms"),
        "DER_percent": rates.get("der_percent"),
        "precision": rates.get("speech_precision"),
        "recall": rates.get("speech_recall"),
        "speaker_accuracy_overlap": rates.get("speaker_assignment_accuracy_on_overlap"),
        "missed_speech": times.get("missed_speech"),
        "false_alarm": times.get("false_alarm"),
        "speaker_confusion": times.get("speaker_confusion"),
        "silhouette_score": cm.get("silhouette_score"),
        "cluster_balance_ratio": cm.get("cluster_balance_ratio"),
        "speaker_switches": cm.get("speaker_switches"),
        "segment_smoothness": cm.get("segment_smoothness"),
        "mean_segment_duration_seconds": cm.get("mean_segment_duration_seconds"),
        "pitch_mean_hz": pitch.get("mean_hz"),
        "pitch_min_hz": pitch.get("min_hz"),
        "pitch_max_hz": pitch.get("max_hz"),
        "spectral_centroid_mean": spec.get("mean"),
        "spectral_centroid_std": spec.get("std"),
        "runtime_seconds": get(r, ["runtime_metrics", "total_processing_seconds"]),
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    requests.get("http://127.0.0.1:8000/health", timeout=10).raise_for_status()

    rows = []
    for fid, spk in SAMPLES.items():
        result = run_one(fid, spk)
        rows.append(row(fid, spk, result))

    summary = OUT / "summary_method3b.csv"
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    zip_path = ROOT / "results" / "review4_method3b_dataset.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob("*"):
            z.write(p, p.relative_to(ROOT))

    print("\nDONE")
    print("ZIP:", zip_path)

if __name__ == "__main__":
    main()
