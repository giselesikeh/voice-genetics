# Voice Genetics Launch Instructions

This document explains how to launch the final Voice Genetics product locally, through Docker, and through the browser frontend.

Voice Genetics has two main components:

```text
FastAPI backend   → audio processing, feature extraction, diarization, evaluation
Streamlit frontend → browser user interface
```

The recommended final submission launch method is the **standalone Docker container**.

---

## 1. Repository

Clone the project:

```bash
git clone https://github.com/giselesikeh/voice-genetics.git
cd voice-genetics
```

The project root should contain:

```text
Dockerfile
start.sh
.dockerignore
backend/
frontend/
scripts/
README.md
instructions/
```

---

## 2. Recommended Option: Launch with Docker

This is the easiest way to run the final product because it starts both the backend and frontend from one container.

### 2.1 Requirements

Install and start Docker Desktop.

For Windows + WSL users:

1. Open Docker Desktop.
2. Go to `Settings → Resources → WSL Integration`.
3. Enable integration for the Ubuntu WSL distro.
4. Apply and restart Docker Desktop.

Check Docker:

```bash
docker --version
docker ps
```

### 2.2 Build the Docker Image

From the project root:

```bash
docker build --progress=plain -t voice-genetics-final .
```

The image is built from the root `Dockerfile`.

Successful build should end with something similar to:

```text
naming to docker.io/library/voice-genetics-final:latest done
unpacking to docker.io/library/voice-genetics-final:latest done
DONE
```

### 2.3 Run the Docker Container

```bash
docker run --rm \
  -p 8000:8000 \
  -p 8501:8501 \
  -v voice-genetics-cache:/app/.cache \
  voice-genetics-final
```

This starts:

```text
FastAPI backend on port 8000
Streamlit frontend on port 8501
```

The cache volume stores downloaded model files between runs.

### 2.4 Open the App

Frontend:

```text
http://localhost:8501
```

Backend health check:

```text
http://localhost:8000/health
```

Swagger API documentation:

```text
http://localhost:8000/docs
```

In the Streamlit sidebar, keep the backend URL as:

```text
http://127.0.0.1:8000
```

---

## 3. Manual Local Launch Without Docker

Use this option if you want to run the backend and frontend separately.

---

## 4. Launch the Backend Manually

From the project root:

```bash
cd backend
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Start FastAPI:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Check the backend:

```text
http://localhost:8000/health
```

Keep this terminal running.

---

## 5. Launch the Frontend Manually

Open a second terminal.

From the project root:

```bash
cd frontend
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Start Streamlit:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

Make sure the sidebar backend URL is:

```text
http://127.0.0.1:8000
```

---

## 6. Recommended Final Demo Settings

Use a short `.wav`, `.mp3`, or `.m4a` file.

Recommended settings:

```text
Segmentation method: ecapa_v2
Expected speakers: 2 or 3
Chunk duration: 2.0
VAD mode: adaptive
VAD top_db: 30
VAD minimum RMS: 0.015
VAD minimum region duration: 0.25
VAD merge gap: 0.8
ECAPA chunk hop: 1.0
ECAPA smoothing passes: 1
```

Recommended final method:

```text
Method 3B: ecapa_v2
```

---

## 7. How to Use the Frontend

1. Open `http://localhost:8501`.
2. Confirm backend URL is `http://127.0.0.1:8000`.
3. Upload a supported audio file.
4. Select a segmentation method.
5. Enter expected speaker count if using diarization.
6. Click the processing button.
7. Review the returned outputs.

The frontend displays:

- audio quality metrics;
- preprocessing information;
- voice activity detection results;
- speaker segmentation output;
- speaker duration table;
- speaker segment table;
- clustering metrics;
- global acoustic features;
- advanced acoustic features;
- speaker-level acoustic features;
- privacy status;
- raw JSON output.

---

## 8. Supported Audio Formats

The product supports:

```text
.wav
.mp3
.m4a
```

Unsupported file types are rejected.

---

## 9. Processing Methods

| Method value | Description |
|---|---|
| `none` | Acoustic features only |
| `auto` | Method 2: handcrafted DSP features + K-Means |
| `ecapa` | Method 3: ECAPA-TDNN speaker embeddings |
| `ecapa_v2` | Method 3B: improved ECAPA-TDNN diarization |
| `wavlm` | Method 4: WavLM embeddings + K-Means |

For the final demonstration, use:

```text
ecapa_v2
```

---

## 10. Test the Backend with cURL

Keep the backend running.

From the project root:

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "file=@data/samples/sample.wav"
```

Method 3B example:

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "file=@data/samples/sample.wav" \
  -F "segmentation_method=ecapa_v2" \
  -F "expected_speakers=2" \
  -F "chunk_duration_seconds=2.0" \
  -F "vad_mode=adaptive" \
  -F "vad_top_db=30" \
  -F "vad_min_rms=0.015" \
  -F "vad_min_region_duration_seconds=0.25" \
  -F "vad_merge_gap_seconds=0.8" \
  -F "ecapa_chunk_hop_seconds=1.0" \
  -F "ecapa_smoothing_passes=1"
```

Save output:

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "file=@data/samples/sample.wav" \
  -F "segmentation_method=ecapa_v2" \
  -F "expected_speakers=2" \
  -o output_result.json
```

---

## 11. Reference-Based Evaluation

If reference speaker timestamps are available, pass them as:

```text
reference_segments_json
```

This is used only for evaluation and does not affect prediction.

The backend can return:

- DER;
- speech precision;
- speech recall;
- speaker assignment accuracy;
- missed speech;
- false alarm;
- speaker confusion;
- boundary metrics.

Manual segments can be passed as:

```text
segments_json
```

This overrides automatic diarization.

---

## 12. Final 73-File Evaluation

The final evaluation script is:

```text
scripts/run_method3b_73_with_bootstrap.py
```

It runs Method 3B on the selected VoxConverse subset and saves:

```text
results/method3b_73_bootstrap/per_file_metrics.csv
results/method3b_73_bootstrap/bootstrap_summary.csv
results/method3b_73_bootstrap/bootstrap_draws.csv
```

Note:

The raw VoxConverse audio dataset is not included in GitHub or Docker because of size and dataset restrictions. The final repository includes the scripts and summary outputs.

---

## 13. Docker Troubleshooting

### Docker cannot find the Dockerfile

Run Docker from the project root:

```bash
ls Dockerfile start.sh .dockerignore backend frontend
docker build --progress=plain -t voice-genetics-final .
```

### Docker build context is too large

If Docker shows several GBs during:

```text
transferring context
```

check `.dockerignore`.

The following should be excluded:

```text
data/
results/
reports/
.venv/
.venv-reference/
backend/.venv/
frontend/.venv/
backend/pretrained_models/
*.wav
*.mp3
*.m4a
*.pt
*.pth
*.bin
*.safetensors
```

### Docker command not found in WSL

Open Docker Desktop and enable:

```text
Settings → Resources → WSL Integration → Ubuntu-22.04
```

Then restart WSL:

```powershell
wsl --shutdown
```

Open a new Ubuntu terminal and check:

```bash
docker --version
docker ps
```

### First model run is slow

The first run may download pretrained models. Use the Docker cache volume:

```bash
-v voice-genetics-cache:/app/.cache
```

---

## 14. Privacy Behavior

The system is privacy-aware.

The response includes privacy status such as:

```json
{
  "raw_audio_stored": false,
  "temporary_files_deleted": true,
  "output_contains_raw_audio": false
}
```

The system returns numerical features and metadata, not raw waveform data.

---

## 15. Quick Final Demo Checklist

Before presenting:

1. Start Docker or start backend/frontend manually.
2. Open `http://localhost:8501`.
3. Confirm backend URL is `http://127.0.0.1:8000`.
4. Upload a short supported audio file.
5. Select `ecapa_v2`.
6. Set expected speakers to `2` or `3`.
7. Run processing.
8. Show quality metrics.
9. Show VAD results.
10. Show speaker segmentation and speaker durations.
11. Show clustering metrics.
12. Show acoustic and advanced acoustic features.
13. Show privacy status.
14. Mention the 73-file bootstrap evaluation.

---

## 16. Summary

Recommended final launch command:

```bash
docker run --rm \
  -p 8000:8000 \
  -p 8501:8501 \
  -v voice-genetics-cache:/app/.cache \
  voice-genetics-final
```

Then open:

```text
http://localhost:8501
```

The final product is ready to process supported audio files and demonstrate the full Voice Genetics pipeline.
