# Voice Genetics: Privacy-Aware Acoustic Feature Extraction and Speaker Diarization System

Voice Genetics is a privacy-aware audio processing system that accepts voice recordings, validates audio quality, preprocesses the signal, detects speech regions, extracts numerical acoustic features, optionally performs speaker diarization, evaluates diarization when reference labels are available, and returns a structured JSON response.

The system does **not** predict genes directly. Instead, it prepares standardized acoustic and speaker-level numerical features that may later support research on possible relationships between vocal traits and genetic markers.

---

## Project Links

- GitHub repository: `https://github.com/giselesikeh/voice-genetics`
- Live Hugging Face Space: `https://gisele-voice-genetics.hf.space`

---

## Team Members

- Sikeh Gisele Wiykiynyuy
- Tivdzua Lubem Noah

---

## Course

Software System Development Using State-of-the-Art Artificial Intelligence Technologies

---

## Product Overview

The product is an end-to-end acoustic feature extraction and speaker segmentation pipeline.

The user uploads an audio file through either:

1. the FastAPI backend, or
2. the Streamlit frontend.

The system then:

```text
audio upload
    ↓
file validation
    ↓
audio decoding
    ↓
mono conversion
    ↓
16 kHz resampling
    ↓
quality analysis
    ↓
adaptive voice activity detection
    ↓
acoustic feature extraction
    ↓
optional speaker diarization
    ↓
optional diarization evaluation
    ↓
privacy-safe JSON output
```

The output is a JSON object containing numerical acoustic features, quality metrics, diarization results, evaluation metrics, runtime metrics, and privacy information.

---

## Expected Functionality

The final system is expected to:

- accept common audio recordings;
- support `.wav`, `.mp3`, and `.m4a`;
- reject unsupported formats;
- validate audio quality;
- preprocess audio by converting to mono and resampling to 16 kHz;
- detect speech and non-speech regions;
- remove silence and pauses where appropriate;
- extract global acoustic features;
- extract speaker-level acoustic features;
- support manual speaker segmentation;
- support automatic speaker diarization;
- evaluate diarization using reference speaker annotations;
- return structured JSON output;
- avoid permanent storage of raw uploaded audio;
- provide a frontend user interface;
- handle incorrect or unexpected user inputs.

---

## Implemented Functionality

The current implementation includes:

- FastAPI backend;
- Streamlit frontend;
- `/health` endpoint;
- `/extract` endpoint;
- browser-based audio upload;
- support for `.wav`, `.mp3`, and `.m4a`;
- rejection of unsupported files;
- audio decoding with FFmpeg/librosa support;
- mono conversion;
- 16 kHz resampling;
- audio quality metrics;
- adaptive voice activity detection;
- silence and pause handling;
- global acoustic feature extraction;
- advanced acoustic feature extraction;
- manual segmentation through `segments_json`;
- reference-based evaluation through `reference_segments_json`;
- Method 2 speaker segmentation using handcrafted DSP features and K-Means;
- Method 3 speaker segmentation using ECAPA-TDNN embeddings;
- Method 3B improved ECAPA-TDNN diarization pipeline;
- Method 4 WavLM-based speaker embedding clustering;
- speaker-level acoustic features;
- clustering quality metrics;
- frame-based permutation-invariant diarization evaluation;
- bootstrap evaluation over multiple samples;
- runtime metrics;
- privacy status reporting;
- Docker support;
- Hugging Face deployment support.

---

## Supported File Formats

The system currently supports:

```text
.wav
.mp3
.m4a
```

Unsupported formats are rejected with a clear error message.

---

## Repository Structure

```text
voice-genetics/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── audio_io.py
│   │   ├── quality.py
│   │   ├── vad.py
│   │   ├── features.py
│   │   ├── advanced_features.py
│   │   ├── speaker_segmentation.py
│   │   ├── evaluation.py
│   │   └── privacy.py
│   ├── tests/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── requirements-docker.txt
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   ├── .streamlit/
│   │   └── config.toml
│   ├── components/
│   │   ├── sidebar.py
│   │   ├── upload_panel.py
│   │   └── results_view.py
│   ├── services/
│   │   └── api_client.py
│   └── utils/
│       └── formatting.py
├── scripts/
│   ├── prepare_voxconverse_kaggle_subset.py
│   ├── rttm_to_reference_json.py
│   ├── select_voxconverse_short_2_3_speakers.py
│   ├── run_review3_method3_dataset.py
│   ├── run_review4_method3b_dataset.py
│   └── run_method3b_73_with_bootstrap.py
├── results/
│   └── method3b_73_bootstrap/
│       ├── per_file_metrics.csv
│       └── bootstrap_summary.csv
├── data/
│   └── samples/
├── docs/
├── reports/
├── README.md
└── .gitignore
```

Large datasets, raw audio files, pretrained model folders, virtual environments, cache folders, and private local files are intentionally excluded from the repository.

---

## System Architecture

```text
User
  ↓
Streamlit Frontend
  ↓
FastAPI Backend
  ↓
Audio I/O Layer
  ↓
Quality Analysis
  ↓
Adaptive VAD
  ↓
Acoustic Feature Extraction
  ↓
Optional Speaker Diarization
  ↓
Optional Evaluation
  ↓
JSON Response
```

Main backend modules:

| Module | Purpose |
|---|---|
| `main.py` | FastAPI routes and request handling |
| `audio_io.py` | File validation, decoding, mono conversion, and resampling |
| `quality.py` | Audio quality metrics |
| `vad.py` | Voice activity detection and adaptive VAD |
| `features.py` | Basic acoustic feature extraction |
| `advanced_features.py` | Formants, jitter, shimmer, HNR, speaking-rate proxy, and VOT proxy |
| `speaker_segmentation.py` | Manual, DSP/K-Means, ECAPA, ECAPA V2, and WavLM diarization |
| `evaluation.py` | Frame-based diarization evaluation |
| `privacy.py` | Privacy status reporting |
| `schemas.py` | Structured response models |

---

## AI Models and Algorithms Used

The product uses pretrained speech models and clustering algorithms.

### ECAPA-TDNN

Model:

```text
speechbrain/spkrec-ecapa-voxceleb
```

Used in:

```text
segmentation_method=ecapa
segmentation_method=ecapa_v2
```

ECAPA-TDNN means:

```text
Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network
```

In this project, ECAPA-TDNN is used to extract speaker embeddings from speech chunks. These embeddings are then clustered to assign anonymous speaker labels.

### WavLM

Model:

```text
microsoft/wavlm-base-plus
```

Used in:

```text
segmentation_method=wavlm
```

WavLM extracts deep speech representations, which are pooled into chunk-level embeddings and clustered.

### Clustering Methods

The system uses:

- K-Means clustering;
- agglomerative clustering with cosine distance;
- L2-normalized speaker embeddings;
- smoothing of isolated speaker labels;
- merging of short or adjacent same-speaker segments.

### Important Training Note

This project does **not** train or fine-tune ECAPA-TDNN or WavLM. These are pretrained models. The project focuses on building a complete production-style audio processing pipeline around these models.

---

## Dataset Used

The main evaluation dataset is **VoxConverse**.

VoxConverse provides conversational audio recordings and RTTM speaker annotations. RTTM files contain speaker timestamps and are used as reference labels for diarization evaluation.

For the final evaluation, we selected:

```text
73 VoxConverse development audio samples
```

Selection criteria:

```text
speaker count: 2 or 3 speakers
maximum duration: under 10 minutes
reference labels: RTTM files available
```

Dataset split used in final evaluation:

| Group | Number of files |
|---|---:|
| 2-speaker recordings | 39 |
| 3-speaker recordings | 34 |
| Total | 73 |

The raw VoxConverse audio files are **not committed** to the repository because of size and dataset licensing considerations. The repository includes the scripts used to select samples and convert RTTM annotations.

---

## Backend API

The backend is built with FastAPI.

### Base URL

```text
http://localhost:8000
```

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Returns basic project information |
| `/health` | GET | Checks whether the backend is running |
| `/extract` | POST | Uploads audio and returns acoustic/speaker-level analysis |

---

## Running the Backend Locally

From the project root, open the backend folder:

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

Run the API:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "voice-genetics-api",
  "message": "Voice Genetics API is running"
}
```

Swagger API documentation:

```text
http://localhost:8000/docs
```

---

## Running the Frontend Locally

Keep the FastAPI backend running on port `8000`.

Open a second terminal:

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

Run Streamlit:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

Default backend URL in the sidebar:

```text
http://127.0.0.1:8000
```

---

## Running with cURL

Keep the backend running.

From the project root:

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "file=@data/samples/sample.wav"
```

Example with Method 3B:

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

---

## Frontend Features

The Streamlit frontend supports:

- browser-based audio upload;
- supported formats: WAV, MP3, M4A;
- backend health check;
- diarization method selection;
- expected speaker count input;
- chunk duration setting;
- adaptive/fixed VAD selection;
- VAD threshold settings;
- ECAPA-specific settings;
- optional manual/reference segment input;
- quality metric display;
- VAD summary display;
- speaker segmentation summary;
- speaker duration table;
- speaker segment table;
- clustering metric display;
- acoustic feature display;
- privacy status display;
- raw JSON display.

Available frontend methods:

| Method value | Frontend label | Purpose |
|---|---|---|
| `none` | Acoustic features only | No speaker segmentation |
| `auto` | Method 2 | Handcrafted DSP features + K-Means |
| `ecapa` | Method 3 | ECAPA-TDNN speaker embeddings |
| `ecapa_v2` | Method 3B | Improved ECAPA diarization pipeline |
| `wavlm` | Method 4 | WavLM embeddings + K-Means |

Recommended method:

```text
ecapa_v2
```

---

## Speaker Segmentation Methods

### Method 1: No Speaker Segmentation

Selected using:

```text
segmentation_method=none
```

This mode extracts only global acoustic features.

### Method 2: DSP Features + K-Means

Selected using:

```text
segmentation_method=auto
```

Pipeline:

```text
audio → VAD speech chunks → handcrafted acoustic features → K-Means → speaker labels
```

This is a lightweight baseline. It is fast but less speaker-specific because it uses features such as RMS, MFCCs, spectral information, and pitch-related measurements.

### Method 3: ECAPA-TDNN Speaker Embeddings

Selected using:

```text
segmentation_method=ecapa
```

Pipeline:

```text
audio → VAD speech chunks → ECAPA speaker embeddings → clustering → speaker segments
```

This is stronger than Method 2 because ECAPA embeddings are speaker-oriented representations.

### Method 3B: Improved ECAPA-TDNN Pipeline

Selected using:

```text
segmentation_method=ecapa_v2
```

This is the recommended final method.

Method 3B includes:

- central/adaptive VAD;
- overlapping speech chunks;
- short-region handling;
- ECAPA-TDNN speaker embeddings;
- L2 embedding normalization;
- agglomerative cosine clustering;
- isolated-label smoothing;
- short-segment merging;
- speaker-level acoustic features;
- clustering metrics;
- optional reference-based evaluation.

Pipeline:

```text
audio
  → adaptive VAD
  → overlapping speech chunks
  → ECAPA embeddings
  → L2 normalization
  → agglomerative cosine clustering
  → smoothing
  → segment merging
  → speaker segments
  → evaluation if reference is provided
```

### Method 4: WavLM Embeddings + K-Means

Selected using:

```text
segmentation_method=wavlm
```

Pipeline:

```text
audio → speech chunks → WavLM hidden states → pooled embeddings → K-Means → speaker segments
```

Limitations:

- slow on CPU;
- can over-group speakers;
- may capture content/context, not only speaker identity;
- useful mainly as a comparison baseline.

---

## Adaptive Voice Activity Detection

Default VAD mode:

```text
vad_mode=adaptive
```

Recommended settings:

```text
vad_mode=adaptive
vad_top_db=30
vad_min_rms=0.015
vad_min_region_duration_seconds=0.25
vad_merge_gap_seconds=0.8
```

Adaptive VAD adjusts the effective RMS threshold based on recording loudness.

Example:

```text
Requested min RMS: 0.015
Effective min RMS: 0.005
Reason: quiet_audio_detected_global_rms_below_0.025_using_min_rms_0.005
```

This helps keep quiet speech instead of removing it too aggressively.

---

## Acoustic Features Extracted

### Audio Quality Metrics

- duration;
- sample rate;
- RMS energy;
- peak amplitude;
- clipping rate.

### Basic Acoustic Features

- MFCC means;
- MFCC standard deviations;
- spectral centroid mean;
- spectral centroid standard deviation;
- pitch mean;
- pitch minimum;
- pitch maximum.

### Advanced Acoustic Features

The final system also extracts:

- formants F1, F2, F3;
- jitter;
- shimmer;
- harmonic-to-noise ratio;
- speaking-rate proxy;
- voice-onset-time proxy.

Important note:

Some advanced acoustic measurements are approximate signal-based estimates:

- formants are estimated using LPC roots;
- jitter and shimmer are estimated from frame-level pitch and RMS variation;
- HNR is estimated using a harmonic/residual energy ratio;
- speaking rate is estimated from syllable-like RMS envelope peaks;
- VOT is estimated as an energy-onset-to-first-voiced-frame proxy.

True clinical-grade jitter, shimmer, HNR, speaking rate, and VOT would require tools such as Praat, transcript alignment, or phoneme-level forced alignment.

---

## Manual Speaker Segmentation

Manual speaker labels can be provided using `segments_json`.

Example:

```json
[
  {
    "speaker": "speaker_1",
    "start": 0.0,
    "end": 5.0
  },
  {
    "speaker": "speaker_2",
    "start": 5.5,
    "end": 10.0
  }
]
```

When `segments_json` is provided, it overrides automatic diarization.

This is useful when exact speaker timestamps are already known.

---

## Reference-Based Diarization Evaluation

Reference speaker labels can be provided using `reference_segments_json`.

Important distinction:

- `segments_json` is used as manual speaker segmentation input.
- `reference_segments_json` is used only for evaluation.
- `reference_segments_json` does not change the automatic prediction.

The evaluation is:

```text
frame-based
permutation-invariant
```

Frame-based means the timeline is divided into small time steps and compared against reference labels.

Permutation-invariant means predicted labels are mapped to reference labels before scoring, because automatic cluster labels are anonymous.

Returned metrics include:

- DER;
- DER percentage;
- speech precision;
- speech recall;
- speaker assignment accuracy;
- missed speech;
- false alarm;
- speaker confusion;
- speaker duration error;
- boundary metrics.

---

## Understanding DER

DER means:

```text
Diarization Error Rate
```

Lower DER is better.

DER combines three error types:

| Error type | Meaning |
|---|---|
| Missed speech | Real speech exists, but the system marks it as non-speech |
| False alarm | System marks non-speech/silence as speech |
| Speaker confusion | System detects speech but assigns the wrong speaker |

Simplified:

```text
DER = missed speech + false alarm + speaker confusion
```

---

## Final Evaluation on 73 VoxConverse Samples

The final Method 3B evaluation was run on 73 VoxConverse samples with 2 or 3 speakers and duration under 10 minutes.

The evaluation script used:

```text
scripts/run_method3b_73_with_bootstrap.py
```

Output files:

```text
results/method3b_73_bootstrap/per_file_metrics.csv
results/method3b_73_bootstrap/bootstrap_summary.csv
```

### Overall Result

| Metric | Value |
|---|---:|
| Number of files | 73 |
| 2-speaker files | 39 |
| 3-speaker files | 34 |
| Bootstrap mean DER | 21.44% |
| Bootstrap std DER | 6.37% |
| 95% bootstrap CI | 12.73% – 36.05% |
| Median per-file DER | 8.02% |
| Micro DER over all frames | 15.53% |

Interpretation:

Method 3B works well on many recordings, but the mean DER is increased by several difficult outlier files. The median DER is much lower than the mean, showing that the typical file performs better than the average suggests.

### DER Distribution

| DER range | Number of files |
|---|---:|
| ≤ 5% | 26 |
| 5–10% | 15 |
| 10–20% | 8 |
| 20–50% | 20 |
| 50–100% | 3 |
| >100% | 1 |

Summary:

```text
41 out of 73 files had DER below 10%.
49 out of 73 files had DER below 20%.
```

### 2-Speaker vs 3-Speaker Results

| Group | Files | Bootstrap mean DER | Bootstrap std | Median DER |
|---|---:|---:|---:|---:|
| 2 speakers | 39 | 15.93% | 3.07% | 7.15% |
| 3 speakers | 34 | 27.96% | 12.92% | 8.50% |

Interpretation:

3-speaker files are more difficult on average. The median DER for 3-speaker files is still close to the 2-speaker median, but the 3-speaker mean is inflated by difficult outliers.

### Error Breakdown

Across all 73 files:

| Error type | Total seconds | Contribution relative to reference speech |
|---|---:|---:|
| Missed speech | 72.3 s | 0.47% |
| False alarm | 788.4 s | 5.07% |
| Speaker confusion | 1551.8 s | 9.99% |
| Total diarization error | 2412.5 s | 15.53% |

Interpretation:

The system rarely misses speech. The main remaining problem is speaker confusion, followed by false alarm speech.

### Speech Detection

| Metric | Mean |
|---|---:|
| Speech precision | 94.08% |
| Speech recall | 99.47% |
| Pause detection accuracy | 30.08% |

Interpretation:

The VAD is recall-heavy. It keeps almost all speech, which reduces missed speech, but it sometimes keeps too much non-speech as speech.

### Clustering Quality

| Metric | Overall mean | 2-speaker mean | 3-speaker mean |
|---|---:|---:|---:|
| Silhouette score | 0.410 | 0.445 | 0.371 |
| Cluster balance ratio | 0.257 | 0.354 | 0.145 |
| Segment smoothness | 0.957 | 0.972 | 0.940 |

Interpretation:

Method 3B produces smooth timelines, but cluster balance remains a challenge, especially for 3-speaker recordings where one speaker may dominate or a minority speaker may be collapsed.

### Runtime

| Runtime metric | Value |
|---|---:|
| Total runtime for 73 files | 12.77 hours |
| Mean runtime per file | 629.57 s |
| Median runtime per file | 491.65 s |
| Average processing time per 1 minute of audio | 162.6 s |

Interpretation:

The system is functional but computationally heavy. ECAPA-based diarization on long files is slow on CPU, so short demo files are preferred for live presentation.

---

## Bootstrap Evaluation

The project uses bootstrapping rather than cross-validation.

Reason:

The system evaluates a fixed diarization and acoustic-feature extraction pipeline. It does not train a supervised model on the 73 VoxConverse files. Therefore, bootstrapping is more appropriate for estimating how stable the average performance is across different possible sample subsets.

Bootstrap procedure:

```text
1. Run Method 3B once on all 73 files.
2. Store one metric row per file.
3. Randomly sample 73 files with replacement.
4. Compute the average metric for that sampled set.
5. Repeat this process 1000 times.
6. Report the mean and standard deviation of the bootstrap averages.
```

The bootstrap output is saved in:

```text
results/method3b_73_bootstrap/bootstrap_summary.csv
```

---

## Privacy Design

The system is designed to avoid permanent storage of uploaded raw audio.

The backend response includes a privacy section similar to:

```json
{
  "raw_audio_stored": false,
  "temporary_files_deleted": true,
  "output_contains_raw_audio": false
}
```

Privacy principles:

- process audio temporarily;
- delete temporary files after processing;
- return numerical features and metadata only;
- do not return raw waveform data;
- do not identify real people;
- use anonymous speaker labels such as `speaker_1`, `speaker_2`.

---

## Handling Incorrect or Unexpected User Actions

The system handles several unexpected cases:

- unsupported file formats are rejected;
- missing uploaded files return an error;
- invalid JSON in `segments_json` or `reference_segments_json` is rejected;
- invalid segment timestamps are checked;
- reference segments that exceed audio duration are rejected;
- missing expected speaker count is handled depending on method;
- low-quality or quiet audio produces quality/VAD warnings;
- low-confidence clustering can produce warnings;
- cluster imbalance can produce warnings;
- raw audio is not returned in the response.

---

## Docker Support

The final project includes a **standalone full-stack Docker container**.

The container starts both:

```text
FastAPI backend  → port 8000
Streamlit UI     → port 8501
```

This means the instructor can build and run the full product with Docker, then open the browser interface without manually starting the backend and frontend separately.

### Docker Files

The repository includes:

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the full-stack Docker image |
| `start.sh` | Starts FastAPI first, waits for the backend health check, then starts Streamlit |
| `.dockerignore` | Excludes datasets, virtual environments, model caches, raw audio, and large result folders from the Docker image |

The Docker image uses:

```text
python:3.10-slim-bookworm
```

System dependencies installed inside the image include:

```text
ffmpeg
libsndfile1
build-essential
curl
git
```

These are required for audio decoding, speech processing, and health checks.

### Important Docker Packaging Note

The Docker image intentionally contains **code only**, not large local datasets or model caches.

Excluded from the Docker build context:

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

This keeps the Docker image suitable for submission and prevents Docker from copying several gigabytes of local experiment files.

### Build the Docker Image

From the project root:

```bash
docker build --progress=plain -t voice-genetics-final .
```

Successful build evidence from the final local test:

```text
naming to docker.io/library/voice-genetics-final:latest done
unpacking to docker.io/library/voice-genetics-final:latest done
DONE
```

### Run the Docker Container

From the project root, run:

```bash
docker run --rm \
  -p 8000:8000 \
  -p 8501:8501 \
  -v voice-genetics-cache:/app/.cache \
  voice-genetics-final
```

The named volume:

```text
voice-genetics-cache
```

is used to persist Hugging Face/Torch model downloads between runs, so models do not need to be downloaded every time the container starts.

### Open the Product

After the container starts, open:

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

In the Streamlit sidebar, the backend URL should remain:

```text
http://127.0.0.1:8000
```

### Recommended Docker Demo

Use a short `.wav`, `.mp3`, or `.m4a` file.

Recommended settings:

```text
Method: ecapa_v2
Expected speakers: 2 or 3
Chunk duration: 2.0
VAD mode: adaptive
VAD top_db: 30
VAD min RMS: 0.015
ECAPA hop: 1.0
Smoothing passes: 1
```

The first run may be slower because the container may download pretrained speech models. Later runs are faster if the cache volume is reused.

### Docker Troubleshooting

If Docker says it cannot find the Dockerfile, make sure the command is run from the project root, where these files exist:

```text
Dockerfile
start.sh
.dockerignore
backend/
frontend/
scripts/
```

If Docker transfers several gigabytes during build, check `.dockerignore`. Large folders such as `data/`, `.venv-reference/`, `backend/pretrained_models/`, `results/`, and virtual environments should be excluded.

If using Windows + WSL, Docker Desktop must be running and WSL integration must be enabled for the Ubuntu distro.

---

## Hugging Face Deployment

The live project is deployed as a Hugging Face Space.

The Space runs:

- FastAPI backend internally on port `8000`;
- Streamlit frontend publicly on port `8501`.

Public app:

```text
https://gisele-voice-genetics.hf.space
```

Inside the Hugging Face container, the frontend communicates with the backend through:

```text
http://127.0.0.1:8000
```

---

## Difficulties Faced

Main difficulties during development:

1. **Speaker diarization is harder than simple audio feature extraction.**  
   Basic acoustic features were easier to compute, but speaker segmentation required model embeddings, clustering, smoothing, and evaluation.

2. **Handcrafted DSP clustering was not speaker-specific enough.**  
   Method 2 sometimes confused pitch, loudness, or content differences with speaker identity.

3. **WavLM was slow and not always speaker-specific.**  
   WavLM embeddings sometimes captured content or acoustic context, not only speaker identity.

4. **Quiet recordings required adaptive VAD.**  
   Fixed VAD thresholds removed too much quiet speech in difficult recordings.

5. **Speaker-count and cluster imbalance remained difficult.**  
   Some files had one dominant speaker and one very small speaker cluster. In some cases, minority speakers were collapsed.

6. **False alarms and speaker confusion remained the main error sources.**  
   The final evaluation showed very low missed speech but higher speaker confusion and false alarm errors.

7. **Runtime was heavy.**  
   Running ECAPA-based diarization over 73 files took many hours.

8. **Some acoustic features are difficult without linguistic labels.**  
   True speaking rate and voice onset time require transcript or phoneme-level alignment, so the current implementation reports proxy estimates.

---

## Current Limitations

Current limitations:

- diarization is slow on long recordings;
- ECAPA V2 can take several minutes for recordings longer than 2–3 minutes;
- speaker confusion still occurs on difficult files;
- false alarms still occur when non-speech is kept as speech;
- cluster imbalance remains a challenge;
- minority speakers can sometimes be collapsed;
- overlap speech is assigned to only one speaker cluster;
- WavLM is slow on CPU;
- true speaking rate requires transcripts;
- true VOT requires phoneme-level alignment;
- jitter, shimmer, and HNR are approximate signal-based estimates;
- standard pyannote DER is not enabled by default;
- `/extract` currently handles both inference and evaluation.

---

## Future Work

Future work may include:

- separate `/evaluate` endpoint;
- direct RTTM upload support;
- standard DER computation using `pyannote.metrics`;
- collar-based DER evaluation;
- overlap-aware diarization scoring;
- stronger noise and SNR estimation;
- better speaker-change boundary detection;
- improved clustering for difficult multi-speaker files;
- pyannote or NeMo diarization baseline;
- external validation on another diarization dataset;
- improved VOT using forced alignment;
- improved speaking rate using transcripts;
- Praat/parselmouth-based jitter, shimmer, HNR, and formants;
- runtime optimization and batching;
- downloadable frontend report.

---

## Final Review Evidence

The final implementation demonstrates that:

- the API starts successfully;
- `/health` returns OK;
- `/extract` accepts supported audio files;
- unsupported formats are rejected;
- audio is decoded and resampled;
- audio quality metrics are computed;
- adaptive VAD is applied;
- basic acoustic features are extracted;
- advanced acoustic features are extracted;
- speaker segmentation methods are selectable;
- ECAPA V2 diarization works;
- speaker-level features are extracted;
- RTTM-based evaluation works;
- bootstrap evaluation over 73 files works;
- JSON responses are returned;
- privacy status is included;
- the Streamlit frontend runs locally and on Hugging Face;
- raw audio is not permanently stored.

---

## Final Project Summary

Voice Genetics is a working privacy-aware acoustic feature extraction and speaker diarization system. It provides a FastAPI backend, Streamlit frontend, multiple speaker segmentation methods, full acoustic feature extraction, reference-based diarization evaluation, and bootstrap-based final reporting.

The final 73-file VoxConverse evaluation shows that Method 3B performs well on many files, with 41 out of 73 files below 10% DER and a median DER of 8.02%. The bootstrap mean DER is 21.44% ± 6.37%, mainly affected by difficult outliers with speaker confusion, false alarms, and cluster imbalance.

The system is therefore successful as a complete software product and evaluation pipeline, while the remaining work is mainly improving diarization robustness and runtime efficiency.
