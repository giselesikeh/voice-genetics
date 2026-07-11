# Voice Genetics: Acoustic Feature Extraction and Speaker Segmentation System

Voice Genetics is a privacy-aware acoustic feature extraction and speaker segmentation system. It accepts supported audio recordings, validates audio quality, preprocesses the signal, performs voice activity detection, extracts numerical acoustic features, optionally performs speaker segmentation/diarization, and returns structured JSON output.

The system does **not** predict genes directly. Instead, it prepares standardized acoustic and speaker-level features that may later support research on possible relationships between vocal traits and genetic markers.

---

## Project Links

* GitHub repository: `https://github.com/giselesikeh/voice-genetics`
* Live Hugging Face Space: `https://gisele-voice-genetics.hf.space`

---

## Current Implementation Status

The current implementation includes:

* FastAPI backend
* Streamlit frontend
* `/health` endpoint
* `/extract` endpoint
* browser-based audio upload
* supported audio formats: `.wav`, `.mp3`, `.m4a`
* rejection of unsupported formats
* audio decoding
* mono conversion
* 16 kHz resampling
* audio quality metrics
* adaptive voice activity detection
* silence and non-speech region handling
* global acoustic feature extraction
* manual speaker segmentation using `segments_json`
* Method 2 automatic speaker segmentation using handcrafted DSP features and K-Means
* Method 3 speaker segmentation using ECAPA-TDNN embeddings
* Method 3B improved ECAPA V2 diarization pipeline
* Method 4 WavLM-based speaker embedding clustering
* speaker-level acoustic feature extraction
* optional diarization evaluation using reference speaker segments
* structured JSON output
* runtime metrics
* privacy status reporting
* Docker support
* Hugging Face deployment support

---

## Project Goals

The full project aims to:

* accept voice recordings through a backend API and frontend interface
* support common audio formats
* reject unsupported file formats
* validate audio quality before feature extraction
* preprocess audio by converting it to mono and resampling it to 16 kHz
* detect speech and non-speech regions
* support speaker-level processing
* support automatic speaker diarization
* extract basic acoustic features
* return privacy-safe JSON output
* avoid permanent raw audio storage
* provide evaluation metrics when reference annotations are available
* provide a frontend interface for non-technical users

---

## Supported File Formats

The system currently supports:

```text
.wav
.mp3
.m4a
```

Unsupported files are rejected with a clear error message.

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
│   │   ├── speaker_segmentation.py
│   │   ├── evaluation.py
│   │   ├── features.py
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
├── data/
│   └── samples/
├── docs/
├── reports/
├── results/
├── scripts/
│   ├── prepare_voxconverse_kaggle_subset.py
│   └── rttm_to_reference_json.py
├── README.md
└── .gitignore
```

Large datasets, pretrained model folders, virtual environments, cache folders, and private raw data are intentionally excluded from the repository.

---

## System Architecture

```text
Audio upload
    ↓
Input validation and decoding
    ↓
Mono conversion and 16 kHz resampling
    ↓
Quality metrics
    ↓
Adaptive VAD
    ↓
Acoustic feature extraction
    ↓
Optional speaker segmentation
    ↓
Optional diarization evaluation
    ↓
Privacy-safe JSON response
```

The backend is modular so each processing stage can be improved independently.

Main backend modules:

| Module                    | Purpose                                                              |
| ------------------------- | -------------------------------------------------------------------- |
| `main.py`                 | FastAPI routes and request handling                                  |
| `audio_io.py`             | Audio validation, loading, decoding, mono conversion, and resampling |
| `quality.py`              | Audio quality metrics                                                |
| `vad.py`                  | Voice activity detection and adaptive VAD logic                      |
| `features.py`             | Acoustic feature extraction                                          |
| `speaker_segmentation.py` | Manual, DSP/K-Means, ECAPA, ECAPA V2, and WavLM speaker segmentation |
| `evaluation.py`           | Frame-based diarization evaluation                                   |
| `privacy.py`              | Privacy status reporting                                             |
| `schemas.py`              | Response models and structured output definitions                    |

---

## Backend API

The backend is built with FastAPI.

### Base URL

```text
http://localhost:8000
```

### Main Endpoints

| Endpoint   | Method | Purpose                                                   |
| ---------- | ------ | --------------------------------------------------------- |
| `/`        | GET    | Returns basic project information                         |
| `/health`  | GET    | Checks whether the backend is running                     |
| `/extract` | POST   | Uploads audio and returns acoustic/speaker-level analysis |

---

## Running the Backend Locally

Run all backend commands from the backend folder:

```bash
cd "/mnt/f/Innopolis 3 year/Industrial Project/Voice genetics/voice-genetics/backend"
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Run the API:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open the health endpoint:

```text
http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "voice-genetics-api",
  "version": "0.10.0",
  "message": "Voice Genetics API is running"
}
```

---

## Running the Streamlit Frontend Locally

Keep the FastAPI backend running on port `8000`.

Open a second terminal and run:

```bash
cd "/mnt/f/Innopolis 3 year/Industrial Project/Voice genetics/voice-genetics/frontend"
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install frontend requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Open the frontend:

```text
http://localhost:8501
```

Default backend URL in the sidebar:

```text
http://127.0.0.1:8000
```

---

## Frontend Features

The Streamlit frontend supports:

* browser-based audio upload
* supported formats: WAV, MP3, M4A
* backend health check
* method selection
* expected speaker count selection
* chunk duration setting
* adaptive/fixed VAD selection
* VAD threshold settings
* ECAPA-specific settings
* optional manual/reference segment input
* quality metrics display
* voice activity display
* speaker segmentation summary
* speaker duration table
* speaker segment table
* acoustic feature display
* privacy status display
* raw JSON display

Available frontend methods:

| Method value | Frontend label         | Purpose                             |
| ------------ | ---------------------- | ----------------------------------- |
| `none`       | Acoustic features only | No speaker segmentation             |
| `auto`       | Method 2               | Handcrafted DSP features + K-Means  |
| `ecapa`      | Method 3               | ECAPA-TDNN speaker embeddings       |
| `ecapa_v2`   | Method 3B              | Improved ECAPA diarization pipeline |
| `wavlm`      | Method 4               | WavLM embeddings + K-Means          |

---

## Testing Audio Feature Extraction with cURL

Keep the API running in one terminal.

Open a second terminal from the project root:

```bash
cd "/mnt/f/Innopolis 3 year/Industrial Project/Voice genetics/voice-genetics"
```

Test the `/extract` endpoint:

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "file=@data/samples/sample.wav"
```

Expected output includes:

* filename
* quality metrics
* voice activity metrics
* preprocessing metrics
* speaker segmentation status
* diarization evaluation status
* runtime metrics
* acoustic features
* speaker-level features
* warnings
* privacy status

---

## Speaker Segmentation Methods

The backend supports several speaker segmentation modes.

| Method     | Description                                  | Main use                              |
| ---------- | -------------------------------------------- | ------------------------------------- |
| `none`     | No speaker segmentation                      | Acoustic feature extraction only      |
| `auto`     | Handcrafted DSP/VAD-based speaker clustering | Lightweight baseline                  |
| `ecapa`    | ECAPA-TDNN speaker embedding clustering      | Speaker-specific embedding baseline   |
| `ecapa_v2` | Improved ECAPA-TDNN diarization pipeline     | Recommended automatic method          |
| `wavlm`    | WavLM speaker embedding clustering           | Deep speech-representation comparison |

Recommended automatic method:

```text
ecapa_v2
```

Example:

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "file=@data/samples/test.wav" \
  -F "segmentation_method=ecapa_v2" \
  -F "expected_speakers=2"
```

---

## Method 2: DSP Features + K-Means

Method 2 is selected using:

```text
segmentation_method=auto
```

It works as follows:

```text
audio → VAD speech chunks → DSP features → K-Means → speaker labels → merged segments
```

Steps:

1. The uploaded audio is decoded and resampled to 16 kHz mono.
2. VAD detects speech regions.
3. Speech regions are divided into fixed-size chunks.
4. Each chunk is represented using handcrafted acoustic features.
5. Features include RMS energy, MFCCs, spectral information, and pitch-related information.
6. K-Means groups the chunk feature vectors into the expected number of speakers.
7. Cluster IDs are converted into anonymous labels such as `speaker_1` and `speaker_2`.
8. Adjacent chunks with the same label are merged into longer speaker segments.

Limitation:

Method 2 uses handcrafted acoustic measurements, not true speaker identity embeddings. Loudness, pitch, or speaking-style changes can therefore be mistaken for speaker changes.

---

## Method 3: ECAPA-TDNN

Method 3 is selected using:

```text
segmentation_method=ecapa
```

ECAPA-TDNN means:

```text
Emphasized Channel Attention, Propagation and Aggregation Time Delay Neural Network
```

In this project, ECAPA-TDNN is used to extract speaker embeddings from speech chunks. These embeddings are more speaker-specific than handcrafted DSP features, so they are better suited for speaker segmentation.

The basic ECAPA pipeline is:

```text
audio → VAD speech chunks → ECAPA embeddings → clustering → speaker segments
```

---

## Method 3B: ECAPA V2

Method 3B is selected using:

```text
segmentation_method=ecapa_v2
```

ECAPA V2 is the improved diarization pipeline.

It includes:

* central VAD speech-region processing
* overlapping speech chunks
* short-region handling
* embedding extraction using ECAPA-TDNN
* cosine/agglomerative clustering
* smoothing of isolated labels
* merging of adjacent same-speaker segments
* speaker-level feature extraction
* optional reference-based evaluation

Recommended method for final project testing:

```text
ecapa_v2
```

---

## Method 4: WavLM Embeddings + K-Means

Method 4 is selected using:

```text
segmentation_method=wavlm
```

WavLM model used:

```text
microsoft/wavlm-base-plus
```

Method 4 works as follows:

```text
audio → speech chunks → WavLM model → pooled embeddings → K-Means → speaker segments
```

Steps:

1. The uploaded audio is decoded and resampled to 16 kHz mono.
2. Speech chunks are prepared.
3. Each chunk is passed through WavLM.
4. WavLM produces hidden speech representations.
5. Hidden states are pooled into one embedding vector per chunk.
6. K-Means clusters the embeddings into speaker groups.
7. Clusters are converted into speaker labels and merged into speaker segments.

Limitations:

* WavLM is slower on CPU.
* Model loading increases runtime.
* It can over-group most audio into one dominant speaker.
* WavLM embeddings may capture speech content or acoustic context, not only speaker identity.
* It needs better batching, smoothing, and pause handling.

---

## Adaptive Voice Activity Detection

The system includes adaptive voice activity detection.

Default VAD mode:

```text
vad_mode=adaptive
```

Default values:

```text
vad_mode=adaptive
vad_top_db=30
vad_min_rms=0.015
vad_min_region_duration_seconds=0.25
vad_merge_gap_seconds=0.8
```

Adaptive VAD adjusts the RMS threshold depending on the loudness of the recording.

For normal or loud audio, the system keeps the requested RMS threshold.

For quiet audio, the system lowers the effective RMS threshold to avoid removing quiet speech.

Example:

```text
Requested min RMS: 0.015
Effective min RMS: 0.005
Reason: quiet_audio_detected_global_rms_below_0.025_using_min_rms_0.005
```

This was added after testing quiet recordings where strict VAD missed too much speech.

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

This mode is useful when exact speaker timestamps are known.

---

## Diarization Evaluation

The API can evaluate diarization if reference speaker segments are provided through `reference_segments_json`.

Important distinction:

* `segments_json` is used as manual speaker segmentation input.
* `reference_segments_json` is used only for evaluation.
* `reference_segments_json` does not affect the automatic prediction.

The evaluation is frame-based and permutation-invariant.

Frame-based means the audio timeline is divided into small time steps, and each time step is compared against the reference labels.

Permutation-invariant means the evaluation does not assume that predicted `speaker_1` must match reference `speaker_1`. It finds the best speaker-label mapping before scoring, because automatic clustering labels are anonymous.

Returned metrics include:

* diarization error rate
* DER percentage
* speech precision
* speech recall
* speaker assignment accuracy on overlap
* missed speech duration
* false alarm duration
* speaker confusion duration
* speaker duration error
* boundary metrics

---

## Understanding DER

DER means:

```text
Diarization Error Rate
```

It measures how much of the speaker timeline is wrong.

A lower DER is better.

DER is affected by three main error types:

| Error type        | Meaning                                                   |
| ----------------- | --------------------------------------------------------- |
| Missed speech     | Real speech exists, but the system marks it as non-speech |
| False alarm       | System marks non-speech/silence as speech                 |
| Speaker confusion | System detects speech but assigns it to the wrong speaker |

A simplified interpretation is:

```text
DER = missed speech + false alarm + speaker confusion
```

In this project, DER-style evaluation is used to check how close automatic speaker segmentation is to manual ground-truth speaker timestamps.

---

## VoxConverse Evaluation

The ECAPA V2 diarization pipeline was evaluated on a clean VoxConverse subset using RTTM ground-truth annotations.

The selected subset contains five recordings:

| File    | Duration | Speakers |
| ------- | -------: | -------: |
| `afjiv` | 151.248s |        5 |
| `akthc` | 114.521s |        2 |
| `ampme` | 148.320s |        3 |
| `asxwr` | 237.767s |        3 |
| `aufkn` | 181.488s |        3 |

The VoxConverse audio data and full extracted dataset are not committed to this repository because of size. Scripts used for preparing subsets and converting RTTM annotations are included in `scripts/`.

---

## Fixed VAD Results

Initial ECAPA V2 evaluation with fixed VAD produced:

| File    | Speakers |    DER |
| ------- | -------: | -----: |
| `afjiv` |        5 | 66.38% |
| `akthc` |        2 |  3.29% |
| `ampme` |        3 |  6.07% |
| `asxwr` |        3 |  3.60% |
| `aufkn` |        3 |  9.14% |

Average DER over all five files:

```text
17.70%
```

Average DER excluding the difficult quiet outlier `afjiv`:

```text
5.53%
```

Interpretation:

The fixed VAD setup worked well on four normal recordings, with DER values between `3.29%` and `9.14%`. However, it failed badly on `afjiv`, producing `66.38%` DER. The reason was that `afjiv` was a difficult quiet multi-speaker recording, so strict VAD removed or misclassified too much speech.

---

## Adaptive VAD Results

Adaptive VAD improved the difficult quiet recording `afjiv`.

| File    | Speakers | Global RMS | Effective RMS | Speech Coverage |    DER |
| ------- | -------: | ---------: | ------------: | --------------: | -----: |
| `afjiv` |        5 |   0.019047 |         0.005 |          0.9540 | 32.91% |
| `akthc` |        2 |   0.058569 |         0.015 |          0.9330 |  3.29% |
| `ampme` |        3 |   0.050048 |         0.015 |          0.8777 |  6.07% |
| `asxwr` |        3 |   0.039366 |         0.010 |          0.9945 |  3.60% |
| `aufkn` |        3 |   0.057042 |         0.015 |          0.9728 |  9.14% |

Average DER over all five files:

```text
11.00%
```

Adaptive VAD reduced the overall average DER from:

```text
17.70% to 11.00%
```

The largest improvement was on `afjiv`:

```text
Fixed VAD DER:     66.38%
Adaptive VAD DER:  32.91%
Improvement:       33.47 percentage points
```

Interpretation:

Adaptive VAD helped because it detected that `afjiv` had low global RMS and lowered the effective RMS threshold from `0.015` to `0.005`. This allowed the system to keep more quiet speech instead of removing it.

The remaining `afjiv` error was mainly due to false alarm and speaker confusion, not missed speech. This means the system became much better at keeping speech, but it still sometimes assigned non-speech or wrong speakers to the timeline.

For the four normal files, ECAPA V2 remained strong, with DER values between:

```text
3.29% and 9.14%
```

This shows that the ECAPA V2 pipeline works well on normal-quality audio, while quiet multi-speaker recordings remain challenging.

---

## Example Adaptive VAD Result

For the quiet file `afjiv.wav`, adaptive VAD produced:

```text
Adaptive VAD: True
Adaptive reason: quiet_audio_detected_global_rms_below_0.025_using_min_rms_0.005
Requested min RMS: 0.015
Effective min RMS: 0.005
Global RMS: 0.019047
Speech coverage: 0.954
DER: 32.91%
Speech precision: 0.8647
Speech recall: 0.9937
Speaker accuracy overlap: 0.8316
Missed speech: 0.8s
False alarm: 19.6s
Speaker confusion: 21.1s
```

This confirms that adaptive VAD successfully reduced missed speech for quiet recordings.

---

## Current Extracted Metrics

### Audio Quality Metrics

* duration
* sample rate
* RMS energy
* peak amplitude
* clipping rate

### Voice Activity Metrics

* speech regions
* removed non-speech segments
* speech region count
* removed non-speech count
* speech duration
* non-speech duration
* speech coverage ratio
* requested RMS threshold
* effective RMS threshold
* adaptive VAD reason
* global RMS

### Preprocessing Metrics

* original duration
* processed duration
* removed silence duration
* removed silence percentage
* speech regions
* removed non-speech segments

### Acoustic Features

* MFCC mean values
* MFCC standard deviation values
* spectral centroid mean
* spectral centroid standard deviation
* pitch mean
* pitch minimum
* pitch maximum

### Speaker Segmentation Metrics

* segmentation method
* detected speakers
* expected speakers
* speaker speech duration
* speaker segment count
* speaker segments
* central VAD information
* embedding model information
* clustering backend information

### Runtime Metrics

* total processing time
* speaker processing time
* evaluation processing time

### Privacy Metrics

The API response includes:

```json
{
  "raw_audio_stored": false,
  "temporary_files_deleted": true,
  "output_contains_raw_audio": false
}
```

This confirms that the backend returns numerical features and metadata, not raw waveform data.

---

## Example Successful Feature Extraction Result

A short sample was tested successfully.

The system returned:

```text
duration_seconds: 7.19
sample_rate: 16000
rms_energy: 0.059408
clipping_rate: 0.0
processed_duration_seconds: 6.688
removed_silence_percentage: 6.98
pitch_mean_hz: 195.449
raw_audio_stored: false
temporary_files_deleted: true
output_contains_raw_audio: false
```

This confirms that the backend pipeline works for basic acoustic feature extraction.

---

## Docker Support

The backend includes Docker support.

The backend Docker image uses:

```text
python:3.12-slim-bookworm
```

The container installs required system dependencies:

```text
ffmpeg
libsndfile1
```

The backend exposes:

```text
8000
```

and runs the FastAPI server with Uvicorn.

---

## Hugging Face Deployment

The live project is deployed as a Hugging Face Space.

The Space runs:

* FastAPI backend internally on port `8000`
* Streamlit frontend publicly on port `8501`

Public app:

```text
https://gisele-voice-genetics.hf.space
```

The Streamlit frontend communicates with the backend using:

```text
http://127.0.0.1:8000
```

inside the same container.

---

## Privacy Rule

The system must not permanently store uploaded raw audio.

The backend should:

* process audio temporarily
* delete temporary files after processing
* return only numerical features
* avoid returning raw waveforms
* avoid returning personal identity information in JSON output
* avoid storing raw user recordings permanently

---

## Current Limitations

The current system has the following limitations:

* diarization is slower on long audio files
* ECAPA V2 can take several minutes for recordings longer than 2-3 minutes
* diarization performance depends on audio quality and number of speakers
* quiet multi-speaker recordings remain difficult
* false alarms and speaker confusion can still occur
* WavLM is slow on CPU and may over-group audio
* K-Means requires the expected number of speakers
* the current evaluation is frame-based and internal
* future work should add standard DER scoring using `pyannote.metrics`
* `/extract` currently handles both inference and evaluation
* a separate `/evaluate` endpoint should be added later

---

## Future Work

Future work may add:

* separate `/evaluate` endpoint
* RTTM upload support
* standard DER computation using `pyannote.metrics`
* collar-based DER evaluation
* overlap-aware diarization scoring
* stronger noise and SNR estimation
* confidence-based warnings for low-quality recordings
* better speaker-change boundary detection
* improved clustering for difficult multi-speaker files
* pyannote or NeMo diarization baseline
* formants F1, F2, and F3
* jitter
* shimmer
* harmonic-to-noise ratio
* prosodic features
* anonymized voice embedding
* stronger Docker testing
* frontend report download
* improved runtime optimization

---

## Review Evidence

The current implementation demonstrates that:

* the API starts successfully
* `/health` returns OK
* `/extract` accepts supported audio files
* audio is decoded and resampled
* quality metrics are computed
* adaptive VAD is applied
* acoustic features are extracted
* speaker segmentation methods are selectable
* ECAPA V2 speaker diarization works
* speaker-level features are extracted
* VoxConverse RTTM-based evaluation works
* JSON response is returned
* privacy status is included
* the Streamlit frontend runs locally and on Hugging Face
* raw audio is not permanently stored

---

## Team Members

* Sikeh Gisele Wiykiynyuy
* Tivdzua Lubem Noah

---

## Course

Software System Development Using State-of-the-Art Artificial Intelligence Technologies
