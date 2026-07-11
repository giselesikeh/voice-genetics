# Voice Genetics: Acoustic Feature Extraction Pipeline

Voice Genetics is a privacy-compliant acoustic feature extraction system. It accepts audio or video recordings, validates audio quality, preprocesses the signal, performs voice activity detection, optionally performs speaker diarization, extracts numerical acoustic features, and returns a structured JSON response without permanently storing raw audio.

The system does **not** predict genes directly. It prepares standardized acoustic and speaker-level features that may later support research on possible relationships between vocal traits and genetic markers.

## Current Implementation Status

The current implementation is a working FastAPI backend for acoustic feature extraction and speaker-level analysis.

It supports:

* FastAPI backend
* `/health` endpoint
* `/extract` endpoint
* audio and video upload through REST API
* supported formats: `.wav`, `.mp3`, `.m4a`, `.mp4`, and `.mov`
* rejection of unsupported formats
* audio/video decoding with audio-track extraction
* mono conversion
* resampling to 16 kHz
* audio quality metrics
* adaptive voice activity detection
* silence and non-speech region handling
* global acoustic feature extraction
* manual speaker segmentation using `segments_json`
* automatic speaker diarization using ECAPA-TDNN
* improved ECAPA V2 diarization pipeline
* WavLM-based speaker embedding clustering
* speaker-level acoustic feature extraction
* optional diarization evaluation using reference speaker segments
* structured JSON output
* runtime metrics
* privacy status reporting
* Docker backend support
* simple frontend upload interface

## Project Goals

The full project aims to:

* accept voice recordings through a REST API
* support common audio and video formats
* reject unsupported file formats
* validate audio quality before feature extraction
* preprocess audio by converting to mono and resampling to 16 kHz
* remove silence and non-speech regions where possible
* support speaker-level processing
* support automatic speaker diarization
* extract basic and advanced acoustic features
* return privacy-safe JSON output
* avoid permanent raw audio storage
* provide evaluation metrics when reference annotations are available

## Supported File Formats

The backend currently accepts the following file types:

```text
.wav
.mp3
.m4a
.mp4
.mov
```

For video files, only the audio track is processed.

Unsupported formats such as `.txt`, `.pdf`, `.zip`, `.flac`, and `.ogg` are rejected with a clear error message.

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
├── data/
│   ├── raw/
│   ├── samples/
│   ├── processed/
│   └── voxconverse_kaggle/
├── docs/
├── frontend/
├── notebooks/
├── reports/
├── scripts/
├── README.md
└── .gitignore
```

## Backend API

The backend runs using FastAPI.

### Base URL

```text
http://localhost:8000
```

### Main Endpoints

| Endpoint   | Method | Purpose                                           |
| ---------- | ------ | ------------------------------------------------- |
| `/`        | GET    | Returns basic project information                 |
| `/health`  | GET    | Checks whether the API is running                 |
| `/extract` | POST   | Uploads audio/video and returns acoustic features |

## Running Locally

Run all commands from the backend folder:

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

Open the health endpoint in the browser:

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

## Testing Audio Feature Extraction

Keep the API running in one terminal.

Open a second terminal and run from the project root:

```bash
cd "/mnt/f/Innopolis 3 year/Industrial Project/Voice genetics/voice-genetics"
```

Test the `/extract` endpoint with a WAV sample:

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

## Testing Speaker Diarization

The backend supports several speaker segmentation modes:

| Method     | Description                              |
| ---------- | ---------------------------------------- |
| `none`     | No speaker segmentation                  |
| `auto`     | Basic DSP/VAD-based speaker clustering   |
| `ecapa`    | ECAPA-TDNN speaker embedding clustering  |
| `ecapa_v2` | Improved ECAPA-TDNN diarization pipeline |
| `wavlm`    | WavLM speaker embedding clustering       |

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

## Adaptive Voice Activity Detection

The system includes adaptive voice activity detection.

The default VAD mode is:

```text
vad_mode=adaptive
```

Adaptive VAD automatically adjusts the RMS threshold depending on the loudness of the recording.

Default values:

```text
vad_mode=adaptive
vad_top_db=30
vad_min_rms=0.015
vad_min_region_duration_seconds=0.25
vad_merge_gap_seconds=0.8
```

For normal or loud audio, the system keeps the requested RMS threshold.

For quiet audio, the system automatically lowers the effective RMS threshold to avoid missing quiet speech.

Example:

```text
Requested min RMS: 0.015
Effective min RMS: 0.005
Reason: quiet_audio_detected_global_rms_below_0.025_using_min_rms_0.005
```

This was added after testing a quiet VoxConverse recording where strict VAD missed too much speech.

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

When `segments_json` is provided, it overrides the automatic diarization method.

This mode is useful when exact speaker timestamps are known.

## Diarization Evaluation

The API can evaluate diarization if reference speaker segments are provided through `reference_segments_json`.

Important distinction:

* `segments_json` is used as manual speaker segmentation input.
* `reference_segments_json` is used only for evaluation.
* `reference_segments_json` does not affect the automatic prediction.

The evaluation returns frame-based, permutation-invariant diarization metrics.

The returned metrics include:

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

## Current Extracted Metrics

The current backend extracts the following metrics.

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
* basic pitch statistics:

  * mean pitch
  * minimum pitch
  * maximum pitch

### Speaker Segmentation Metrics

* segmentation method
* detected speakers
* expected speakers
* speaker speech duration
* speaker segment count
* speaker segments
* cluster confidence
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

This confirms that the backend returns numerical features only and does not permanently store or return raw audio.

## VoxConverse Evaluation

The ECAPA V2 diarization pipeline was evaluated on a clean VoxConverse Kaggle subset using RTTM ground-truth annotations.

The selected subset contains five recordings:

| File    | Duration | Speakers |
| ------- | -------- | -------- |
| `afjiv` | 151.248s | 5        |
| `akthc` | 114.521s | 2        |
| `ampme` | 148.320s | 3        |
| `asxwr` | 237.767s | 3        |
| `aufkn` | 181.488s | 3        |

### Fixed VAD Results

Initial ECAPA V2 evaluation with fixed VAD produced the following DER values:

| File    | Speakers | DER    |
| ------- | -------- | ------ |
| `afjiv` | 5        | 66.38% |
| `akthc` | 2        | 3.29%  |
| `ampme` | 3        | 6.07%  |
| `asxwr` | 3        | 3.60%  |
| `aufkn` | 3        | 9.14%  |

Average DER over all five files:

```text
17.70%
```

Average DER excluding the difficult quiet outlier `afjiv`:

```text
5.53%
```

### Adaptive VAD Results

Adaptive VAD improved the difficult quiet recording `afjiv`.

| File    | Speakers | Global RMS | Effective RMS | Speech Coverage | DER    |
| ------- | -------- | ---------- | ------------- | --------------- | ------ |
| `afjiv` | 5        | 0.019047   | 0.005         | 0.9540          | 32.91% |
| `akthc` | 2        | 0.058569   | 0.015         | 0.9330          | 3.29%  |
| `ampme` | 3        | 0.050048   | 0.015         | 0.8777          | 6.07%  |
| `asxwr` | 3        | 0.039366   | 0.010         | 0.9945          | 3.60%  |
| `aufkn` | 3        | 0.057042   | 0.015         | 0.9728          | 9.14%  |

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

The remaining errors on `afjiv` are mainly false alarm and speaker confusion, not missed speech.

For the four normal files, ECAPA V2 achieved strong DER values between:

```text
3.29% and 9.14%
```

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

## Docker Support

The backend includes Docker support.

The Docker image uses:

```text
python:3.12-slim-bookworm
```

The container installs required system dependencies such as:

```text
ffmpeg
libsndfile1
```

The backend exposes:

```text
8000
```

and runs the FastAPI server with Uvicorn.

## Frontend

A simple frontend upload interface is included.

The frontend supports:

* selecting an audio file
* uploading it to the backend
* displaying the JSON response
* downloading the JSON response

The frontend is intended as a lightweight demo interface for the backend.

## Privacy Rule

The system must not permanently store uploaded raw audio.

The backend should:

* process audio temporarily
* delete temporary files after processing
* return only numerical features
* avoid returning raw waveforms
* avoid returning personal identity information in JSON output
* avoid storing raw user recordings permanently

## Current Limitations

The current system has the following limitations:

* diarization is slower on long audio files
* ECAPA V2 can take several minutes for recordings longer than 2–3 minutes
* diarization performance depends on audio quality and number of speakers
* quiet multi-speaker recordings remain difficult
* false alarms and speaker confusion can still occur
* the current evaluation is frame-based and internal
* future work should add standard DER scoring using `pyannote.metrics`
* `/extract` currently handles both inference and evaluation
* a separate `/evaluate` endpoint should be added later

## Future Work

The later implementation may add:

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
* stronger frontend demo
* improved runtime optimization

## Review Evidence

The current implementation demonstrates:

* the API starts successfully
* `/health` returns OK
* `/extract` accepts audio and video files
* audio is decoded and resampled
* quality metrics are computed
* adaptive VAD is applied
* acoustic features are extracted
* ECAPA V2 speaker diarization works
* speaker-level features are extracted
* VoxConverse RTTM-based evaluation works
* JSON response is returned
* privacy status is included
* raw audio is not permanently stored

## Team Members

* Sikeh Gisele Wiykiynyuy
* Tivdzua Lubem Noah

## Course

Software System Development Using State-of-the-Art Artificial Intelligence Technologies
