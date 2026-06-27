# Voice Genetics: Midterm MVP Documentation

## Project Title

**Voice Genetics: Acoustic Feature Extraction Pipeline**

## Course

Software System Development Using State-of-the-Art Artificial Intelligence Technologies

## Team Members

* Sikeh Gisele Wiykiynyuy
* Tivdzua Lubem Noah

## Midterm Scope

This README documents the **midterm MVP version** of the Voice Genetics project.

The midterm version focuses on a browser-based backend demonstration using the FastAPI Swagger interface at:

```text
http://localhost:8000/docs
```

The midterm MVP demonstrates:

* audio upload through the browser
* audio decoding and validation
* mono conversion and 16 kHz resampling
* voice activity detection and preprocessing
* acoustic feature extraction
* automatic speaker segmentation using Method 2
* automatic speaker segmentation using Method 4
* speaker-level feature extraction
* structured JSON response
* privacy-safe output
* browser-based demonstration evidence

The final diarization method is **not yet presented as the main deliverable** in this midterm version. ECAPA-TDNN is planned for the next review session.

---

## Product Overview

Voice Genetics is a privacy-compliant acoustic feature extraction system. It accepts uploaded audio recordings, validates and preprocesses the waveform, extracts acoustic features, optionally performs automatic speaker segmentation, and returns structured JSON output.

The system does **not** predict genes directly. Instead, it prepares standardized acoustic and speaker-level features that may later support research on possible relationships between voice traits and genetic markers.

The current midterm MVP is a working backend prototype. It can process audio requests from the browser and return machine-readable acoustic and speaker-segmentation results.

---

## Final Product Vision

The final version of the product is expected to:

* accept voice recordings from a user-facing interface
* reject invalid or unsupported files
* convert audio to mono and resample to 16 kHz
* detect speech and non-speech regions
* extract acoustic features such as MFCCs, spectral centroid, pitch, and quality metrics
* support speaker-level segmentation
* return speaker-level acoustic features
* evaluate automatic segmentation against reference annotations during development
* provide clear launch instructions and documentation
* avoid permanent raw audio storage
* return privacy-safe JSON output

---

## Midterm MVP Functionality Implemented

The current midterm MVP includes:

* FastAPI backend
* browser-accessible Swagger documentation at `/docs`
* `GET /health` endpoint
* `POST /extract` endpoint
* audio upload using browser form-data
* audio loading and decoding
* mono conversion
* internal 16 kHz processing
* adaptive voice activity detection
* preprocessing and silence handling
* quality metrics
* MFCC feature extraction
* spectral centroid feature extraction
* pitch statistics
* Method 2 automatic speaker segmentation
* Method 4 WavLM-based speaker segmentation
* speaker-level feature extraction
* runtime metrics
* privacy status reporting
* JSON response output

---

## Supported File Types

The backend currently supports common audio formats such as:

```text
.wav
.mp3
.m4a
```

The backend also has support for video files when an audio track is available:

```text
.mp4
.mov
```

For the midterm demonstration, the tested files are:

```text
test2.m4a
test3.m4a
```

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
├── data/
│   ├── raw/
│   ├── samples/
│   └── processed/
├── docs/
├── frontend/
├── reports/
├── scripts/
├── README.md
├── MIDTERM_README.md
└── .gitignore
```

---

## Running the Midterm MVP Locally

Run all backend commands from:

```bash
cd "/mnt/f/Innopolis 3 year/Industrial Project/Voice genetics/voice-genetics/backend"
```

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Start the FastAPI backend:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Keep this terminal open while using the browser demo.

---

## Browser Demo

Open the Swagger browser interface:

```text
http://localhost:8000/docs
```

The page shows:

* `GET /`
* `GET /health`
* `POST /extract`

---

## Health Check

In the browser:

```text
GET /health
→ Try it out
→ Execute
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

This confirms that the backend server is running.

---

## Audio Extraction Endpoint

The main endpoint is:

```text
POST /extract
```

It accepts an uploaded audio file and returns a JSON response containing:

* filename
* quality metrics
* voice activity information
* preprocessing metrics
* speaker segmentation output
* diarization evaluation status
* runtime metrics
* acoustic features
* speaker-level features
* warnings
* privacy status

---

## Midterm Demonstration Settings

For the midterm, the system is tested using Method 2 and Method 4.

Shared settings:

```text
expected_speakers: 2
chunk_duration_seconds: 2.0
vad_mode: adaptive
vad_top_db: 30
vad_min_rms: 0.015
vad_min_region_duration_seconds: 0.25
vad_merge_gap_seconds: 0.8
evaluation_frame_step_seconds: 0.1
```

The following fields are left empty during automatic inference:

```text
segments_json
reference_segments_json
```

Manual reference segments are used only after the run for interpretation and comparison, not as input during automatic inference.

---

# Implemented Midterm Methods

## Method 2: Automatic VAD + Handcrafted Acoustic Feature Clustering

In the API, Method 2 is selected using:

```text
segmentation_method = auto
```

In the JSON output, this appears as:

```text
vad_simple_speaker_clustering
```

### What Method 2 Means

Method 2 is a lightweight automatic speaker segmentation baseline. It does not use a large pretrained speaker model. Instead, it uses acoustic features extracted from short chunks of the recording.

### How Method 2 Is Implemented

The Method 2 pipeline works as follows:

1. The uploaded audio file is decoded.
2. The signal is converted to mono.
3. The signal is resampled to 16 kHz.
4. Adaptive VAD detects speech regions.
5. Speech regions are divided into fixed-size chunks.
6. Each chunk is represented using handcrafted acoustic features.
7. The features include information such as:

   * RMS energy
   * MFCC-based information
   * spectral information
   * pitch-related information
8. K-Means clustering groups the chunk feature vectors into the expected number of speakers.
9. Cluster labels are converted into anonymous speaker names such as `speaker_1` and `speaker_2`.
10. Adjacent chunks with the same label are merged into longer speaker segments.
11. Speaker-level acoustic features are extracted for each predicted speaker group.

### Why Method 2 Is Used for Midterm

Method 2 is suitable for the midterm MVP because it is:

* lightweight
* easier to run in a browser-based demo
* CPU-friendly compared with deep speech models
* useful as a baseline method
* simple to explain and debug

### Limitations of Method 2

Method 2 is not a final diarization solution.

Its limitations are:

* handcrafted acoustic features are not true speaker embeddings
* pitch or loudness changes may be confused with speaker changes
* fixed-size chunks may cut across speaker boundaries
* K-Means requires the expected number of speakers
* pauses and ignored regions can still be assigned to a speaker
* automatic speaker labels are arbitrary and need manual interpretation

---

## Method 4: WavLM Speaker Embedding Clustering

In the API, Method 4 is selected using:

```text
segmentation_method = wavlm
```

In the JSON output, this appears as:

```text
wavlm_speaker_embedding_clustering
```

### What WavLM Means

WavLM is a pretrained speech representation model. Instead of using only handcrafted acoustic features, it converts audio chunks into learned speech embeddings. These embeddings are then clustered to estimate speaker groups.

### How Method 4 Is Implemented

The Method 4 pipeline works as follows:

1. The uploaded audio file is decoded.
2. The signal is converted to mono.
3. The signal is resampled to 16 kHz.
4. Speech regions are detected.
5. The audio is divided into chunks.
6. Each chunk is passed through a WavLM model.
7. The model output is pooled into one embedding vector per chunk.
8. Embeddings are scaled using a standard scaler.
9. K-Means clustering groups the embeddings into the expected number of speakers.
10. The predicted clusters are converted into anonymous speaker labels.
11. Adjacent chunks with the same speaker label are merged.
12. Speaker-level features are extracted for each predicted speaker group.

### WavLM Model Used

```text
microsoft/wavlm-base-plus
```

### Why Method 4 Is Included

Method 4 is included as a deep-learning comparison method. It helps compare the lightweight Method 2 baseline with a pretrained speech representation approach.

### Limitations of Method 4

Method 4 is not selected as the main midterm MVP method because:

* it is slower on CPU
* it requires loading a large pretrained model
* it can over-group most of the file into one dominant speaker
* WavLM embeddings may capture speech content or acoustic context, not only speaker identity
* it needs better batching, smoothing, and pause handling

---

# Manual Reference Segments

Manual reference segments were prepared for interpretation of the automatic outputs.

These segments are treated as ground truth for checking behavior, but they are not used by Method 2 or Method 4 during automatic inference.

## test2.m4a

Manual speaker 1:

```text
0-11s
16-22s
33-39s
40-42s
46-50s
```

Manual speaker 1 total duration:

```text
29.0s
```

Manual speaker 2:

```text
11-16s
27-33s
39-40s
42-46s
```

Manual speaker 2 total duration:

```text
16.0s
```

Ignored region:

```text
22-27s
```

## test3.m4a

Manual speaker 1:

```text
2-9s
28-39s
49-51s
```

Manual speaker 1 total duration:

```text
20.0s
```

Manual speaker 2:

```text
13-26s
51-58s
```

Manual speaker 2 total duration:

```text
20.0s
```

Ignored regions:

```text
0-2s
9-13s
26-28s
39-49s
after 58s
```

---

# Midterm Sample Results

## test2.m4a: Method 2

Selected API value:

```text
segmentation_method = auto
```

Output method name:

```text
vad_simple_speaker_clustering
```

Quality summary:

```text
duration_seconds: 50.9
sample_rate: 16000
rms_energy: 0.101448
peak_amplitude: 1.008065
clipping_rate: 0.000014
```

Voice activity summary:

```text
method: librosa_split_rms_guard_adaptive
adaptive_vad_enabled: true
adaptive_reason: normal_audio_using_requested_min_rms
speech_duration_seconds: 50.836
speech_coverage_ratio: 0.9987
```

Preprocessing summary:

```text
processed_duration_seconds: 44.02
removed_silence_seconds: 6.816
removed_silence_percentage: 13.41
```

Speaker segmentation output:

```text
detected_speakers: 2
speaker_1 duration: 20s
speaker_2 duration: 30s
speaker_1 segment count: 4
speaker_2 segment count: 4
usable_speech_chunks: 25
```

Runtime:

```text
total_processing_seconds: 94.1329
speaker_processing_seconds: 68.8706
```

Interpretation:

Method 2 produced two speaker groups and returned balanced segment counts. Compared with the manual reference duration pattern, Method 2 is closer than Method 4 after considering that automatic speaker labels are arbitrary.

---

## test2.m4a: Method 4

Selected API value:

```text
segmentation_method = wavlm
```

Output method name:

```text
wavlm_speaker_embedding_clustering
```

Quality summary:

```text
duration_seconds: 50.9
sample_rate: 16000
rms_energy: 0.101448
peak_amplitude: 1.008065
clipping_rate: 0.000014
```

Voice activity summary:

```text
method: librosa_split_rms_guard_adaptive
adaptive_vad_enabled: true
adaptive_reason: normal_audio_using_requested_min_rms
speech_duration_seconds: 50.836
speech_coverage_ratio: 0.9987
```

Preprocessing summary:

```text
processed_duration_seconds: 44.02
removed_silence_seconds: 6.816
removed_silence_percentage: 13.41
```

Speaker segmentation output:

```text
detected_speakers: 2
speaker_1 duration: 44s
speaker_2 duration: 6s
speaker_1 segment count: 3
speaker_2 segment count: 2
usable_speech_chunks: 25
embedding_model: microsoft/wavlm-base-plus
clustering_backend: kmeans
```

Runtime:

```text
total_processing_seconds: 218.0568
speaker_processing_seconds: 198.7153
```

Interpretation:

Method 4 produced two speaker groups, but most of the audio was grouped into one dominant speaker cluster. It also took much longer than Method 2. This shows that WavLM is useful as a comparison method, but it is not the most stable midterm MVP method yet.

---

## test3.m4a: Method 2

Selected API value:

```text
segmentation_method = auto
```

Output method name:

```text
vad_simple_speaker_clustering
```

Quality summary:

```text
duration_seconds: 59.924
sample_rate: 16000
rms_energy: 0.091077
peak_amplitude: 1.009309
clipping_rate: 0.000005
```

Preprocessing summary:

```text
processed_duration_seconds: 59.764
removed_silence_seconds: 0.16
removed_silence_percentage: 0.27
```

Speaker segmentation output:

```text
detected_speakers: 2
speaker_1 duration: 22.0s
speaker_2 duration: 37.924s
speaker_1 segment count: 9
speaker_2 segment count: 9
usable_speech_chunks: 30
```

Interpretation:

Method 2 produced a complete segmentation for test3, but it switched speakers frequently and assigned ignored regions to speakers. This shows that Method 2 is useful for a midterm baseline, but it still needs better pause handling and segment smoothing.

---

## test3.m4a: Method 4

Selected API value:

```text
segmentation_method = wavlm
```

Output method name:

```text
wavlm_speaker_embedding_clustering
```

Quality summary:

```text
duration_seconds: 59.924
sample_rate: 16000
rms_energy: 0.091077
peak_amplitude: 1.009309
clipping_rate: 0.000005
```

Preprocessing summary:

```text
processed_duration_seconds: 59.764
removed_silence_seconds: 0.16
removed_silence_percentage: 0.27
```

Speaker segmentation output:

```text
detected_speakers: 2
speaker_1 duration: 18.0s
speaker_2 duration: 41.924s
speaker_1 segment count: 4
speaker_2 segment count: 4
usable_speech_chunks: 30
embedding_model: microsoft/wavlm-base-plus
```

Interpretation:

Method 4 produced fewer speaker switches than Method 2, but it still over-grouped much of the audio into one speaker. It remains useful as a deep-learning comparison method, but additional smoothing and evaluation are needed before using it as the main system.

---

# Comparison Summary

| Audio     | Manual Reference                    | Method 2 Result                      | Method 4 Result                      | Interpretation                                                                         |
| --------- | ----------------------------------- | ------------------------------------ | ------------------------------------ | -------------------------------------------------------------------------------------- |
| test2.m4a | S1: 29.0s, S2: 16.0s, ignored: 5.0s | speaker_1: 20s, speaker_2: 30s       | speaker_1: 44s, speaker_2: 6s        | Method 2 is closer after label mapping; WavLM over-groups most audio into one cluster. |
| test3.m4a | S1: 20.0s, S2: 20.0s, ignored gaps  | speaker_1: 22.0s, speaker_2: 37.924s | speaker_1: 18.0s, speaker_2: 41.924s | Both methods assign ignored gaps to speakers; Method 2 remains simpler and faster.     |

Important note:

```text
speaker_1 and speaker_2 are anonymous cluster labels.
They do not always correspond directly to manual speaker_1 and manual speaker_2.
```

---

# Acoustic Features Returned

For each uploaded audio file, the backend returns global acoustic features.

The current MVP extracts:

* MFCC mean coefficients
* MFCC standard deviation coefficients
* spectral centroid mean
* spectral centroid standard deviation
* pitch mean
* pitch minimum
* pitch maximum

The backend also extracts speaker-level features for each predicted speaker group.

---

# Privacy Status

The API response includes:

```json
{
  "raw_audio_stored": false,
  "temporary_files_deleted": true,
  "output_contains_raw_audio": false
}
```

This confirms that the backend returns numerical features and metadata, not raw audio.

---

# Difficulties Faced

## Method 2 Difficulties

Method 2 has the following limitations:

* handcrafted acoustic features are not true speaker embeddings
* pitch and energy changes can be mistaken for speaker changes
* fixed-size chunks can include speaker transitions
* K-Means requires the expected number of speakers
* ignored or uncertain regions may still be assigned to speaker clusters
* speaker labels are arbitrary and need manual interpretation

## Method 4 Difficulties

Method 4 has the following limitations:

* WavLM is slower on CPU
* model loading increases processing time
* embeddings may capture speech content instead of only speaker identity
* it can over-group audio into one dominant speaker
* batching and smoothing still need improvement

## Evaluation Difficulties

Evaluation has the following limitations:

* manual reference segments are useful for development but cannot be required from final users
* current comparison is based on duration patterns and segment behavior
* ignored regions can still be assigned as speaker speech
* formal DER-style evaluation is planned for the next stage

---

# Plan for Review Session 3

For Review Session 3, the next steps are:

* introduce ECAPA-TDNN as the next speaker-specific embedding method
* improve VAD so pauses and ignored regions are not forced into speaker clusters
* add smoothing and merging of short speaker segments
* add formal evaluation against manual reference segments
* compute duration error and DER-like metrics
* improve the frontend display for non-technical users
* prepare cleaner browser evidence and demo outputs
* prepare the GitHub repository with all project files needed so far
* add launch instructions for supervisors and the instructor
* add sample inputs and expected outputs to the repository
* continue organizing documentation for later final submission


---

# Midterm Readiness Status

The current MVP is ready for the midterm because:

* the backend runs locally
* the Swagger browser interface opens at `/docs`
* `/health` returns a successful response
* `/extract` accepts audio uploads
* the system processes test2.m4a and test3.m4a
* Method 2 runs as the lightweight automatic segmentation baseline
* Method 4 runs as the WavLM-based deep-learning comparison method
* global acoustic features are returned
* speaker-level features are returned
* JSON output is produced
* privacy status is included
* manual reference segments are available for interpretation
* ECAPA-TDNN is planned for Review Session 3

---

# Main Demo Flow

During the midterm demonstration:

1. Start the FastAPI server.
2. Open `http://localhost:8000/docs`.
3. Run `GET /health`.
4. Open `POST /extract`.
5. Upload `test2.m4a` or `test3.m4a`.
6. Run Method 2 using `segmentation_method=auto`.
7. Show JSON output with quality metrics, VAD, speaker segments, features, and privacy status.
8. Run Method 4 using `segmentation_method=wavlm`.
9. Compare Method 2 and Method 4 outputs.

