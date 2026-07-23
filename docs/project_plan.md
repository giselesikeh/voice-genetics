# Voice Genetics Project Plan

Voice Genetics is a Dockerized, privacy-aware acoustic feature extraction and speaker diarization system. It accepts a voice recording through a REST API or browser interface, validates the uploaded file, preprocesses the audio, detects speech regions, extracts global and speaker-level acoustic features, optionally performs speaker diarization, evaluates diarization when reference labels are available, and returns a structured JSON response.

The system does **not** directly predict genes. It prepares clean, standardized numerical voice features that may support future research on possible relationships between vocal traits and genetic markers.

---

## 1. Final Product Goal

The final goal of Voice Genetics is to provide a working software product that can:

- accept voice recordings from users;
- process `.wav`, `.mp3`, and `.m4a` files;
- validate file type and audio quality;
- convert audio to a standard processing format;
- detect speech and non-speech regions;
- extract basic and advanced acoustic features;
- perform optional speaker diarization;
- evaluate diarization using reference labels when available;
- display results through a frontend user interface;
- return privacy-safe structured JSON;
- run as a standalone Docker container.

---

## 2. Full Final Project Pipeline

The final project pipeline is:

```text
User audio upload
    ↓
File validation
    ↓
Audio decoding
    ↓
Mono conversion
    ↓
16 kHz resampling
    ↓
Audio quality analysis
    ↓
Adaptive voice activity detection
    ↓
Speech-region extraction
    ↓
Global acoustic feature extraction
    ↓
Optional speaker diarization
    ↓
Speaker-level acoustic feature extraction
    ↓
Optional reference-based diarization evaluation
    ↓
Runtime and privacy reporting
    ↓
Structured JSON response
    ↓
Frontend display
```

---

## 3. Implemented Components

The final implementation includes:

- FastAPI backend;
- Streamlit frontend;
- Dockerized full-stack deployment;
- `/health` endpoint;
- `/extract` endpoint;
- file upload through API and frontend;
- support for `.wav`, `.mp3`, and `.m4a`;
- unsupported-format rejection;
- audio decoding;
- mono conversion;
- 16 kHz resampling;
- audio quality metrics;
- adaptive voice activity detection;
- preprocessing and silence handling;
- global acoustic feature extraction;
- speaker-level acoustic feature extraction;
- advanced acoustic feature extraction;
- manual speaker segmentation support;
- reference-based diarization evaluation;
- multiple speaker segmentation methods;
- clustering quality metrics;
- bootstrap evaluation;
- privacy status reporting;
- raw JSON output display.

---

## 4. Backend Plan

The backend is responsible for audio processing, feature extraction, diarization, evaluation, and JSON response generation.

Main backend modules:

| Module | Purpose |
|---|---|
| `main.py` | FastAPI routes and request handling |
| `audio_io.py` | Audio validation, decoding, mono conversion, and resampling |
| `quality.py` | Audio quality metrics |
| `vad.py` | Voice activity detection and adaptive VAD |
| `features.py` | Basic acoustic feature extraction |
| `advanced_features.py` | Formants, jitter, shimmer, HNR, speaking-rate proxy, and VOT proxy |
| `speaker_segmentation.py` | Speaker diarization methods |
| `evaluation.py` | DER-style diarization evaluation |
| `privacy.py` | Privacy and temporary-file cleanup reporting |
| `schemas.py` | Structured response models |

---

## 5. Frontend Plan

The frontend is built with Streamlit and provides a browser-based interface for users.

The frontend supports:

- audio upload;
- backend URL configuration;
- segmentation method selection;
- expected speaker count input;
- VAD setting controls;
- ECAPA-specific controls;
- optional manual/reference segment input;
- quality metric display;
- VAD summary display;
- speaker segmentation display;
- speaker duration tables;
- speaker segment tables;
- clustering metric display;
- acoustic feature display;
- advanced acoustic feature display;
- privacy status display;
- raw JSON display.

---

## 6. Speaker Segmentation Methods

The project includes several speaker segmentation methods.

### Method 1: Acoustic Features Only

```text
segmentation_method = none
```

This mode extracts only global acoustic features and does not perform speaker diarization.

### Method 2: Handcrafted DSP Features + K-Means

```text
segmentation_method = auto
```

This method uses handcrafted acoustic features such as RMS, MFCCs, spectral centroid, and pitch-related values, then clusters them using K-Means.

This method is lightweight but less speaker-specific because handcrafted features can change with loudness, pitch, and speaking style.

### Method 3: ECAPA-TDNN Speaker Embeddings

```text
segmentation_method = ecapa
```

This method uses ECAPA-TDNN speaker embeddings and clustering to group audio chunks by speaker identity.

It improves over Method 2 because ECAPA embeddings are more speaker-specific than handcrafted acoustic features.

### Method 3B: Improved ECAPA-TDNN Pipeline

```text
segmentation_method = ecapa_v2
```

This is the recommended final diarization method.

Method 3B includes:

- adaptive VAD;
- overlapping speech chunks;
- ECAPA-TDNN embeddings;
- L2 embedding normalization;
- agglomerative cosine clustering;
- smoothing of isolated label changes;
- short-segment merging;
- speaker-level acoustic features;
- clustering metrics;
- DER-style evaluation when reference labels are provided.

### Method 4: WavLM Embeddings + K-Means

```text
segmentation_method = wavlm
```

This method uses WavLM hidden states, pooled embeddings, and K-Means clustering.

It was implemented as a comparison baseline, but it is slower and less diarization-specific than Method 3B.

---

## 7. Acoustic Features Plan

The final system extracts both basic and advanced acoustic features.

### Basic Features

- duration;
- RMS energy;
- peak amplitude;
- clipping rate;
- MFCC mean;
- MFCC standard deviation;
- spectral centroid mean;
- spectral centroid standard deviation;
- pitch mean;
- pitch minimum;
- pitch maximum.

### Advanced Features

The final system also extracts:

- formants F1, F2, F3;
- jitter;
- shimmer;
- harmonic-to-noise ratio;
- speaking-rate proxy;
- voice-onset-time proxy.

Important note:

Some advanced features are approximate signal-based estimates. True clinical-grade jitter, shimmer, HNR, speaking rate, and VOT would require tools such as Praat, transcripts, syllable labels, or phoneme-level forced alignment.

---

## 8. Dataset Plan

The main evaluation dataset is VoxConverse.

VoxConverse was used because it provides multi-speaker conversational audio and RTTM speaker annotations.

Final selected subset:

| Criterion | Value |
|---|---|
| Dataset | VoxConverse development set |
| Speaker count | 2 or 3 speakers |
| Maximum duration | under 10 minutes |
| Reference labels | RTTM available |
| Number of selected files | 73 |

Dataset split:

| Group | Number of files |
|---|---:|
| 2-speaker recordings | 39 |
| 3-speaker recordings | 34 |
| Total | 73 |

The raw dataset is not committed to GitHub or packaged inside Docker because of size and dataset restrictions. The repository includes scripts for sample selection, RTTM conversion, and evaluation.

---

## 9. Evaluation Plan

The evaluation focuses on diarization quality, clustering quality, acoustic-feature completeness, runtime, and privacy.

### Diarization Metrics

When reference labels are available, the system reports:

- DER;
- missed speech;
- false alarm;
- speaker confusion;
- speech precision;
- speech recall;
- speaker assignment accuracy;
- boundary metrics.

### Clustering Metrics

The system reports internal clustering quality metrics:

- silhouette score;
- cluster balance ratio;
- speaker switches;
- segment smoothness;
- mean segment duration.

### Acoustic Feature Reporting

The system reports:

- global acoustic features;
- speaker-level acoustic features;
- advanced acoustic features.

### Bootstrap Evaluation

The final evaluation uses bootstrapping instead of cross-validation because the project evaluates a fixed pipeline rather than training a supervised model.

Bootstrap process:

```text
1. Run Method 3B on all 73 selected files.
2. Save one metric row per file.
3. Randomly sample 73 files with replacement.
4. Compute average metrics for that sampled set.
5. Repeat 1000 times.
6. Report mean and standard deviation of the bootstrap averages.
```

---

## 10. Final Evaluation Results

Final Method 3B results on the 73-file VoxConverse subset:

| Metric | Result |
|---|---:|
| Number of files | 73 |
| Bootstrap mean DER | 21.44% |
| Bootstrap std DER | 6.37% |
| Median per-file DER | 8.02% |
| Micro DER over all frames | 15.53% |

DER distribution:

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

Main interpretation:

The system performs well on many recordings, but the mean DER is increased by difficult outlier files. The main remaining diarization errors are speaker confusion, false alarms, and cluster imbalance.

---

## 11. Dockerization Plan

The final project is Dockerized as a standalone full-stack container.

The Docker container starts:

```text
FastAPI backend  → port 8000
Streamlit UI     → port 8501
```

Docker files:

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the final full-stack container |
| `start.sh` | Starts the backend first, waits for health check, then starts Streamlit |
| `.dockerignore` | Excludes datasets, virtual environments, model caches, and large files |

Docker build command:

```bash
docker build --progress=plain -t voice-genetics-final .
```

Docker run command:

```bash
docker run --rm \
  -p 8000:8000 \
  -p 8501:8501 \
  -v voice-genetics-cache:/app/.cache \
  voice-genetics-final
```

Frontend:

```text
http://localhost:8501
```

Backend health check:

```text
http://localhost:8000/health
```

---

## 12. Privacy Plan

The project follows a privacy-aware design.

Privacy behavior:

- uploaded audio is processed temporarily;
- raw audio is not permanently stored;
- temporary files are deleted after processing;
- raw waveform data is not returned;
- output contains numerical features and metadata only;
- speaker labels are anonymous cluster labels such as `speaker_1` and `speaker_2`.

The response includes privacy status fields such as:

```json
{
  "raw_audio_stored": false,
  "temporary_files_deleted": true,
  "output_contains_raw_audio": false
}
```

---

## 13. Incorrect and Unexpected User Actions

The system is designed to handle:

- unsupported file formats;
- missing files;
- invalid JSON input;
- invalid manual/reference segment timestamps;
- reference segments beyond the audio duration;
- quiet audio;
- low-confidence clustering;
- cluster imbalance;
- missing or incorrect expected speaker counts;
- long-running audio files.

---

## 14. Review Session Progress

### Review Session 1

Focus:

- initial project idea;
- problem definition;
- expected pipeline;
- privacy motivation;
- first backend planning.

### Review Session 2

Focus:

- FastAPI backend foundation;
- `/health` endpoint;
- `/extract` endpoint;
- WAV support;
- basic preprocessing;
- basic quality metrics;
- basic acoustic features;
- initial Docker support.

### Midterm

Focus:

- improved backend;
- Streamlit frontend;
- more audio formats;
- Method 2 and Method 4 comparison;
- speaker segmentation demonstration;
- early diarization and feature outputs.

### Review Session 3

Focus:

- Method 3: ECAPA-TDNN speaker embeddings + K-Means;
- VoxConverse dataset usage;
- RTTM reference evaluation;
- DER-style results;
- clustering metrics;
- acoustic feature reporting.

### Review Session 4

Focus:

- Method 3B improved ECAPA pipeline;
- adaptive VAD;
- overlapping chunks;
- agglomerative cosine clustering;
- smoothing;
- short-segment merging;
- improved clustering metrics;
- professor sample testing;
- preparation for final evaluation.

### Final Stage

Focus:

- 73-file VoxConverse evaluation;
- bootstrap mean and standard deviation;
- advanced acoustic features;
- full frontend display;
- final README;
- launch instructions;
- standalone Docker container;
- final report and presentation.

---

## 15. Main Difficulties Faced

Main difficulties included:

1. Speaker diarization was harder than basic audio feature extraction.
2. Handcrafted DSP features were not speaker-specific enough.
3. WavLM was slower and less stable for diarization than expected.
4. Fixed VAD thresholds failed on quiet recordings.
5. Adaptive VAD was needed to preserve quiet speech.
6. K-Means required the expected number of speakers.
7. Some files had strong cluster imbalance.
8. Small speakers were sometimes collapsed during smoothing or merging.
9. Speaker confusion remained the main error source in the final evaluation.
10. False alarms remained on pause-heavy files.
11. Runtime was high for long audio files.
12. Some acoustic features required approximate proxy estimation because transcripts or phoneme-level labels were unavailable.
13. Docker build initially failed because large datasets, virtual environments, and pretrained model folders were included in the Docker context.
14. Docker WSL integration needed to be fixed before building from VS Code.

---

## 16. Current Limitations

Current limitations:

- diarization is slow on long recordings;
- overlap speech is not fully separated;
- speaker confusion remains on difficult recordings;
- cluster imbalance remains challenging;
- minority speakers can be collapsed;
- true speaking rate needs transcripts;
- true VOT needs phoneme alignment;
- jitter, shimmer, and HNR are approximate estimates;
- pyannote.metrics DER is not yet integrated;
- external validation on a second dataset is still future work.

---

## 17. Future Work

Future work includes:

- external validation on another diarization dataset;
- standard DER calculation with `pyannote.metrics`;
- collar-based DER scoring;
- overlap-aware diarization evaluation;
- stronger speaker-count estimation;
- better boundary refinement;
- stronger pause/noise rejection;
- more efficient model inference;
- batching ECAPA embedding extraction;
- transcript-based speaking rate;
- forced-alignment-based VOT;
- Praat/parselmouth-based formants, jitter, shimmer, and HNR;
- downloadable frontend reports;
- separate `/evaluate` endpoint for reference-based scoring.

---

## 18. Final Summary

Voice Genetics is now a complete working software product. It provides a FastAPI backend, Streamlit frontend, Docker deployment, multiple speaker segmentation methods, acoustic feature extraction, advanced acoustic feature extraction, speaker-level analysis, reference-based diarization evaluation, and bootstrap-based final reporting.

The final system can be launched locally or through Docker, processes supported audio files, displays results in a user interface, returns structured JSON, and reports privacy status.

The project is successful as a full acoustic feature extraction and diarization pipeline, while future work should focus on improving diarization robustness, reducing runtime, and validating on an additional dataset.
