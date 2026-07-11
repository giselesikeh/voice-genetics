
# Voice Genetics Project Plan

Voice Genetics is a Dockerized acoustic feature extraction system. It accepts a voice recording through a REST API, validates the file, preprocesses the audio, removes irrelevant parts such as silence and noise, and extracts numerical voice features.

The system does not directly predict genes. It prepares clean and standardized acoustic features for possible downstream research on relationships between vocal traits and genetic markers.

## Full Project Pipeline

1. User uploads an audio file.
2. API validates the file.
3. Audio is decoded and converted to a standard format.
4. Audio is converted to mono, resampled, and normalized.
5. Silence and non-speech regions are removed.
6. Speaker segmentation is applied for two-speaker recordings.
7. Acoustic features are extracted.
8. Privacy cleanup is performed.
9. JSON response is returned.

## Review Session 2 Target

Before Review Session 2, the goal is to build the first backend foundation:

- FastAPI backend.
- Dockerfile.
- `/health` endpoint.
- `/extract` endpoint.
- WAV upload support.
- Basic validation.
- Basic preprocessing.
- Basic quality metrics.
- Basic acoustic features.
- JSON output.
- README instructions.

## Later Reviews

Later reviews will focus on:

- Full MP3 and M4A support.
- Stronger VAD.
- Speaker segmentation.
- Advanced acoustic features.
- Jitter and shimmer.
- Formants F1, F2, F3.
- HNR.
- Biometric embedding.
- Demo interface.
- Stronger privacy testing.
