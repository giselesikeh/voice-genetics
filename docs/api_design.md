# API Design

## Base URL

When running locally, the API will be available at:

```text
http://localhost:8000
```

## Purpose of the API

The Voice Genetics API receives an uploaded audio file, processes it through the acoustic feature extraction pipeline, and returns a structured JSON response.

The API does not return raw audio. It returns only numerical acoustic features, quality metrics, warnings, speaker segmentation status, and privacy status.

## Main Endpoints

## 1. Root Endpoint

```text
GET /
```

### Purpose

This endpoint returns basic information about the project and confirms that the API is reachable.

### Expected Response

```json
{
  "project": "Voice Genetics",
  "message": "Privacy-compliant acoustic feature extraction API",
  "docs": "/docs"
}
```

## 2. Health Check Endpoint

```text
GET /health
```

### Purpose

This endpoint checks whether the backend service is running correctly.

### Expected Response

```json
{
  "status": "ok",
  "message": "Voice Genetics API is running"
}
```

## 3. Acoustic Feature Extraction Endpoint

```text
POST /extract
```

### Purpose

This endpoint receives one audio file and processes it through the first version of the Voice Genetics pipeline.

The pipeline includes:

1. Audio upload
2. File validation
3. Audio loading
4. Mono conversion
5. Resampling
6. Basic quality assessment
7. Simple silence trimming
8. Basic acoustic feature extraction
9. Privacy status confirmation
10. JSON response generation

### Supported File Types

For the first implementation, the main supported format is:

```text
.wav
```

Later versions will include:

```text
.mp3
.m4a
```

## Example Request Using curl

Run this from the `backend` folder:

```bash
curl -X POST "http://localhost:8000/extract" \
  -F "file=@../data/samples/sample.wav"
```

## Expected JSON Response Structure

```json
{
  "filename": "sample.wav",
  "quality_metrics": {
    "duration_seconds": 10.5,
    "sample_rate": 16000,
    "rms_energy": 0.034521,
    "peak_amplitude": 0.812345,
    "clipping_rate": 0.0
  },
  "preprocessing_metrics": {
    "original_duration_seconds": 10.5,
    "processed_duration_seconds": 8.7,
    "removed_silence_seconds": 1.8,
    "removed_silence_percentage": 17.14,
    "trim_start_sample": 1200,
    "trim_end_sample": 140000
  },
  "speaker_segmentation": {
    "enabled": false,
    "method": "not_implemented_yet",
    "message": "Speaker segmentation is reserved for later reviews.",
    "detected_speakers": null,
    "speaker_segments": []
  },
  "features": {
    "duration_seconds": 8.7,
    "mfcc": {
      "mean": [],
      "std": []
    },
    "spectral_centroid": {
      "mean": 1845.23,
      "std": 321.45
    },
    "pitch": {
      "mean_hz": 145.7,
      "min_hz": 92.4,
      "max_hz": 236.8
    }
  },
  "warnings": [],
  "privacy_status": {
    "raw_audio_stored": false,
    "temporary_files_deleted": true,
    "output_contains_raw_audio": false
  }
}
```

## Explanation of Main Response Fields

## filename

The name of the uploaded audio file.

## quality_metrics

Basic measurements that describe whether the uploaded audio is good enough for feature extraction.

Examples include:

* duration
* sample rate
* RMS energy
* peak amplitude
* clipping rate

## preprocessing_metrics

Information about what happened during audio preprocessing.

Examples include:

* original duration
* processed duration
* removed silence duration
* removed silence percentage

## speaker_segmentation

This field shows the current status of speaker segmentation.

For Review Session 2, speaker segmentation is not fully implemented. The API will return a placeholder showing that this feature is reserved for later reviews.

In later versions, this section will contain speaker-level segments such as Speaker 1 and Speaker 2 timestamps.

## features

This section contains the extracted acoustic features.

For the early implementation, the planned features are:

* MFCC mean and standard deviation
* spectral centroid mean and standard deviation
* basic pitch statistics
* processed audio duration

Later versions will add:

* formants F1, F2, F3
* jitter
* shimmer
* harmonic-to-noise ratio
* prosodic features
* speaker-level features
* anonymized voice fingerprint

## warnings

This section contains messages about possible problems in the uploaded audio.

Examples:

```text
Audio is too short for reliable feature extraction.
Audio may be silent or too quiet.
Audio may contain clipping or distortion.
```

## privacy_status

This section confirms that the system follows the project privacy rule.

Expected values:

```json
{
  "raw_audio_stored": false,
  "temporary_files_deleted": true,
  "output_contains_raw_audio": false
}
```

## Error Handling

The API should return clear errors for invalid inputs.

## Unsupported File Format

If the user uploads an unsupported format, the API should return an error message.

Example:

```json
{
  "detail": "Unsupported file format: .txt. Use .wav, .mp3, or .m4a."
}
```

## Empty Audio File

If the uploaded file is empty, the API should return:

```json
{
  "detail": "Uploaded audio file is empty."
}
```

## Undecodable Audio

If the audio cannot be decoded, the API should return:

```json
{
  "detail": "Could not process audio file."
}
```

## Current API Status for Review Session 2

For Review Session 2, the API target is:

* `GET /` works
* `GET /health` works
* `POST /extract` accepts one WAV file
* audio is loaded and resampled
* basic quality metrics are returned
* silence trimming is applied
* basic features are extracted
* JSON response is returned
* privacy status is included

## Later API Improvements

After Review Session 2, the API can be extended with:

* full MP3 and M4A support
* stronger voice activity detection
* speaker segmentation
* speaker-level acoustic feature extraction
* advanced acoustic features
* biometric embedding
* simple demo interface
* stronger privacy checks
