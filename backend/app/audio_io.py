
import os
import subprocess
import tempfile
from typing import List, Tuple

import librosa
import numpy as np
from fastapi import HTTPException, UploadFile

try:
    from app.config import (
        DEFAULT_SAMPLE_RATE,
        MAX_FILE_SIZE_BYTES,
        MAX_FILE_SIZE_MB,
        SUPPORTED_MEDIA_FORMATS,
        SUPPORTED_VIDEO_FORMATS,
    )
except ImportError:
    DEFAULT_SAMPLE_RATE = 16000
    MAX_FILE_SIZE_MB = 50
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov"]
    SUPPORTED_MEDIA_FORMATS = [".wav", ".mp3", ".m4a", ".mp4", ".mov"]


def _decode_with_ffmpeg(file_path: str, sample_rate: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        file_path,
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        error_message = result.stderr.decode("utf-8", errors="ignore").strip()
        raise ValueError(f"FFmpeg could not decode file. {error_message}")

    audio = np.frombuffer(result.stdout, dtype=np.float32)

    if audio.size == 0:
        raise ValueError("Decoded audio is empty.")

    return audio.astype(np.float32)


async def load_audio_file(file: UploadFile) -> Tuple[np.ndarray, int, List[str]]:
    """
    Validate and decode uploaded audio/video.

    Output format:
    - mono
    - 16 kHz
    - float32 waveform

    For video files, only the audio track is processed.
    Raw files are stored only temporarily during request processing.
    """
    warnings: List[str] = []

    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided.")

    filename = file.filename.lower()
    extension = os.path.splitext(filename)[1]

    if extension not in SUPPORTED_MEDIA_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format: {extension}. "
                f"Use only: {', '.join(SUPPORTED_MEDIA_FORMATS)}."
            ),
        )

    if extension in SUPPORTED_VIDEO_FORMATS:
        warnings.append("Video file accepted. Only the audio track was processed.")

    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB.",
        )

    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=extension) as temp_file:
            temp_file.write(content)
            temp_file.flush()

            try:
                audio = _decode_with_ffmpeg(temp_file.name, DEFAULT_SAMPLE_RATE)
            except Exception as ffmpeg_error:
                warnings.append(
                    "FFmpeg decoding failed; fallback Librosa decoding was used."
                )

                audio, _ = librosa.load(
                    temp_file.name,
                    sr=DEFAULT_SAMPLE_RATE,
                    mono=True,
                )

                if audio.size == 0:
                    raise ValueError(str(ffmpeg_error))

        if audio.ndim != 1:
            audio = np.mean(audio, axis=0).astype(np.float32)

        audio = np.asarray(audio, dtype=np.float32)

        if audio.size == 0:
            raise HTTPException(status_code=400, detail="Audio could not be decoded.")

        if not np.all(np.isfinite(audio)):
            raise HTTPException(
                status_code=400,
                detail="Decoded audio contains non-finite values.",
            )

        return audio, DEFAULT_SAMPLE_RATE, warnings

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process uploaded media file: {str(exc)}",
        )
