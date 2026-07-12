from itertools import combinations, permutations
from typing import Any, Dict, List, Optional

import numpy as np


def _segment_duration(segment: Dict[str, Any]) -> float:
    return max(0.0, float(segment["end"]) - float(segment["start"]))


def _speaker_durations(segments: List[Dict[str, Any]]) -> Dict[str, float]:
    durations: Dict[str, float] = {}
    for segment in segments:
        speaker = str(segment["speaker"])
        durations[speaker] = durations.get(speaker, 0.0) + _segment_duration(segment)
    return {speaker: round(duration, 3) for speaker, duration in durations.items()}


def _speaker_segment_counts(segments: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for segment in segments:
        speaker = str(segment["speaker"])
        counts[speaker] = counts.get(speaker, 0) + 1
    return counts


def _segments_to_frame_labels(
    segments: List[Dict[str, Any]],
    audio_duration_seconds: float,
    frame_step_seconds: float,
) -> List[Optional[str]]:
    frame_count = int(np.ceil(audio_duration_seconds / frame_step_seconds))
    labels: List[Optional[str]] = [None] * frame_count

    for segment in segments:
        speaker = str(segment["speaker"])
        start = max(0.0, float(segment["start"]))
        end = min(audio_duration_seconds, float(segment["end"]))
        if end <= start:
            continue

        start_index = int(np.floor(start / frame_step_seconds))
        end_index = int(np.ceil(end / frame_step_seconds))
        start_index = max(0, start_index)
        end_index = min(frame_count, end_index)

        for index in range(start_index, end_index):
            labels[index] = speaker

    return labels


def _best_speaker_mapping(
    reference_labels: List[Optional[str]],
    predicted_labels: List[Optional[str]],
    frame_step_seconds: float,
) -> Dict[str, Optional[str]]:
    reference_speakers = sorted({label for label in reference_labels if label is not None})
    predicted_speakers = sorted({label for label in predicted_labels if label is not None})

    if not reference_speakers or not predicted_speakers:
        return {speaker: None for speaker in predicted_speakers}

    overlap: Dict[str, Dict[str, float]] = {
        predicted: {reference: 0.0 for reference in reference_speakers}
        for predicted in predicted_speakers
    }

    for ref_label, pred_label in zip(reference_labels, predicted_labels):
        if ref_label is not None and pred_label is not None:
            overlap[pred_label][ref_label] += frame_step_seconds

    best_mapping: Dict[str, Optional[str]] = {speaker: None for speaker in predicted_speakers}
    best_score = -1.0

    if len(predicted_speakers) <= len(reference_speakers):
        for reference_assignment in permutations(reference_speakers, len(predicted_speakers)):
            candidate_mapping = {
                predicted: reference
                for predicted, reference in zip(predicted_speakers, reference_assignment)
            }
            score = sum(
                overlap[predicted][reference]
                for predicted, reference in candidate_mapping.items()
            )
            if score > best_score:
                best_score = score
                best_mapping = candidate_mapping
    else:
        for predicted_subset in combinations(predicted_speakers, len(reference_speakers)):
            for reference_assignment in permutations(reference_speakers):
                candidate_mapping = {speaker: None for speaker in predicted_speakers}
                for predicted, reference in zip(predicted_subset, reference_assignment):
                    candidate_mapping[predicted] = reference
                score = sum(
                    overlap[predicted][reference]
                    for predicted, reference in candidate_mapping.items()
                    if reference is not None
                )
                if score > best_score:
                    best_score = score
                    best_mapping = candidate_mapping

    return best_mapping


def _mapped_predicted_durations(
    predicted_segments: List[Dict[str, Any]],
    speaker_mapping: Dict[str, Optional[str]],
) -> Dict[str, float]:
    mapped_durations: Dict[str, float] = {}
    for segment in predicted_segments:
        predicted_speaker = str(segment["speaker"])
        reference_speaker = speaker_mapping.get(predicted_speaker)
        if reference_speaker is None:
            continue
        mapped_durations[reference_speaker] = (
            mapped_durations.get(reference_speaker, 0.0) + _segment_duration(segment)
        )
    return {speaker: round(duration, 3) for speaker, duration in mapped_durations.items()}


def _duration_errors(
    reference_segments: List[Dict[str, Any]],
    predicted_segments: List[Dict[str, Any]],
    speaker_mapping: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    reference_durations = _speaker_durations(reference_segments)
    predicted_mapped_durations = _mapped_predicted_durations(
        predicted_segments,
        speaker_mapping,
    )

    per_speaker: Dict[str, Dict[str, float]] = {}
    for speaker, reference_duration in reference_durations.items():
        predicted_duration = predicted_mapped_durations.get(speaker, 0.0)
        signed_error = predicted_duration - reference_duration
        absolute_error = abs(signed_error)
        per_speaker[speaker] = {
            "reference_duration_seconds": round(reference_duration, 3),
            "mapped_predicted_duration_seconds": round(predicted_duration, 3),
            "signed_error_seconds": round(signed_error, 3),
            "absolute_error_seconds": round(absolute_error, 3),
        }

    total_reference = sum(reference_durations.values())
    total_predicted_mapped = sum(predicted_mapped_durations.values())

    return {
        "per_speaker": per_speaker,
        "total_reference_duration_seconds": round(total_reference, 3),
        "total_mapped_predicted_duration_seconds": round(total_predicted_mapped, 3),
        "total_signed_error_seconds": round(total_predicted_mapped - total_reference, 3),
        "total_absolute_error_seconds": round(abs(total_predicted_mapped - total_reference), 3),
    }


def _boundary_times(segments: List[Dict[str, Any]]) -> List[float]:
    if len(segments) < 2:
        return []

    sorted_segments = sorted(segments, key=lambda segment: float(segment["start"]))
    boundaries: List[float] = []

    for previous_segment, next_segment in zip(sorted_segments, sorted_segments[1:]):
        previous_speaker = str(previous_segment["speaker"])
        next_speaker = str(next_segment["speaker"])
        previous_end = float(previous_segment["end"])
        next_start = float(next_segment["start"])
        if previous_speaker != next_speaker:
            boundaries.append(round((previous_end + next_start) / 2.0, 3))

    return boundaries


def _boundary_metrics(
    reference_segments: List[Dict[str, Any]],
    predicted_segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    reference_boundaries = _boundary_times(reference_segments)
    predicted_boundaries = _boundary_times(predicted_segments)

    if not reference_boundaries or not predicted_boundaries:
        return {
            "reference_boundary_count": len(reference_boundaries),
            "predicted_boundary_count": len(predicted_boundaries),
            "mean_nearest_boundary_error_seconds": None,
            "median_nearest_boundary_error_seconds": None,
            "max_nearest_boundary_error_seconds": None,
            "nearest_boundary_errors_seconds": [],
        }

    errors = []
    for reference_boundary in reference_boundaries:
        nearest_error = min(
            abs(reference_boundary - predicted_boundary)
            for predicted_boundary in predicted_boundaries
        )
        errors.append(nearest_error)

    return {
        "reference_boundary_count": len(reference_boundaries),
        "predicted_boundary_count": len(predicted_boundaries),
        "mean_nearest_boundary_error_seconds": round(float(np.mean(errors)), 3),
        "median_nearest_boundary_error_seconds": round(float(np.median(errors)), 3),
        "max_nearest_boundary_error_seconds": round(float(np.max(errors)), 3),
        "nearest_boundary_errors_seconds": [round(float(error), 3) for error in errors],
    }


def _annotation_from_segments(segments: List[Dict[str, Any]], uri: str):
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=uri)
    for index, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        speaker = str(segment["speaker"])
        if end <= start:
            continue
        annotation[Segment(start, end), f"track_{index}"] = speaker
    return annotation


def _try_pyannote_der(
    reference_segments: List[Dict[str, Any]],
    predicted_segments: List[Dict[str, Any]],
    collar_seconds: float,
    skip_overlap: bool,
) -> Dict[str, Any]:
    """
    Optional standard DER using pyannote.metrics.

    This does not break the project when pyannote.metrics is not installed.
    """
    try:
        from pyannote.metrics.diarization import DiarizationErrorRate
    except Exception as exc:
        return {
            "enabled": False,
            "available": False,
            "message": (
                "pyannote.metrics is not installed. Install it later with: "
                "pip install pyannote.metrics"
            ),
            "error": str(exc),
        }

    try:
        reference = _annotation_from_segments(reference_segments, uri="reference")
        hypothesis = _annotation_from_segments(predicted_segments, uri="prediction")
        metric = DiarizationErrorRate(collar=collar_seconds, skip_overlap=skip_overlap)
        der_value = abs(float(metric(reference, hypothesis)))
        components = metric.compute_components(reference, hypothesis)
        components_clean = {
            str(key): round(float(value), 4)
            for key, value in dict(components).items()
        }

        return {
            "enabled": True,
            "available": True,
            "method": "pyannote.metrics.DiarizationErrorRate",
            "collar_seconds": collar_seconds,
            "skip_overlap": skip_overlap,
            "der": round(der_value, 4),
            "der_percent": round(der_value * 100.0, 2),
            "components": components_clean,
            "important_note": (
                "This is the standard pyannote.metrics DER. The internal frame-based DER "
                "is still returned for comparison and debugging."
            ),
        }
    except Exception as exc:
        return {
            "enabled": False,
            "available": True,
            "message": "pyannote.metrics is installed, but standard DER computation failed.",
            "error": str(exc),
        }


def evaluate_diarization(
    reference_segments: List[Dict[str, Any]],
    predicted_segments: List[Dict[str, Any]],
    audio_duration_seconds: float,
    frame_step_seconds: float = 0.1,
    use_pyannote_metrics: bool = False,
    collar_seconds: float = 0.25,
    skip_overlap: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate automatic diarization against reference segments.

    The default evaluation is frame-based and permutation-invariant. If requested,
    the function also tries to compute standard pyannote.metrics DER.
    """
    if frame_step_seconds <= 0:
        raise ValueError("frame_step_seconds must be greater than zero.")
    if audio_duration_seconds <= 0:
        raise ValueError("audio_duration_seconds must be greater than zero.")

    reference_labels = _segments_to_frame_labels(
        reference_segments,
        audio_duration_seconds,
        frame_step_seconds,
    )
    predicted_labels = _segments_to_frame_labels(
        predicted_segments,
        audio_duration_seconds,
        frame_step_seconds,
    )

    speaker_mapping = _best_speaker_mapping(
        reference_labels,
        predicted_labels,
        frame_step_seconds,
    )

    reference_speech_frames = 0
    predicted_speech_frames = 0
    overlapped_speech_frames = 0
    correct_speaker_frames = 0
    missed_speech_frames = 0
    false_alarm_frames = 0
    speaker_confusion_frames = 0
    correct_non_speech_frames = 0
    reference_non_speech_frames = 0

    for reference_label, predicted_label in zip(reference_labels, predicted_labels):
        reference_is_speech = reference_label is not None
        predicted_is_speech = predicted_label is not None

        if reference_is_speech:
            reference_speech_frames += 1
        else:
            reference_non_speech_frames += 1

        if predicted_is_speech:
            predicted_speech_frames += 1

        if reference_is_speech and predicted_is_speech:
            overlapped_speech_frames += 1
            mapped_prediction = speaker_mapping.get(predicted_label)
            if mapped_prediction == reference_label:
                correct_speaker_frames += 1
            else:
                speaker_confusion_frames += 1
        elif reference_is_speech and not predicted_is_speech:
            missed_speech_frames += 1
        elif predicted_is_speech and not reference_is_speech:
            false_alarm_frames += 1
        elif not reference_is_speech and not predicted_is_speech:
            correct_non_speech_frames += 1

    reference_speech_seconds = reference_speech_frames * frame_step_seconds
    predicted_speech_seconds = predicted_speech_frames * frame_step_seconds
    overlapped_speech_seconds = overlapped_speech_frames * frame_step_seconds
    correct_speaker_seconds = correct_speaker_frames * frame_step_seconds
    missed_speech_seconds = missed_speech_frames * frame_step_seconds
    false_alarm_seconds = false_alarm_frames * frame_step_seconds
    speaker_confusion_seconds = speaker_confusion_frames * frame_step_seconds
    reference_non_speech_seconds = reference_non_speech_frames * frame_step_seconds
    correct_non_speech_seconds = correct_non_speech_frames * frame_step_seconds

    diarization_error_seconds = (
        missed_speech_seconds + false_alarm_seconds + speaker_confusion_seconds
    )

    der = (
        diarization_error_seconds / reference_speech_seconds
        if reference_speech_seconds > 0
        else None
    )
    speech_precision = (
        overlapped_speech_seconds / predicted_speech_seconds
        if predicted_speech_seconds > 0
        else None
    )
    speech_recall = (
        overlapped_speech_seconds / reference_speech_seconds
        if reference_speech_seconds > 0
        else None
    )
    speaker_assignment_accuracy_on_overlap = (
        correct_speaker_seconds / overlapped_speech_seconds
        if overlapped_speech_seconds > 0
        else None
    )
    speaker_assignment_accuracy_on_reference = (
        correct_speaker_seconds / reference_speech_seconds
        if reference_speech_seconds > 0
        else None
    )
    pause_detection_accuracy = (
        correct_non_speech_seconds / reference_non_speech_seconds
        if reference_non_speech_seconds > 0
        else None
    )

    result = {
        "enabled": True,
        "method": "frame_based_permutation_invariant_diarization_evaluation",
        "frame_step_seconds": frame_step_seconds,
        "audio_duration_seconds": round(audio_duration_seconds, 3),
        "speaker_mapping_predicted_to_reference": speaker_mapping,
        "reference_speaker_durations_seconds": _speaker_durations(reference_segments),
        "predicted_speaker_durations_seconds": _speaker_durations(predicted_segments),
        "reference_segment_count": len(reference_segments),
        "predicted_segment_count": len(predicted_segments),
        "reference_speaker_segment_count": _speaker_segment_counts(reference_segments),
        "predicted_speaker_segment_count": _speaker_segment_counts(predicted_segments),
        "time_metrics_seconds": {
            "reference_speech": round(reference_speech_seconds, 3),
            "predicted_speech": round(predicted_speech_seconds, 3),
            "overlapped_speech": round(overlapped_speech_seconds, 3),
            "correct_speaker": round(correct_speaker_seconds, 3),
            "missed_speech": round(missed_speech_seconds, 3),
            "false_alarm": round(false_alarm_seconds, 3),
            "speaker_confusion": round(speaker_confusion_seconds, 3),
            "diarization_error": round(diarization_error_seconds, 3),
            "reference_non_speech": round(reference_non_speech_seconds, 3),
            "correct_non_speech": round(correct_non_speech_seconds, 3),
        },
        "rate_metrics": {
            "der": round(der, 4) if der is not None else None,
            "der_percent": round(der * 100.0, 2) if der is not None else None,
            "speech_precision": round(speech_precision, 4)
            if speech_precision is not None
            else None,
            "speech_recall": round(speech_recall, 4) if speech_recall is not None else None,
            "speaker_assignment_accuracy_on_overlap": round(
                speaker_assignment_accuracy_on_overlap,
                4,
            )
            if speaker_assignment_accuracy_on_overlap is not None
            else None,
            "speaker_assignment_accuracy_on_reference": round(
                speaker_assignment_accuracy_on_reference,
                4,
            )
            if speaker_assignment_accuracy_on_reference is not None
            else None,
            "pause_detection_accuracy": round(pause_detection_accuracy, 4)
            if pause_detection_accuracy is not None
            else None,
        },
        "speaker_duration_error": _duration_errors(
            reference_segments,
            predicted_segments,
            speaker_mapping,
        ),
        "boundary_metrics": _boundary_metrics(reference_segments, predicted_segments),
        "standard_der_pyannote": {
            "enabled": False,
            "available": False,
            "message": "Not requested. Set use_pyannote_metrics=true to try standard DER.",
        },
        "important_note": (
            "DER combines missed speech, false alarm, and speaker confusion. "
            "Speaker labels are matched before scoring, so predicted speaker_1 does not "
            "need to equal reference speaker_1."
        ),
    }

    if use_pyannote_metrics:
        result["standard_der_pyannote"] = _try_pyannote_der(
            reference_segments=reference_segments,
            predicted_segments=predicted_segments,
            collar_seconds=collar_seconds,
            skip_overlap=skip_overlap,
        )

    return result
