from typing import Any, Dict, List
import pandas as pd


def safe_get(data: Dict[str, Any], path: List[str], default: Any = "N/A") -> Any:
    """
    Safely get a nested value from a dictionary.
    """
    current = data

    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def format_seconds(value: Any) -> str:
    """
    Format seconds for display.
    """
    try:
        return f"{float(value):.3f} s"
    except Exception:
        return "N/A"


def speaker_segments_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert speaker segments from JSON into a table.
    """
    segments = safe_get(result, ["speaker_segmentation", "speaker_segments"], [])

    if not segments:
        return pd.DataFrame(columns=["speaker", "start", "end", "duration", "rms"])

    return pd.DataFrame(segments)


def speaker_durations_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert speaker duration dictionary into a table.
    """
    durations = safe_get(
        result,
        ["speaker_segmentation", "speaker_speech_duration_seconds"],
        {},
    )

    if not durations:
        return pd.DataFrame(columns=["speaker", "duration_seconds"])

    rows = [
        {"speaker": speaker, "duration_seconds": duration}
        for speaker, duration in durations.items()
    ]

    return pd.DataFrame(rows)
