import streamlit as st
import pandas as pd

from utils.formatting import (
    safe_get,
    format_seconds,
    speaker_segments_to_dataframe,
    speaker_durations_to_dataframe,
)


def render_results(result: dict) -> None:
    """
    Render analysis results from backend JSON.
    """
    st.header("Analysis Results")

    tab_summary, tab_quality, tab_speaker, tab_features, tab_privacy, tab_json = st.tabs(
        [
            "Summary",
            "Quality & VAD",
            "Speaker Segmentation",
            "Acoustic Features",
            "Privacy",
            "Raw JSON",
        ]
    )

    with tab_summary:
        render_summary_tab(result)

    with tab_quality:
        render_quality_tab(result)

    with tab_speaker:
        render_speaker_tab(result)

    with tab_features:
        render_features_tab(result)

    with tab_privacy:
        render_privacy_tab(result)

    with tab_json:
        st.json(result)


def render_summary_tab(result: dict) -> None:
    filename = safe_get(result, ["filename"])
    method = safe_get(result, ["speaker_segmentation", "method"], "not used")
    detected_speakers = safe_get(result, ["speaker_segmentation", "detected_speakers"], "N/A")
    total_time = safe_get(result, ["runtime_metrics", "total_processing_seconds"], "N/A")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("File", filename)
    col2.metric("Method", method)
    col3.metric("Detected Speakers", detected_speakers)

    try:
        col4.metric("Processing Time", f"{float(total_time):.2f} s")
    except Exception:
        col4.metric("Processing Time", "N/A")

    st.info(
        "This MVP extracts acoustic features and optionally performs baseline speaker segmentation. "
        "It does not predict genes directly."
    )


def render_quality_tab(result: dict) -> None:
    st.subheader("Quality Metrics")

    quality = result.get("quality_metrics", {})
    if quality:
        st.dataframe(pd.DataFrame([quality]), use_container_width=True)
    else:
        st.warning("No quality metrics returned.")

    st.subheader("Voice Activity / Preprocessing")

    voice_activity = result.get("voice_activity", {})
    preprocessing = result.get("preprocessing_metrics", {})

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Speech Duration",
        format_seconds(voice_activity.get("speech_duration_seconds", "N/A")),
    )
    col2.metric(
        "Non-Speech Duration",
        format_seconds(voice_activity.get("non_speech_duration_seconds", "N/A")),
    )
    col3.metric(
        "Speech Coverage",
        voice_activity.get("speech_coverage_ratio", "N/A"),
    )

    if preprocessing:
        st.write("Preprocessing details")
        st.dataframe(pd.DataFrame([preprocessing]).drop(columns=[
            col for col in ["speech_regions", "removed_non_speech_segments"]
            if col in preprocessing
        ]), use_container_width=True)


def render_speaker_tab(result: dict) -> None:
    speaker_info = result.get("speaker_segmentation", {})

    if not speaker_info or not speaker_info.get("enabled", False):
        st.warning("Speaker segmentation was not enabled for this run.")
        return

    st.subheader("Speaker Segmentation Summary")

    summary = {
        "method": speaker_info.get("method"),
        "detected_speakers": speaker_info.get("detected_speakers"),
        "expected_speakers": speaker_info.get("expected_speakers"),
        "chunk_duration_seconds": speaker_info.get("chunk_duration_seconds"),
        "usable_speech_chunks": speaker_info.get("usable_speech_chunks"),
        "removed_pause_chunks": speaker_info.get("removed_pause_chunks"),
    }

    st.dataframe(pd.DataFrame([summary]), use_container_width=True)

    st.subheader("Speaker Durations")

    durations_df = speaker_durations_to_dataframe(result)
    if not durations_df.empty:
        st.bar_chart(durations_df.set_index("speaker"))
        st.dataframe(durations_df, use_container_width=True)
    else:
        st.info("No speaker duration table available.")

    st.subheader("Speaker Segments")

    segments_df = speaker_segments_to_dataframe(result)
    if not segments_df.empty:
        st.dataframe(segments_df, use_container_width=True)
    else:
        st.info("No speaker segments available.")


def render_features_tab(result: dict) -> None:
    features = result.get("features", {})

    if not features:
        st.warning("No acoustic features returned.")
        return

    st.subheader("Global Acoustic Features")

    pitch = features.get("pitch", {})
    spectral = features.get("spectral_centroid", {})

    col1, col2, col3 = st.columns(3)

    col1.metric("Pitch Mean", f"{pitch.get('mean_hz', 'N/A')} Hz")
    col2.metric("Pitch Min", f"{pitch.get('min_hz', 'N/A')} Hz")
    col3.metric("Pitch Max", f"{pitch.get('max_hz', 'N/A')} Hz")

    st.write("Spectral centroid")
    st.dataframe(pd.DataFrame([spectral]), use_container_width=True)

    mfcc = features.get("mfcc", {})
    if mfcc:
        st.write("MFCC coefficients")
        mfcc_df = pd.DataFrame({
            "mfcc_index": list(range(1, len(mfcc.get("mean", [])) + 1)),
            "mean": mfcc.get("mean", []),
            "std": mfcc.get("std", []),
        })
        st.dataframe(mfcc_df, use_container_width=True)


def render_privacy_tab(result: dict) -> None:
    privacy = result.get("privacy_status", {})

    if not privacy:
        st.warning("No privacy status returned.")
        return

    st.subheader("Privacy Status")

    for key, value in privacy.items():
        if value is False:
            st.success(f"{key}: {value}")
        elif value is True and key == "temporary_files_deleted":
            st.success(f"{key}: {value}")
        else:
            st.info(f"{key}: {value}")

    st.caption(
        "The current MVP returns numerical features and metadata. "
        "It does not expose raw audio in the JSON output."
    )
