import json

import pandas as pd
import streamlit as st

from utils.formatting import (
    format_seconds,
    safe_get,
    speaker_durations_to_dataframe,
    speaker_segments_to_dataframe,
)


def render_results(result: dict) -> None:
    """Render analysis results from backend JSON."""
    st.header("Analysis Results")

    tab_summary, tab_quality, tab_speaker, tab_cluster, tab_eval, tab_features, tab_speaker_features, tab_privacy, tab_json = st.tabs(
        [
            "Summary",
            "Quality & VAD",
            "Speaker Segmentation",
            "Clustering Metrics",
            "DER Evaluation",
            "Global Acoustic Features",
            "Speaker Acoustic Features",
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

    with tab_cluster:
        render_clustering_tab(result)

    with tab_eval:
        render_evaluation_tab(result)

    with tab_features:
        render_features_tab(result)

    with tab_speaker_features:
        render_speaker_features_tab(result)

    with tab_privacy:
        render_privacy_tab(result)

    with tab_json:
        st.download_button(
            label="Download JSON result",
            data=json.dumps(result, indent=2),
            file_name="voice_genetics_result.json",
            mime="application/json",
        )
        st.json(result)


def render_summary_tab(result: dict) -> None:
    filename = safe_get(result, ["filename"])
    method = safe_get(result, ["speaker_segmentation", "method"], "not used")
    detected_speakers = safe_get(
        result,
        ["speaker_segmentation", "detected_speakers"],
        "N/A",
    )
    total_time = safe_get(result, ["runtime_metrics", "total_processing_seconds"], "N/A")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("File", filename)
    col2.metric("Method", method)
    col3.metric("Detected Speakers", detected_speakers)

    try:
        col4.metric("Processing Time", f"{float(total_time):.2f} s")
    except Exception:
        col4.metric("Processing Time", "N/A")

    speaker_info = result.get("speaker_segmentation", {})
    if speaker_info.get("method") == "ecapa_v2_full_pipeline":
        st.success(
            "Method 3B is active: adaptive/central VAD, overlapping chunks, ECAPA embeddings, "
            "smoothing, short-segment merging, clustering metrics, and optional speaker-count estimation."
        )

        limitations = speaker_info.get("limitations_addressed", {})
        if limitations:
            st.write("Method 3B limitation coverage")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"limitation": key, "status": value}
                        for key, value in limitations.items()
                    ]
                ),
                use_container_width=True,
            )
    else:
        st.info(
            "Voice Genetics extracts acoustic features and optionally performs speaker segmentation. "
            "It does not predict genes directly."
        )


def render_quality_tab(result: dict) -> None:
    st.subheader("Audio Quality Metrics")

    quality = result.get("quality_metrics", {})
    if quality:
        st.dataframe(pd.DataFrame([quality]), use_container_width=True)
    else:
        st.warning("No quality metrics returned.")

    st.subheader("Voice Activity / Preprocessing")
    voice_activity = result.get("voice_activity", {})
    preprocessing = result.get("preprocessing_metrics", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Speech Duration",
        format_seconds(voice_activity.get("speech_duration_seconds", "N/A")),
    )
    col2.metric(
        "Non-Speech Duration",
        format_seconds(voice_activity.get("non_speech_duration_seconds", "N/A")),
    )
    col3.metric("Speech Coverage", voice_activity.get("speech_coverage_ratio", "N/A"))
    col4.metric("Effective RMS", voice_activity.get("effective_min_rms", "N/A"))

    if voice_activity:
        vad_summary = {
            key: value
            for key, value in voice_activity.items()
            if key not in ["speech_regions", "removed_non_speech_segments"]
        }
        st.write("Voice activity summary")
        st.dataframe(pd.DataFrame([vad_summary]), use_container_width=True)

        with st.expander("Speech regions and removed non-speech segments"):
            if voice_activity.get("speech_regions"):
                st.write("Speech regions")
                st.dataframe(pd.DataFrame(voice_activity["speech_regions"]), use_container_width=True)
            if voice_activity.get("removed_non_speech_segments"):
                st.write("Removed non-speech segments")
                st.dataframe(
                    pd.DataFrame(voice_activity["removed_non_speech_segments"]),
                    use_container_width=True,
                )

    if preprocessing:
        preprocessing_summary = {
            key: value
            for key, value in preprocessing.items()
            if key not in ["speech_regions", "removed_non_speech_segments"]
        }
        st.write("Preprocessing summary")
        st.dataframe(pd.DataFrame([preprocessing_summary]), use_container_width=True)


def render_speaker_tab(result: dict) -> None:
    speaker_info = result.get("speaker_segmentation", {})

    if not speaker_info or not speaker_info.get("enabled", False):
        st.warning("Speaker segmentation was not enabled for this run.")
        return

    st.subheader("Speaker Segmentation Summary")

    summary_keys = [
        "method",
        "detected_speakers",
        "expected_speakers",
        "expected_speakers_requested",
        "auto_detect_speakers",
        "chunk_duration_seconds",
        "chunk_hop_seconds",
        "usable_speech_chunks",
        "removed_pause_chunks",
        "clustering_backend",
        "embedding_type",
        "embedding_model",
        "smoothing_passes",
        "labels_changed_by_smoothing",
        "short_segment_merging_enabled",
        "min_segment_duration_seconds",
        "short_segments_before_merging",
        "cluster_balance_warning",
        "low_confidence_warning",
    ]
    summary = {key: speaker_info.get(key) for key in summary_keys if key in speaker_info}
    st.dataframe(pd.DataFrame([summary]), use_container_width=True)

    if speaker_info.get("speaker_count_estimation", {}).get("enabled"):
        st.subheader("Automatic Speaker-Count Estimation")
        estimation = speaker_info["speaker_count_estimation"]
        st.metric("Selected K", estimation.get("selected_k"))
        candidates = estimation.get("candidates", [])
        if candidates:
            st.dataframe(pd.DataFrame(candidates), use_container_width=True)

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


def render_clustering_tab(result: dict) -> None:
    st.subheader("Internal Clustering Quality Metrics")

    metrics = safe_get(result, ["speaker_segmentation", "clustering_metrics"], {})
    if not metrics:
        st.warning(
            "No clustering metrics returned for this run. Run an automatic method such as "
            "auto, ecapa, ecapa_v2, or wavlm."
        )
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Silhouette Score", metrics.get("silhouette_score", "N/A"))
    col2.metric("Cluster Balance", metrics.get("cluster_balance_ratio", "N/A"))
    col3.metric("Speaker Switches", metrics.get("speaker_switches", "N/A"))
    col4.metric("Segment Smoothness", metrics.get("segment_smoothness", "N/A"))

    if metrics.get("silhouette_note"):
        st.warning(metrics["silhouette_note"])

    summary_fields = {
        key: value
        for key, value in metrics.items()
        if key not in ["cluster_counts", "cluster_percentages", "interpretation"]
    }
    st.write("Clustering metric summary")
    st.dataframe(pd.DataFrame([summary_fields]), use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.write("Cluster counts")
        counts = metrics.get("cluster_counts", {})
        if counts:
            st.dataframe(
                pd.DataFrame(
                    [{"cluster": key, "chunks": value} for key, value in counts.items()]
                ),
                use_container_width=True,
            )

    with col_b:
        st.write("Cluster percentages")
        percentages = metrics.get("cluster_percentages", {})
        if percentages:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"cluster": key, "percentage": value}
                        for key, value in percentages.items()
                    ]
                ),
                use_container_width=True,
            )

    interpretation = metrics.get("interpretation", {})
    if interpretation:
        st.info(
            "Silhouette checks cluster separation. Cluster balance checks whether one cluster dominates. "
            "Speaker switches and segment smoothness show whether labels change too frequently."
        )
        st.json(interpretation)


def render_evaluation_tab(result: dict) -> None:
    st.subheader("DER / Diarization Evaluation")

    evaluation = result.get("diarization_evaluation", {})
    if not evaluation or not evaluation.get("enabled", False):
        st.warning(
            evaluation.get(
                "message",
                "No diarization evaluation returned. Provide reference_segments_json to score predictions.",
            )
        )
        return

    rates = evaluation.get("rate_metrics", {})
    times = evaluation.get("time_metrics_seconds", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("DER %", rates.get("der_percent", "N/A"))
    col2.metric("Speech Precision", rates.get("speech_precision", "N/A"))
    col3.metric("Speech Recall", rates.get("speech_recall", "N/A"))
    col4.metric(
        "Speaker Accuracy",
        rates.get("speaker_assignment_accuracy_on_overlap", "N/A"),
    )

    st.write("Time metrics")
    st.dataframe(pd.DataFrame([times]), use_container_width=True)

    st.write("Rate metrics")
    st.dataframe(pd.DataFrame([rates]), use_container_width=True)

    mapping = evaluation.get("speaker_mapping_predicted_to_reference", {})
    if mapping:
        st.write("Best predicted-to-reference speaker mapping")
        st.dataframe(
            pd.DataFrame(
                [
                    {"predicted_speaker": key, "reference_speaker": value}
                    for key, value in mapping.items()
                ]
            ),
            use_container_width=True,
        )

    st.write("Speaker duration error")
    duration_error = evaluation.get("speaker_duration_error", {})
    per_speaker = duration_error.get("per_speaker", {})
    if per_speaker:
        rows = []
        for speaker, payload in per_speaker.items():
            rows.append({"reference_speaker": speaker, **payload})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.write("Boundary metrics")
    st.dataframe(pd.DataFrame([evaluation.get("boundary_metrics", {})]), use_container_width=True)

    pyannote = evaluation.get("standard_der_pyannote", {})
    if pyannote:
        st.subheader("Optional standard pyannote.metrics DER")
        if pyannote.get("enabled"):
            st.success("pyannote.metrics DER computed.")
        else:
            st.info(pyannote.get("message", "pyannote.metrics DER was not computed."))
        st.json(pyannote)


def render_features_tab(result: dict) -> None:
    features = result.get("features", {})
    if not features:
        st.warning("No acoustic features returned.")
        return

    st.subheader("Global Acoustic Feature Summary")

    duration = features.get("duration_seconds", "N/A")
    pitch = features.get("pitch", {})
    spectral = features.get("spectral_centroid", {})
    mfcc = features.get("mfcc", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Feature Audio Duration", format_seconds(duration))
    col2.metric("Pitch Mean", f"{pitch.get('mean_hz', 'N/A')} Hz")
    col3.metric("Pitch Min", f"{pitch.get('min_hz', 'N/A')} Hz")
    col4.metric("Pitch Max", f"{pitch.get('max_hz', 'N/A')} Hz")

    st.subheader("Pitch Features")
    st.dataframe(pd.DataFrame([pitch]), use_container_width=True)

    st.subheader("Timbre / Spectral Features")
    spectral_table = {
        "spectral_centroid_mean": spectral.get("mean"),
        "spectral_centroid_std": spectral.get("std"),
    }
    st.dataframe(pd.DataFrame([spectral_table]), use_container_width=True)

    st.subheader("MFCC Features")
    if mfcc:
        means = mfcc.get("mean", [])
        stds = mfcc.get("std", [])
        mfcc_df = pd.DataFrame(
            {
                "mfcc_index": list(range(1, len(means) + 1)),
                "mean": means,
                "std": stds,
            }
        )
        st.dataframe(mfcc_df, use_container_width=True)
    else:
        st.info("No MFCC features available.")


def render_speaker_features_tab(result: dict) -> None:
    speaker_features = result.get("speaker_features", {})
    if not speaker_features:
        st.warning("No speaker-level acoustic features returned.")
        return

    st.subheader("Speaker-Level Acoustic Features")

    summary_rows = []
    mfcc_rows = []

    for speaker, payload in speaker_features.items():
        features = payload.get("features", {})
        preprocessing = payload.get("preprocessing_metrics", {})
        pitch = features.get("pitch", {})
        spectral = features.get("spectral_centroid", {})
        mfcc = features.get("mfcc", {})

        summary_rows.append(
            {
                "speaker": speaker,
                "feature_audio_duration_seconds": features.get("duration_seconds"),
                "processed_duration_seconds": preprocessing.get("processed_duration_seconds"),
                "removed_silence_percentage": preprocessing.get("removed_silence_percentage"),
                "pitch_mean_hz": pitch.get("mean_hz"),
                "pitch_min_hz": pitch.get("min_hz"),
                "pitch_max_hz": pitch.get("max_hz"),
                "spectral_centroid_mean": spectral.get("mean"),
                "spectral_centroid_std": spectral.get("std"),
            }
        )

        means = mfcc.get("mean", [])
        stds = mfcc.get("std", [])
        for index, mean_value in enumerate(means):
            mfcc_rows.append(
                {
                    "speaker": speaker,
                    "mfcc_index": index + 1,
                    "mean": mean_value,
                    "std": stds[index] if index < len(stds) else None,
                }
            )

    if summary_rows:
        st.write("Speaker acoustic summary")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    if mfcc_rows:
        st.write("Speaker MFCC features")
        st.dataframe(pd.DataFrame(mfcc_rows), use_container_width=True)


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
        "The current system returns numerical features and metadata. It does not expose raw audio "
        "in the JSON output."
    )
