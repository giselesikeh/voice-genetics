import streamlit as st

from components.sidebar import render_sidebar
from components.upload_panel import render_upload_panel
from components.results_view import render_results
from services.api_client import (
    VoiceGeneticsAPIError,
    check_health,
    extract_audio_features,
)


st.set_page_config(
    page_title="Voice Genetics",
    page_icon="🎙️",
    layout="wide",
)


def main() -> None:
    st.title("🎙️ Voice Genetics")
    st.caption("Acoustic Feature Extraction and Speaker Segmentation System")

    st.markdown(
        """
        Voice Genetics is a privacy-aware acoustic feature extraction system.
        Upload a supported audio sample, select a processing method, and view
        quality metrics, preprocessing information, acoustic features, speaker
        segmentation output, clustering metrics, diarization evaluation, runtime
        metrics, and privacy status.
        """
    )

    settings = render_sidebar()

    with st.sidebar:
        st.divider()
        st.subheader("Backend Check")

        if st.button("Check Backend Health"):
            try:
                health = check_health(settings["backend_url"])
                st.success("Backend is running")
                st.json(health)
            except VoiceGeneticsAPIError as exc:
                st.error(str(exc))

    uploaded_file, segments_json, reference_segments_json = render_upload_panel()

    st.divider()

    col1, col2 = st.columns([1, 3])

    with col1:
        run_button = st.button(
            "Run Analysis",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        )

    with col2:
        st.caption(
            "Supported formats: WAV, MP3, M4A. Recommended for Review Session 4: "
            "Method 3B (`ecapa_v2`) with adaptive VAD, overlapping chunks, smoothing, "
            "clustering metrics, and optional speaker-count estimation."
        )

    if run_button:
        if uploaded_file is None:
            st.warning("Please upload an audio file first.")
            return

        with st.spinner("Processing audio. Please wait..."):
            try:
                result = extract_audio_features(
                    backend_url=settings["backend_url"],
                    uploaded_file=uploaded_file,
                    segmentation_method=settings["segmentation_method"],
                    expected_speakers=settings["expected_speakers"],
                    chunk_duration_seconds=settings["chunk_duration_seconds"],
                    vad_mode=settings["vad_mode"],
                    vad_top_db=settings["vad_top_db"],
                    vad_min_rms=settings["vad_min_rms"],
                    vad_min_region_duration_seconds=settings[
                        "vad_min_region_duration_seconds"
                    ],
                    vad_merge_gap_seconds=settings["vad_merge_gap_seconds"],
                    ecapa_chunk_hop_seconds=settings["ecapa_chunk_hop_seconds"],
                    ecapa_smoothing_passes=settings["ecapa_smoothing_passes"],
                    ecapa_auto_detect_speakers=settings[
                        "ecapa_auto_detect_speakers"
                    ],
                    ecapa_min_speakers=settings["ecapa_min_speakers"],
                    ecapa_max_speakers=settings["ecapa_max_speakers"],
                    ecapa_min_segment_duration_seconds=settings[
                        "ecapa_min_segment_duration_seconds"
                    ],
                    ecapa_merge_gap_seconds=settings["ecapa_merge_gap_seconds"],
                    ecapa_clustering_backend=settings["ecapa_clustering_backend"],
                    use_pyannote_metrics=settings["use_pyannote_metrics"],
                    der_collar_seconds=settings["der_collar_seconds"],
                    der_skip_overlap=settings["der_skip_overlap"],
                    segments_json=segments_json,
                    reference_segments_json=reference_segments_json,
                )

                st.success("Analysis completed successfully.")
                render_results(result)

            except VoiceGeneticsAPIError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
