import streamlit as st

from components.sidebar import render_sidebar
from components.upload_panel import render_upload_panel
from components.results_view import render_results
from services.api_client import (
    check_health,
    extract_audio_features,
    VoiceGeneticsAPIError,
)


st.set_page_config(
    page_title="Voice Genetics",
    page_icon="🎙️",
    layout="wide",
)


def main() -> None:
    st.title("🎙️ Voice Genetics")
    st.caption("Acoustic Feature Extraction and Baseline Speaker Segmentation MVP")

    st.markdown(
        """
        This Streamlit frontend connects to the Voice Genetics FastAPI backend.
        Upload an audio sample, select a processing method, and view the returned
        acoustic features, speaker segmentation output, and privacy status.
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
            "For Review/Midterm demo: use Method 2 (`auto`) or Method 4 (`wavlm`). "
            "WavLM can take longer because it loads a pretrained speech model."
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
                    segments_json=segments_json,
                    reference_segments_json=reference_segments_json,
                )

                st.success("Analysis completed successfully.")
                render_results(result)

            except VoiceGeneticsAPIError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
