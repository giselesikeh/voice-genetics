import streamlit as st


def render_upload_panel():
    """Render audio upload section."""
    st.subheader("Upload Audio Sample")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a"],
        help="Supported formats: WAV, MP3, M4A.",
    )

    if uploaded_file is not None:
        st.success(f"Selected file: {uploaded_file.name}")

    with st.expander("Optional manual/reference segments"):
        st.caption(
            "Leave these empty for automatic inference. Use segments_json only for "
            "manual segmentation. Use reference_segments_json only to evaluate automatic "
            "predictions against manual/RTTM ground truth."
        )

        segments_json = st.text_area(
            "segments_json",
            value="",
            height=100,
            placeholder='Example: [{"speaker":"speaker_1","start":0,"end":5}]',
        )

        reference_segments_json = st.text_area(
            "reference_segments_json",
            value="",
            height=100,
            placeholder='Example: [{"speaker":"spk00","start":0,"end":5}]',
        )

    return uploaded_file, segments_json, reference_segments_json
