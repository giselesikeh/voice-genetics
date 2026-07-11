import streamlit as st


def render_upload_panel():
    """
    Render audio upload section.
    """
    st.subheader("Upload Audio Sample")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "mp4", "mov"],
        help="Upload a voice or audio sample for acoustic feature extraction.",
    )

    if uploaded_file is not None:
        st.success(f"Selected file: {uploaded_file.name}")

    with st.expander("Optional manual/reference segments"):
        st.caption(
            "Leave these empty for automatic inference. "
            "Manual/reference segments are mainly used for development evaluation."
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
            placeholder='Example: [{"speaker":"speaker_1","start":0,"end":5}]',
        )

    return uploaded_file, segments_json, reference_segments_json
