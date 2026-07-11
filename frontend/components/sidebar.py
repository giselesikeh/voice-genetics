import streamlit as st


def render_sidebar() -> dict:
    """
    Render sidebar controls and return selected settings.
    """
    st.sidebar.header("Processing Settings")

    backend_url = st.sidebar.text_input(
        "Backend URL",
        value="http://localhost:8000",
        help="FastAPI backend URL. For local development, keep http://localhost:8000",
    )

    method_label = st.sidebar.selectbox(
        "Segmentation Method",
        options=[
            "none — acoustic features only",
            "auto — Method 2: handcrafted DSP features + K-Means",
            "wavlm — Method 4: WavLM embeddings + K-Means",
        ],
        index=1,
    )

    segmentation_method = method_label.split(" — ")[0]

    expected_speakers = st.sidebar.number_input(
        "Expected Speakers",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
    )

    chunk_duration_seconds = st.sidebar.number_input(
        "Chunk Duration (seconds)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
    )

    st.sidebar.subheader("Voice Activity Settings")

    vad_mode = st.sidebar.selectbox(
        "VAD Mode",
        options=["adaptive", "fixed"],
        index=0,
    )

    vad_top_db = st.sidebar.slider(
        "VAD top_db",
        min_value=10,
        max_value=60,
        value=30,
        step=5,
    )

    vad_min_rms = st.sidebar.number_input(
        "VAD minimum RMS",
        min_value=0.0,
        max_value=0.1,
        value=0.015,
        step=0.005,
        format="%.3f",
    )

    vad_min_region_duration_seconds = st.sidebar.number_input(
        "Minimum speech region duration",
        min_value=0.05,
        max_value=2.0,
        value=0.25,
        step=0.05,
    )

    vad_merge_gap_seconds = st.sidebar.number_input(
        "Merge gap seconds",
        min_value=0.0,
        max_value=3.0,
        value=0.8,
        step=0.1,
    )

    return {
        "backend_url": backend_url,
        "segmentation_method": segmentation_method,
        "expected_speakers": int(expected_speakers),
        "chunk_duration_seconds": float(chunk_duration_seconds),
        "vad_mode": vad_mode,
        "vad_top_db": int(vad_top_db),
        "vad_min_rms": float(vad_min_rms),
        "vad_min_region_duration_seconds": float(vad_min_region_duration_seconds),
        "vad_merge_gap_seconds": float(vad_merge_gap_seconds),
    }
