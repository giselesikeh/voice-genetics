import streamlit as st


def render_sidebar() -> dict:
    """
    Render sidebar controls and return selected processing settings.
    """
    st.sidebar.header("Processing Settings")

    backend_url = st.sidebar.text_input(
        "Backend URL",
        value="http://127.0.0.1:8000",
        help="FastAPI backend URL. For local use, keep http://127.0.0.1:8000",
    )

    method_label = st.sidebar.selectbox(
        "Segmentation Method",
        options=[
            "none — acoustic features only",
            "auto — Method 2: handcrafted DSP features + K-Means",
            "ecapa — Method 3: ECAPA-TDNN speaker embeddings + K-Means",
            "ecapa_v2 — Method 3B: improved ECAPA pipeline",
            "wavlm — Method 4: WavLM embeddings + K-Means",
        ],
        index=3,
    )
    segmentation_method = method_label.split(" — ")[0]

    expected_speakers = st.sidebar.number_input(
        "Expected Speakers",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Manual expected speaker count. Method 3B can optionally auto-estimate speaker count.",
    )

    chunk_duration_seconds = st.sidebar.number_input(
        "Chunk Duration (seconds)",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Length of each speech chunk used for speaker embedding/clustering.",
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
        "Minimum Speech Region Duration",
        min_value=0.05,
        max_value=2.0,
        value=0.25,
        step=0.05,
    )

    vad_merge_gap_seconds = st.sidebar.number_input(
        "VAD Merge Gap Seconds",
        min_value=0.0,
        max_value=3.0,
        value=0.8,
        step=0.1,
    )

    with st.sidebar.expander("Method 3B ECAPA advanced settings", expanded=segmentation_method == "ecapa_v2"):
        ecapa_chunk_hop_seconds = st.number_input(
            "ECAPA Chunk Hop Seconds",
            min_value=0.25,
            max_value=5.0,
            value=1.0,
            step=0.25,
            help="Hop size between ECAPA chunks. Smaller hop means more overlap but slower runtime.",
        )

        ecapa_smoothing_passes = st.number_input(
            "ECAPA Smoothing Passes",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
            help="Number of isolated-label smoothing passes for Method 3B.",
        )

        ecapa_auto_detect_speakers = st.checkbox(
            "Auto-detect speaker count for Method 3B",
            value=False,
            help="Searches several K values using silhouette score, cluster balance, and smoothness.",
        )

        col_min, col_max = st.columns(2)
        with col_min:
            ecapa_min_speakers = st.number_input(
                "Min Speakers",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
                disabled=not ecapa_auto_detect_speakers,
            )
        with col_max:
            ecapa_max_speakers = st.number_input(
                "Max Speakers",
                min_value=2,
                max_value=10,
                value=6,
                step=1,
                disabled=not ecapa_auto_detect_speakers,
            )

        ecapa_min_segment_duration_seconds = st.number_input(
            "Minimum Final Segment Duration",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.25,
            help="Very short isolated segments are merged into neighboring segments.",
        )

        ecapa_merge_gap_seconds = st.number_input(
            "ECAPA Merge Gap Seconds",
            min_value=0.0,
            max_value=3.0,
            value=0.3,
            step=0.1,
            help="Merge same-speaker ECAPA segments separated by short gaps.",
        )

        ecapa_clustering_backend = st.selectbox(
            "Method 3B Clustering Backend",
            options=["agglomerative_cosine", "kmeans"],
            index=0,
        )

    with st.sidebar.expander("Evaluation settings", expanded=False):
        use_pyannote_metrics = st.checkbox(
            "Try standard pyannote.metrics DER",
            value=False,
            help="Works only if pyannote.metrics is installed in the backend environment.",
        )

        der_collar_seconds = st.number_input(
            "DER Collar Seconds",
            min_value=0.0,
            max_value=2.0,
            value=0.25,
            step=0.05,
        )

        der_skip_overlap = st.checkbox(
            "Skip overlap in standard DER",
            value=False,
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
        "ecapa_chunk_hop_seconds": float(ecapa_chunk_hop_seconds),
        "ecapa_smoothing_passes": int(ecapa_smoothing_passes),
        "ecapa_auto_detect_speakers": bool(ecapa_auto_detect_speakers),
        "ecapa_min_speakers": int(ecapa_min_speakers),
        "ecapa_max_speakers": int(ecapa_max_speakers),
        "ecapa_min_segment_duration_seconds": float(ecapa_min_segment_duration_seconds),
        "ecapa_merge_gap_seconds": float(ecapa_merge_gap_seconds),
        "ecapa_clustering_backend": ecapa_clustering_backend,
        "use_pyannote_metrics": bool(use_pyannote_metrics),
        "der_collar_seconds": float(der_collar_seconds),
        "der_skip_overlap": bool(der_skip_overlap),
    }
