import cv2
import numpy as np
import streamlit as st
# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Veinalyze - Vein Detection",
    page_icon="🌿",
    layout="wide"
)
# --------------------------------------------------
# CUSTOM UI STYLING
# --------------------------------------------------
st.markdown("""
<style>
    /* Main page */
    .main {
        padding-top: 2rem;
    }
    /* Title */
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        opacity: 0.75;
        margin-bottom: 2rem;
    }
    /* Upload box */
    .upload-title {
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    /* Result card */
    .result-card {
        padding: 1.5rem;
        border-radius: 18px;
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-top: 1rem;
    }
    .result-number {
        font-size: 4rem;
        font-weight: 800;
        margin: 0;
    }
    .result-label {
        font-size: 1rem;
        opacity: 0.7;
    }
    /* Verdict card */
    .verdict-card {
        padding: 1.4rem;
        border-radius: 18px;
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-top: 1rem;
    }
    .verdict-title {
        font-size: 1.5rem;
        font-weight: 700;
    }
    .verdict-text {
        font-size: 1rem;
        opacity: 0.8;
    }
    /* Footer */
    .footer {
        text-align: center;
        opacity: 0.55;
        margin-top: 3rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🌿 Veinalyze</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Your leaf has secrets. We count them.</div>',
    unsafe_allow_html=True
)
st.markdown("---")
# --------------------------------------------------
# UPLOAD SECTION
# --------------------------------------------------
st.markdown(
    '<div class="upload-title">📸 Upload Your Leaf</div>',
    unsafe_allow_html=True
)
st.write(
    "Give Veinalyze a leaf photo and let it inspect the visible vein network."
)
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
# --------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------
if uploaded_file is not None:
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )
    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )
    # --------------------------------------------------
    # 1. SEGMENT LEAF TO REMOVE BACKGROUND
    # --------------------------------------------------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s_channel, _ = cv2.split(hsv)
    _, leaf_mask = cv2.threshold(
        s_channel,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    border_sample = np.concatenate(
        [
            leaf_mask[0, :],
            leaf_mask[-1, :],
            leaf_mask[:, 0],
            leaf_mask[:, -1],
        ]
    )
    if np.mean(border_sample) > 127:
        leaf_mask = cv2.bitwise_not(leaf_mask)
    mask_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )
    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_CLOSE,
        mask_kernel
    )
    leaf_mask = cv2.erode(
        leaf_mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7)
        )
    )
    # --------------------------------------------------
    # 2. PRE-PROCESS LEAF TEXTURE
    # --------------------------------------------------
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )
    denoised = cv2.bilateralFilter(
        gray,
        d=9,
        sigmaColor=75,
        sigmaSpace=75
    )
    # --------------------------------------------------
    # 3. EXTRACT MAIN VEIN RIDGES
    # --------------------------------------------------
    # Fixed settings instead of sidebar controls
    min_vein_length = 80
    vein_thickness_kernel = 11
    k = vein_thickness_kernel
    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (k, k)
    )
    ridge = cv2.morphologyEx(
        denoised,
        cv2.MORPH_BLACKHAT,
        morph_kernel
    )
    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )
    enhanced_ridge = clahe.apply(ridge)
    # --------------------------------------------------
    # 4. EDGE DETECTION & MASKING
    # --------------------------------------------------
    canny_thresh1 = 50
    canny_thresh2 = 150
    edges = cv2.Canny(
        enhanced_ridge,
        canny_thresh1,
        canny_thresh2
    )
    masked_edges = cv2.bitwise_and(
        edges,
        edges,
        mask=leaf_mask
    )
    # --------------------------------------------------
    # 5. DETECT & FILTER PROMINENT VEINS
    # --------------------------------------------------
    contours, _ = cv2.findContours(
        masked_edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
    main_veins = [
        c for c in contours
        if cv2.arcLength(c, False) > min_vein_length
    ]
    vein_count = len(main_veins)
    # --------------------------------------------------
    # DRAW RESULTS
    # --------------------------------------------------
    processed_image = image.copy()
    cv2.drawContours(
        processed_image,
        main_veins,
        -1,
        (0, 255, 60),
        2
    )
    display_original = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
    display_processed = cv2.cvtColor(
        processed_image,
        cv2.COLOR_BGR2RGB
    )
    # --------------------------------------------------
    # IMAGE RESULTS
    # --------------------------------------------------
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👀 Original Leaf")
        st.image(
            display_original,
            use_container_width=True
        )
    with col2:
        st.subheader("🔬 Vein Analysis")
        st.image(
            display_processed,
            use_container_width=True
        )
    # --------------------------------------------------
    # MAIN RESULT
    # --------------------------------------------------
    st.markdown("---")
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">VISIBLE VEINS DETECTED</div>
            <div class="result-number">{vein_count}</div>
            <div class="result-label">
                Veinalyze has finished inspecting your leaf 🌿
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # --------------------------------------------------
    # FUN VERDICTS
    # --------------------------------------------------
    if vein_count == 0:
        verdict = "🕵️ The Invisible Leaf"
        verdict_text = (
            "No convincing veins were detected. "
            "Either this leaf is extremely mysterious "
            "or the camera needs a better look."
        )
    elif 1 <= vein_count <= 10:
        verdict = "🌱 The Minimalist"
        verdict_text = (
            "Simple, clean, and efficient. "
            "This leaf clearly believes that less is more."
        )
    elif 11 <= vein_count <= 25:
        verdict = "🍃 The Balanced One"
        verdict_text = (
            "A respectable amount of vascular activity. "
            "Not too much, not too little, just leaf."
        )
    elif 26 <= vein_count <= 50:
        verdict = "⚡ The Overachiever"
        verdict_text = (
            "Someone forgot to tell this leaf to relax. "
            "Veins everywhere. Maximum productivity."
        )
    elif 51 <= vein_count <= 100:
        verdict = "🌿 The Network Engineer"
        verdict_text = (
            "This leaf has built an entire communication network. "
            "Somewhere, tiny messages are definitely being delivered."
        )
    else:
        verdict = "🌪️ The Vein Multiverse"
        verdict_text = (
            "At this point, this isn't just a leaf. "
            "It's a complete transportation system."
        )
    st.markdown(
        f"""
        <div class="verdict-card">
            <div class="verdict-title">{verdict}</div>
            <div class="verdict-text">{verdict_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # --------------------------------------------------
    # SMALL EXTRA INTERACTION
    # --------------------------------------------------
    st.markdown("---")
    with st.expander("🔍 What did Veinalyze actually do?"):
        st.write(
            "Veinalyze processes the uploaded image using "
            "OpenCV. It separates the leaf from the background, "
            "enhances vein-like structures, detects edges, "
            "and counts prominent detected contours."
        )
        st.caption(
            "The result is an experimental visual estimate, "
            "not a botanical measurement."
        )
else:
    st.info(
        "🌿 Upload a leaf image above to begin the analysis."
    )
# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    '<div class="footer">Made for the Useless Project Hackathon 🌿</div>',
    unsafe_allow_html=True
)
