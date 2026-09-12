import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Venalyze - Main Vein Auditor", layout="wide")

st.title("🌿 Veinalyze: Vein Detection")
st.caption(
    "Mapping of Primary and Micro-Vein Networks."
)

# Sidebar controls to dial in sensitivity for different leaf varieties
with st.sidebar:
    st.header("🎛️ Main Vein Calibration")
    min_vein_length = st.slider(
        "Minimum Vein Length (px)",
        min_value=30,
        max_value=300,
        value=80,
        help="Higher values ignore small branches and keep only primary veins.",
    )
    vein_thickness_kernel = st.slider(
        "Ridge Filter Size",
        min_value=5,
        max_value=25,
        value=11,
        step=2,
        help="Controls the thickness of veins to isolate. Larger numbers focus on thicker veins.",
    )
    canny_thresh1 = st.slider("Edge Sensitivity (Low)", 20, 150, 50)
    canny_thresh2 = st.slider("Edge Sensitivity (High)", 80, 250, 150)

uploaded_file = st.file_uploader(
    "Upload a leaf image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # --- 1. SEGMENT LEAF TO REMOVE BACKGROUND ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s_channel, _ = cv2.split(hsv)

    _, leaf_mask = cv2.threshold(
        s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
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

    mask_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, mask_kernel)
    leaf_mask = cv2.erode(leaf_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # --- 2. PRE-PROCESS LEAF TEXTURE ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # --- 3. EXTRACT MAIN VEIN RIDGES ---
    k = vein_thickness_kernel
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    ridge = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, morph_kernel)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_ridge = clahe.apply(ridge)

    # --- 4. EDGE DETECTION & MASKING ---
    edges = cv2.Canny(enhanced_ridge, canny_thresh1, canny_thresh2)
    masked_edges = cv2.bitwise_and(edges, edges, mask=leaf_mask)

    # --- 5. DETECT & FILTER PROMINENT VEINS ---
    contours, _ = cv2.findContours(
        masked_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    main_veins = [c for c in contours if cv2.arcLength(c, False) > min_vein_length]
    vein_count = len(main_veins)

    # Render results
    processed_image = image.copy()
    cv2.drawContours(processed_image, main_veins, -1, (0, 255, 60), 2)

    display_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_processed = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Specimen")
        st.image(display_original, use_container_width=True)
    with col2:
        st.subheader("Primary Vein Network")
        st.image(display_processed, use_container_width=True)

    st.markdown("---")

    # Primary Vein Count display
    st.metric("Primary Vein Count", f"{vein_count}")

    # Playful Leaf Personality Verdicts
    if vein_count == 0:
        st.info("🌱 **Leaf Verdict:** Clean Slate Minimalist — either this leaf likes to keep secrets, or you need to tweak the sliders!")
    elif 1 <= vein_count <= 5:
        st.info("🍃 **Leaf Verdict:** Such an Organized Leaf — minimalist, disciplined, and definitely uses a spreadsheet.")
    elif 6 <= vein_count <= 15:
        st.success("✨ **Leaf Verdict:** Pretty Organized Leaf — great balance, nice aesthetic, just living its best photosynthetic life.")
    elif 16 <= vein_count <= 25:
        st.warning("⚡ **Leaf Verdict:** Type-A Overachiever — running 40 side projects and completely caffeinated.")
    else:
        st.error("🌪️ **Leaf Verdict:** Chaotic Evil Jungle Mess — looks like a tangled pair of wired headphones from 2008.")
