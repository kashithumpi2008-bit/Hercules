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
    # 1. RESIZE IMAGE
    # --------------------------------------------------

    max_width = 1200

    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    # --------------------------------------------------
    # 2. CREATE LEAF MASK
    # --------------------------------------------------

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Instead of relying only on OTSU,
    # use saturation to identify the leaf.
    #
    # This works better when the background is white,
    # grey, brown, etc.

    _, leaf_mask = cv2.threshold(
        s,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Clean mask

    kernel_large = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (21, 21)
    )

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_CLOSE,
        kernel_large
    )

    leaf_mask = cv2.morphologyEx(
        leaf_mask,
        cv2.MORPH_OPEN,
        kernel_large
    )

    # --------------------------------------------------
    # KEEP ONLY THE LARGEST CONNECTED OBJECT
    # --------------------------------------------------

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        leaf_mask,
        connectivity=8
    )

    if num_labels > 1:

        largest_label = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        leaf_mask = np.where(
            labels == largest_label,
            255,
            0
        ).astype(np.uint8)

    # Slight erosion to make sure background edges
    # are not included.

    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    leaf_mask = cv2.erode(
        leaf_mask,
        erosion_kernel,
        iterations=1
    )

    # --------------------------------------------------
    # 3. GRAYSCALE
    # --------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Smooth texture/noise

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # --------------------------------------------------
    # 4. ONLY LOOK INSIDE THE LEAF
    # --------------------------------------------------

    leaf_gray = cv2.bitwise_and(
        gray,
        gray,
        mask=leaf_mask
    )

    # --------------------------------------------------
    # 5. ENHANCE DARK VEINS
    # --------------------------------------------------

    # Black-hat detects dark structures
    # surrounded by lighter structures.

    vein_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )

    blackhat = cv2.morphologyEx(
        leaf_gray,
        cv2.MORPH_BLACKHAT,
        vein_kernel
    )

    # --------------------------------------------------
    # 6. CONTRAST ENHANCEMENT
    # --------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        blackhat
    )

    # --------------------------------------------------
    # 7. THRESHOLD VEIN RESPONSE
    # --------------------------------------------------

    # Only keep strong vein responses.

    _, vein_binary = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Only inside leaf

    vein_binary = cv2.bitwise_and(
        vein_binary,
        vein_binary,
        mask=leaf_mask
    )

    # --------------------------------------------------
    # 8. REMOVE SMALL NOISE
    # --------------------------------------------------

    clean_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    vein_binary = cv2.morphologyEx(
        vein_binary,
        cv2.MORPH_OPEN,
        clean_kernel
    )

    # --------------------------------------------------
    # 9. CONNECT BROKEN VEINS
    # --------------------------------------------------

    connect_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    vein_binary = cv2.morphologyEx(
        vein_binary,
        cv2.MORPH_CLOSE,
        connect_kernel
    )

    # --------------------------------------------------
    # 10. REMOVE OBJECTS TOUCHING LEAF BORDER
    # --------------------------------------------------

    # Background artifacts often appear around
    # the outside boundary of the leaf.

    border_mask = np.zeros_like(
        vein_binary
    )

    border_mask[3:-3, 3:-3] = 255

    vein_binary = cv2.bitwise_and(
        vein_binary,
        border_mask
    )

    # --------------------------------------------------
    # 11. FIND VEIN CANDIDATES
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        vein_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    main_veins = []

    # --------------------------------------------------
    # 12. FILTER FOR PROMINENT VEINS
    # --------------------------------------------------

    for contour in contours:

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(
            contour,
            True
        )

        # Ignore tiny structures

        if area < 25:
            continue

        if perimeter < 80:
            continue

        # Bounding box

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # Reject extremely small structures

        if max(w, h) < 40:
            continue

        # --------------------------------------------------
        # ELONGATION TEST
        # --------------------------------------------------

        elongation = max(w, h) / max(
            min(w, h),
            1
        )

        # Veins tend to be elongated.
        #
        # Random background noise tends to be
        # more compact.

        if elongation < 1.5:
            continue

        # --------------------------------------------------
        # SOLIDITY
        # --------------------------------------------------

        hull = cv2.convexHull(
            contour
        )

        hull_area = cv2.contourArea(
            hull
        )

        if hull_area == 0:
            continue

        solidity = area / hull_area

        if solidity < 0.15:
            continue

        # Candidate accepted

        main_veins.append(
            contour
        )

    # --------------------------------------------------
    # 13. LIMIT TO PROMINENT VEINS
    # --------------------------------------------------

    # Sort by length.

    main_veins = sorted(
        main_veins,
        key=lambda c: cv2.arcLength(
            c,
            False
        ),
        reverse=True
    )

    # Keep only the strongest structures.

    max_veins = 50

    main_veins = main_veins[:max_veins]

    vein_count = len(
        main_veins
    )

    # --------------------------------------------------
    # 14. DRAW RESULTS
    # --------------------------------------------------

    processed_image = image.copy()

    # Draw leaf boundary

    leaf_contours, _ = cv2.findContours(
        leaf_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        processed_image,
        leaf_contours,
        -1,
        (255, 180, 0),
        2
    )

    # Draw detected prominent veins

    cv2.drawContours(
        processed_image,
        main_veins,
        -1,
        (0, 255, 60),
        3
    )

    # --------------------------------------------------
    # 15. DISPLAY
    # --------------------------------------------------

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

        st.subheader(
            "👀 Original Leaf"
        )

        st.image(
            display_original,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "🔬 Vein Analysis"
        )

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
            <div class="result-label">
                PROMINENT VEINS DETECTED
            </div>

            <div class="result-number">
                {vein_count}
            </div>

            <div class="result-label">
                Veinalyze has finished inspecting your leaf 🌿
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    '<div class="footer">Made for the Useless Project Hackathon 🌿</div>',
    unsafe_allow_html=True
)
