import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
import tempfile
import os
import cv2
import pandas as pd
import altair as alt

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🕵️‍♂️",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Global Dark Theme */
    html, body, .stApp {
        background: #141414 !important;
        color: #e5e5e5 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #e5e5e5 !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1f1f1f !important;
        color: #e5e5e5 !important;
        box-shadow: 2px 0 5px -2px rgba(0,0,0,0.5);
        border-radius: 0.75rem !important;
        padding: 1rem !important;
    }

    /* File uploader button - green accent */
    .stFileUpload > div > button {
        background-color: #00b894 !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 1rem !important;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease-in-out !important;
    }
    .stFileUpload > div > button:hover {
        background-color: #00d88a !important;
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }
    .stFileUpload > div > button:active {
        transform: translateY(0) scale(1) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    /* General button */
    .stButton > button {
        background-color: #2d2d2d !important;
        color: #00b894 !important;
        border: 1px solid #00b894 !important;
        border-radius: 1rem !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        transition: all 0.3s ease-in-out !important;
        margin: 0.2rem !important;
        min-width: 100px !important;
    }
    .stButton > button:hover {
        background-color: #00b894 !important;
        color: #141414 !important;
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
    }

    /* Image container */
    .stImage > div {
        border-radius: 1rem !important;
        max-width: 400px !important;
        margin: 1rem auto !important;
        overflow: hidden !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
        border: 2px solid #2d2d2d !important;
        background-color: #1f1f1f !important;
        transition: transform 0.3s ease;
    }
    .stImage > div:hover {
        transform: scale(1.02);
    }

    /* Prediction box */
    .prediction-box {
        border-radius: 1rem !important;
        background-color: #2d2d2d !important;
        color: #e5e5e5 !important;
        padding: 1rem !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.5);
        margin-bottom: 1rem;
    }

    /* Buttons next to image (Real/Fake) */
    .prediction-buttons {
        display: flex !important;
        flex-direction: row !important;
        justify-content: flex-start !important;
        gap: 1rem !important;
        margin-top: 0.5rem !important;
    }
    .prediction-buttons .stButton > button {
        flex: 1;
    }

    /* Confidence bars */
    .altair-Chart svg {
        background-color: #1f1f1f !important;
    }

    /* DataFrame styling */
    .stDataFrame {
        border-radius: 1rem !important;
        overflow: hidden !important;
        border: 2px solid #2d2d2d !important;
        background-color: #1f1f1f !important;
        color: #e5e5e5 !important;
    }

    /* Alerts */
    .stAlert {
        border-radius: 1rem !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #00b894;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-track {
        background: #1f1f1f;
    }
    </style>
    """,
    unsafe_allow_html=True
)


IMAGE_SIZE = (240, 240)
MODEL_PATH = "model_optimized.tflite"
THRESHOLD = 0.5

# =========================
# ELA and Preprocessing
# =========================
def perform_ela(image, rescale_size=IMAGE_SIZE):
    quality = 90
    image = image.convert('RGB')
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name
        image.save(temp_path, 'JPEG', quality=quality)

    try:
        compressed = Image.open(temp_path)
        ela_image = ImageChops.difference(image, compressed)
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        ela_image = ela_image.resize(rescale_size)
        return np.array(ela_image)
    finally:
        os.remove(temp_path)

def preprocess_image(image, input_dtype):
    ela_img = perform_ela(image)
    ela_img = tf.image.resize(ela_img, IMAGE_SIZE)
    if input_dtype == np.float32:
        ela_img = tf.cast(ela_img, tf.float32)
        ela_img = tf.keras.applications.efficientnet.preprocess_input(ela_img)
    else:
        ela_img = tf.cast(ela_img, tf.uint8)
    return np.expand_dims(ela_img, axis=0)

# =========================
# Load TFLite Model
# =========================
@st.cache_resource
def load_tflite_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_dtype = input_details[0]['dtype']
        return interpreter, input_details, output_details, input_dtype
    except Exception as e:
        st.error(f"🚨 Could not load model: {e}")
        return None, None, None, None

# =========================
# Predict Deepfake
# =========================
def tflite_predict(interpreter, input_details, output_details, input_dtype, image):
    input_data = preprocess_image(image, input_dtype)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Compute fake probability
    if len(raw_output.shape) == 0:  # single scalar
        fake_prob = float(raw_output)
    elif len(raw_output) == 1:
        fake_prob = float(raw_output[0])
    elif len(raw_output) == 2:
        fake_prob = float(raw_output[1])
    else:
        raise ValueError(f"Unexpected model output shape: {raw_output.shape}")

    is_fake = fake_prob >= THRESHOLD
    return is_fake, fake_prob

# =========================
# Display Confidence Bar
# =========================
def display_confidence_bar(filename, probability):
    data = pd.DataFrame({
        'Label': ['Real', 'Fake'],
        'Confidence': [1 - probability, probability]
    })
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Confidence:Q', axis=alt.Axis(format='.1%', title='Confidence')),
        y=alt.Y('Label:N', sort=None, title=''),
        color=alt.Color('Label:N', scale=alt.Scale(range=['#28a745', '#dc3545']), legend=None)
    ).properties(
        width=400,
        height=70,
        title=f"Confidence for {filename}"
    ).configure_view(stroke='transparent')
    st.altair_chart(chart, use_container_width=True)

# =========================
# Main App
# =========================
def main():
    st.markdown("<h1 style='text-align: center;'>🕵️‍♂️ Deepfake Image Detector</h1>", unsafe_allow_html=True)
    st.markdown("---")

    interpreter, input_details, output_details, input_dtype = load_tflite_model()
    if interpreter is None:
        st.stop()

    uploaded_files = st.file_uploader(
        "Choose images", type=["jpg","jpeg","png"], accept_multiple_files=True
    )

    if uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_container_width=True)

            with st.spinner("Analyzing..."):
                is_fake, fake_prob = tflite_predict(interpreter, input_details, output_details, input_dtype, image)

            if is_fake:
                st.error(f"🚨 FAKE IMAGE! ({fake_prob*100:.2f}%)")
            else:
                st.success(f"✅ REAL IMAGE ({fake_prob*100:.2f}%)")

            display_confidence_bar(uploaded_file.name, fake_prob)

            results.append({
                "Filename": uploaded_file.name,
                "Prediction": "FAKE" if is_fake else "REAL",
                "Confidence": f"{fake_prob*100:.2f}%"
            })

        st.markdown("### Overall Results")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("Upload one or more images to check.")

if __name__ == "__main__":
    main()
