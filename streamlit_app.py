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
    /* Global styles for white background and black text */
    html, body, .stApp {
        background-color: white !important;
        color: black !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Ensure specific Streamlit text components are black on light backgrounds */
    .stMarkdown, .stText, .stSpinner div, .stProgress div, .stDownloadButton {
        color: black !important;
    }

    /* Sidebar background and styling */
    [data-testid="stSidebar"] {
        background-color: white !important;
        color: black !important;
        box-shadow: 2px 0 5px -2px rgba(0,0,0,0.1);
        border-radius: 0.75rem !important;
    }

    /* Headers color */
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }

    /* Apply rounded corners and shadows to various Streamlit elements */
    .stImage > div,
    .prediction-box,
    .stDataFrame,
    .stFileUpload,
    .stTextInput,
    .stSelectbox,
    .stNumberInput,
    .stDateInput,
    .stTimeInput,
    .stCheckbox,
    .stRadio,
    .stSlider {
        border-radius: 0.75rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* File uploader button styling */
    .stFileUpload > div > button {
        background-color: #f0f2f6 !important;
        color: black !important;
        border: 1px solid black !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stFileUpload > div > button:hover {
        background-color: #e2e8f0 !important;
        transform: translateY(-2px) !important;
    }
    .stFileUpload > div > button:active {
        transform: translateY(0) !important;
        box-shadow: none !important;
    }

    /* General button styling */
    .stButton > button {
        background-color: #f0f2f6 !important;
        color: black !important;
        border: 1px solid black !important;
        border-radius: 0.75rem !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        background-color: #e2e8f0 !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: none !important;
    }

    /* Image display container */
    .stImage > div {
        border: 2px solid black !important;
        padding: 5px !important;
        background-color: white !important;
    }

    /* Prediction result box */
    .prediction-box {
        border: 2px solid black !important;
        padding: 15px !important;
        background-color: white !important;
        margin-bottom: 1rem;
    }

    /* Dataframe container styling */
    .stDataFrame {
        border: 2px solid black !important;
        overflow: hidden !important;
    }

    /* Styling for Streamlit's alert boxes (info, success, error) */
    .stAlert {
        border-radius: 0.75rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* Ensure text color is white if a dark background were to be introduced */
    /* (Currently, all backgrounds are light, so text is black for contrast) */
    .dark-background-element {
        background-color: black;
        color: white !important;
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
