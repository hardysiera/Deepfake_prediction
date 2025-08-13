import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import pandas as pd
import altair as alt

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide"
)

# Apply custom CSS for white background, centered heading, black boxes
st.markdown("""
    <style>
    .main {
        background-color: white;
        color: black;
    }
    .stImage img {
        border: 2px solid black;
        border-radius: 5px;
    }
    .box {
        border: 2px solid black;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .center {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Constants
# =========================
IMAGE_SIZE = (240, 240)
MODEL_PATH = 'optimized_model.tflite'  # TFLite model
THRESHOLD = 0.5

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
        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        return None, None, None

# =========================
# Preprocess Image
# =========================
def preprocess_image(image):
    img_array = np.array(image)
    if img_array.shape[-1] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_array, IMAGE_SIZE)
    img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_resized)
    img_batch = np.expand_dims(img_preprocessed, axis=0).astype(np.float32)
    return img_batch

# =========================
# TFLite Prediction
# =========================
def tflite_predict(interpreter, input_details, output_details, img_batch):
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probability = float(output_data[0][0])
    is_fake = probability > THRESHOLD
    return is_fake, probability

# =========================
# Confidence Bar
# =========================
def display_confidence_bar(filename, probability):
    data = pd.DataFrame({
        'Label': ['Real', 'Fake'],
        'Confidence': [1 - probability, probability]
    })
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Confidence:Q', axis=alt.Axis(format='%')),
        y=alt.Y('Label:N', sort=None),
        color=alt.Color('Label:N', scale=alt.Scale(range=['#2ca02c', '#d62728']))
    ).properties(width=400, height=80, title=f"Confidence for {filename}")
    st.altair_chart(chart)

# =========================
# Main App
# =========================
def main():
    # Title and description
    st.markdown("<h1 class='center'>🔍 Deepfake Detection</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='center'>
    This application uses a fine-tuned **EfficientNetB0** deep learning model to detect whether an image is real or a deepfake.
    Upload one or multiple images to get instant predictions along with confidence metrics.  
    Each result is displayed with a confidence bar and summary table.
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.title("About")
    st.sidebar.info("""
    - **Model Architecture:** EfficientNetB0  
    - **Training:** Fine-tuned on real vs fake images  
    - **Prediction Threshold:** 0.5 (probability above this indicates fake)
    """)

    # Load TFLite
    interpreter, input_details, output_details = load_tflite_model()
    if interpreter is None:
        st.stop()

    # Upload images
    uploaded_files = st.file_uploader(
        "Upload one or more images (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            img_batch = preprocess_image(image)
            is_fake, probability = tflite_predict(interpreter, input_details, output_details, img_batch)

            # Display image and result in black-bordered box
            st.markdown("<div class='box'>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption=uploaded_file.name, use_container_width=True)
            with col2:
                st.subheader("Prediction")
                if is_fake:
                    st.error("FAKE IMAGE DETECTED")
                else:
                    st.success("REAL IMAGE")
                st.write(f"Confidence: **{probability*100:.2f}%**")
                display_confidence_bar(uploaded_file.name, probability)
            st.markdown("</div>", unsafe_allow_html=True)

            results.append({
                "Filename": uploaded_file.name,
                "Prediction": "FAKE" if is_fake else "REAL",
                "Confidence": f"{probability*100:.2f}%"
            })

        # Summary table
        st.markdown("### 📊 Summary Table")
        df_results = pd.DataFrame(results)
        for _, row in df_results.iterrows():
            color = '#ffcccc' if row['Prediction'] == 'FAKE' else '#ccffcc'
            st.markdown(
                f"<div style='background-color:{color}; padding:5px; border-radius:5px; border:2px solid black;'>"
                f"<b>{row['Filename']}</b> — {row['Prediction']} ({row['Confidence']})</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("👆 Please upload one or more images to test.")

if __name__ == "__main__":
    main()
