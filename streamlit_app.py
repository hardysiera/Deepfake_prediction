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
    page_title="Deepfake Detector",
    page_icon="🛡️",  
    layout="wide"
)

# =========================
# Custom CSS for styling
# =========================
st.markdown(
    """
    <style>
    /* Page background and font */
    .stApp {
        background-color: #ffffff;
        color: #000000;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Card styling */
    .card {
        border: 2px solid #000000;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 25px;
        background-color: #f9f9f9;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card:hover {
        transform: scale(1.02);
        box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
    }
    .center {
        text-align: center;
    }
    .confidence-bar {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    h1, h3, h4 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True
)

# =========================
# Constants
# =========================
IMAGE_SIZE = (240, 240)
MODEL_PATH = 'deepfake_inference_model.keras'  # Keras or .h5 file
THRESHOLD = 0.5

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# =========================
# Preprocess Image
# =========================
def preprocess_image(image):
    img_array = np.array(image)
    if img_array.shape[-1] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_array, IMAGE_SIZE)
    img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_resized)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    return img_batch

# =========================
# Prediction
# =========================
def predict_deepfake(model, img_batch):
    prediction = model.predict(img_batch, verbose=0)
    probability = float(prediction[0][0])
    is_fake = probability > THRESHOLD
    return is_fake, probability

# =========================
# Visual Confidence Bar
# =========================
def display_confidence_bar(probability):
    data = pd.DataFrame({
        'Label': ['Real', 'Fake'],
        'Confidence': [1 - probability, probability]
    })
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Confidence:Q', axis=alt.Axis(format='%')),
        y=alt.Y('Label:N', sort=None),
        color=alt.Color('Label:N', scale=alt.Scale(range=['#2ca02c', '#d62728'])),
        tooltip=['Label:N', alt.Tooltip('Confidence:Q', format='.2%')]
    ).properties(width=400, height=80)
    st.altair_chart(chart)

# =========================
# Main App
# =========================
def main():
    st.markdown('<h1 class="center">🔍 Deepfake Image Detector</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="center">Upload one or multiple images and our AI model (EfficientNetB0) will analyze them. You will see if they are real or deepfake along with a confidence score.</p>',
        unsafe_allow_html=True
    )

    # Sidebar info
    st.sidebar.title("About")
    st.sidebar.info("""
    - **Model Architecture:** EfficientNetB0  
    - **Training:** Fine-tuned on real vs fake images  
    - **Prediction Threshold:** 0.5  
    - **Hover over the cards** to see subtle animation effect
    """)

    # Load model
    model = load_model()
    if model is None:
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
            is_fake, probability = predict_deepfake(model, img_batch)

            # Card container for each image
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, caption=uploaded_file.name, use_container_width=True)

            st.markdown('<h3 class="center">Prediction</h3>', unsafe_allow_html=True)
            if is_fake:
                st.markdown(f'<h4 style="color:red; text-align:center;">FAKE IMAGE DETECTED</h4>', unsafe_allow_html=True)
            else:
                st.markdown(f'<h4 style="color:green; text-align:center;">REAL IMAGE</h4>', unsafe_allow_html=True)

            st.markdown(f'<p class="center" title="Confidence Tooltip">Confidence: <b>{probability*100:.2f}%</b></p>', unsafe_allow_html=True)
            display_confidence_bar(probability)
            st.markdown('</div>', unsafe_allow_html=True)

            results.append({
                "Filename": uploaded_file.name,
                "Prediction": "FAKE" if is_fake else "REAL",
                "Confidence": f"{probability*100:.2f}%"
            })

        # Summary Table
        st.markdown('<h3 class="center">📊 Summary Table</h3>', unsafe_allow_html=True)
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

    else:
        st.info("👆 Please upload one or more images to test.")

if __name__ == "__main__":
    main()
