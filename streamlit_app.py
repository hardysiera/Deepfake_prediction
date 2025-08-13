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
    page_icon="🧠",  # Change this to any emoji from the table above
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
# Set White Background
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
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
TFLITE_MODEL_PATH = "optimized_model.tflite"
THRESHOLD = 0.5

# =========================
# Load TFLite Model
# =========================
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

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
    # Apply sigmoid in case output is logits
    probability = float(tf.sigmoid(output_data[0][0]))
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
    st.markdown(
        "<h1 style='text-align: center; color: black;'>🔍 Deepfake Image Detector</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: black;'>"
        "Upload one or more images and the model will predict whether they are real or deepfake, "
        "providing a confidence score for each."
        "</p>", unsafe_allow_html=True
    )
    
    st.markdown("<hr style='border: 1px solid black;'>", unsafe_allow_html=True)

    interpreter, input_details, output_details = load_tflite_model()

    uploaded_files = st.file_uploader(
        "Upload JPG, JPEG, or PNG images:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            img_batch = preprocess_image(image)
            is_fake, probability = tflite_predict(interpreter, input_details, output_details, img_batch)

            # Display image and results
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption=uploaded_file.name, use_container_width=True)
            with col2:
                st.markdown(
                    f"<div style='border: 1px solid black; padding: 10px;'>"
                    f"<h3>Prediction:</h3>"
                    f"{'FAKE' if is_fake else 'REAL'}<br>"
                    f"Confidence: {probability*100:.2f}%"
                    f"</div>", unsafe_allow_html=True
                )
                display_confidence_bar(uploaded_file.name, probability)

            results.append({
                "Filename": uploaded_file.name,
                "Prediction": "FAKE" if is_fake else "REAL",
                "Confidence": f"{probability*100:.2f}%"
            })

        # Summary Table
        st.markdown("<h2 style='color:black;'>📊 Summary Table</h2>", unsafe_allow_html=True)
        df_results = pd.DataFrame(results)
        st.dataframe(df_results.style.set_table_styles([
            {'selector': '', 'props': [('border', '1px solid black')]}
        ]))
    else:
        st.info("👆 Please upload images to detect deepfakes.")

if __name__ == "__main__":
    main()
