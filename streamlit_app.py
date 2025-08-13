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
    page_title="Deepfake Image Detector",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# =========================
# CSS Styling for White Background and Black Text
# =========================
st.markdown(
    """
    <style>
    /* Entire page background */
    body {
        background-color: white;
        color: black;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: white;
        color: black;
    }

    /* Headers and text */
    .css-1d391kg, .css-18ni7ap, .stMarkdown p, .stButton button {
        color: black;
    }

    /* Image container */
    .stImage > div {
        border: 2px solid black;
        padding: 5px;
        background-color: white;
    }

    /* Prediction box */
    .prediction-box {
        border: 2px solid black;
        padding: 10px;
        background-color: white;
    }

    /* Dataframe container */
    .stDataFrame, .st-ag-grid {
        border: 2px solid black;
    }

    /* File uploader */
    .stFileUpload {
        background-color: white;
        border: 2px solid black;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Constants
# =========================
IMAGE_SIZE = (240, 240)
MODEL_PATH = 'optimized_model.tflite'
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
# Predict Deepfake
# =========================
def tflite_predict(interpreter, input_details, output_details, img_batch):
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probability = float(output_data[0][0])
    is_fake = probability > THRESHOLD
    return is_fake, probability

# =========================
# Display Confidence Bar
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
    st.altair_chart(chart, use_container_width=True)

# =========================
# Main App
# =========================
def main():
    st.title("🔍 Deepfake Image Detector")
    st.markdown("""
    This application uses a fine-tuned **EfficientNetB0** TFLite model to detect whether an image is real or a deepfake.  
    Upload one or more images and the app will analyze them with probability confidence and visual metrics.
    """)

    # Sidebar info
    st.sidebar.title("About")
    st.sidebar.info("""
    - **Model Architecture:** EfficientNetB0  
    - **Model Type:** TFLite Optimized  
    - **Prediction Threshold:** 0.5 (probability above this indicates fake)  
    - **Upload:** JPG, JPEG, PNG images
    """)

    # Load TFLite model
    interpreter, input_details, output_details = load_tflite_model()
    if interpreter is None:
        st.stop()

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload one or more images:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            img_batch = preprocess_image(image)
            is_fake, probability = tflite_predict(interpreter, input_details, output_details, img_batch)

            # Layout for image and prediction
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption=uploaded_file.name, use_container_width=True)

            with col2:
                st.markdown(f"<div class='prediction-box'>", unsafe_allow_html=True)
                st.subheader("Prediction Result")
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
        st.dataframe(df_results, use_container_width=True)
    else:
        st.info("👆 Please upload one or more images to test.")

if __name__ == "__main__":
    main()
