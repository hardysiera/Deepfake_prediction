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
    page_icon="🔍",
    layout="wide"
)

# =========================
# Constants
# =========================
IMAGE_SIZE = (240, 240)
MODEL_PATH = 'optimized_model.tflte'
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
    st.title("🔍 Deepfake Image Detector")
    st.markdown("""
    This application uses a fine-tuned **EfficientNetB0** model to detect whether an image is real or a deepfake.
    Upload one or multiple images and get instant predictions with confidence metrics.
    """)

    # Sidebar info
    st.sidebar.title("About")
    st.sidebar.info("""
    - **Model Architecture:** EfficientNetB0  
    - **Training:** Fine-tuned on real vs fake images  
    - **Prediction Threshold:** 0.5 (probability above this indicates fake)
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

            # Display image and result
            col1, col2 = st.columns([1,2])
            with col1:
                st.image(image, caption=uploaded_file.name, use_column_width=True)

            with col2:
                st.subheader("Prediction")
                if is_fake:
                    st.error(f"FAKE IMAGE DETECTED")
                else:
                    st.success(f"REAL IMAGE")

                st.write(f"Confidence: **{probability*100:.2f}%**")

                # Graphical confidence bar
                display_confidence_bar(uploaded_file.name, probability)

            results.append({
                "Filename": uploaded_file.name,
                "Prediction": "FAKE" if is_fake else "REAL",
                "Confidence": f"{probability*100:.2f}%"
            })

        # =========================
        # Summary Table
        # =========================
        st.markdown("### 📊 Summary Table")
        df_results = pd.DataFrame(results)

        def highlight_rows(row):
            color = '#ffcccc' if row['Prediction'] == 'FAKE' else '#ccffcc'
            return [color]*len(row)

        st.dataframe(df_results.style.apply(highlight_rows, axis=1))
    else:
        st.info("👆 Please upload one or more images to test.")

if __name__ == "__main__":
    main()
