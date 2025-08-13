import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
import tempfile
import os
import pandas as pd
import altair as alt

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Disentangling the Real: Deepfake Detection",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# =========================
# CSS Styling
# =========================
st.markdown(
    """
    <style>
    html, body, .stApp { background: #141414 !important; color: #e5e5e5 !important; font-family: 'Inter', sans-serif !important; }
    h1,h2,h3,h4,h5,h6 { color: #e5e5e5 !important; font-weight: 700 !important; }
    [data-testid="stSidebar"] { background-color: #1f1f1f !important; color: #e5e5e5 !important; box-shadow: 2px 0 5px -2px rgba(0,0,0,0.5); border-radius: 0.75rem !important; padding:1rem !important; }
    .stFileUpload > div > button { background-color: #00b894 !important; color: white !important; border: none !important; padding:0.6rem 1.2rem !important; border-radius: 1rem !important; font-weight:600; box-shadow:0 5px 15px rgba(0,0,0,0.3); transition: all 0.3s ease-in-out !important;}
    .stFileUpload > div > button:hover { background-color:#00d88a !important; transform:translateY(-3px) scale(1.05) !important; box-shadow:0 8px 20px rgba(0,0,0,0.4); }
    .stFileUpload > div > button:active { transform:translateY(0) scale(1) !important; box-shadow:0 4px 10px rgba(0,0,0,0.3);}
    .stButton > button { background-color:#2d2d2d !important; color:#00b894 !important; border:1px solid #00b894 !important; border-radius:1rem !important; padding:0.6rem 1.2rem !important; font-weight:600; box-shadow:0 4px 8px rgba(0,0,0,0.3); transition: all 0.3s ease-in-out !important; margin:0.2rem !important; min-width:100px !important;}
    .stButton > button:hover { background-color:#00b894 !important; color:#141414 !important; transform:translateY(-3px) scale(1.05) !important; box-shadow:0 8px 20px rgba(0,0,0,0.5);}
    .stImage > div { border-radius:1rem !important; max-width:400px !important; margin:1rem auto !important; overflow:hidden !important; box-shadow:0 8px 20px rgba(0,0,0,0.5); border:2px solid #2d2d2d !important; background-color:#1f1f1f !important; transition: transform 0.3s ease;}
    .stImage > div:hover { transform:scale(1.02);}
    .prediction-box { border-radius:1rem !important; background-color:#2d2d2d !important; color:#e5e5e5 !important; padding:1rem !important; box-shadow:0 8px 20px rgba(0,0,0,0.5); margin-bottom:1rem;}
    .prediction-buttons { display:flex !important; flex-direction:row !important; justify-content:flex-start !important; gap:1rem !important; margin-top:0.5rem !important; }
    .prediction-buttons .stButton > button { flex:1; }
    .altair-Chart svg { background-color:#1f1f1f !important; }
    .stDataFrame { border-radius:1rem !important; overflow:hidden !important; border:2px solid #2d2d2d !important; background-color:#1f1f1f !important; color:#e5e5e5 !important; }
    .stAlert { border-radius:1rem !important; box-shadow:0 8px 20px rgba(0,0,0,0.4);}
    ::-webkit-scrollbar { width:10px; }
    ::-webkit-scrollbar-thumb { background:#00b894; border-radius:5px; }
    ::-webkit-scrollbar-track { background:#1f1f1f; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Constants
# =========================
IMAGE_SIZE = (240, 240)
THRESHOLD = 0.5
MODEL_PATH = "optimized_model.tflite"

# =========================
# ELA preprocessing
# =========================
def perform_ela(image_path, rescale_size=IMAGE_SIZE):
    quality = 90
    image = Image.open(image_path).convert('RGB')
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

def preprocess_image(image_path, input_dtype):
    ela_img = perform_ela(image_path)
    ela_img = tf.image.resize(ela_img, IMAGE_SIZE)
    if input_dtype == np.float32:
        ela_img = tf.cast(ela_img, tf.float32)
        ela_img = tf.keras.applications.efficientnet.preprocess_input(ela_img)
    else:
        ela_img = tf.cast(ela_img, tf.uint8)
    return np.expand_dims(ela_img, axis=0)

# =========================
# Load TFLite model
# =========================
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_model()
input_dtype = input_details[0]['dtype']

# =========================
# Prediction function
# =========================
def tflite_predict(image_path):
    input_data = preprocess_image(image_path, input_dtype)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]['index'])[0]
    if len(raw_output.shape) == 0:  # scalar
        prob = float(raw_output)
    elif len(raw_output) == 1:  # single probability
        prob = float(raw_output[0])
    elif len(raw_output) == 2:  # softmax
        prob = float(raw_output[1])
    else:
        raise ValueError(f"Unexpected model output shape: {raw_output.shape}")
    return prob

# =========================
# Display confidence bar
# =========================
def display_confidence_bar(filename, probability):
    data = pd.DataFrame({
        'Label': ['Real', 'Fake'],
        'Confidence': [1 - probability, probability]
    })
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Confidence:Q', axis=alt.Axis(format='.1%', title='Confidence')),
        y=alt.Y('Label:N', sort=None, title=''),
        color=alt.Color('Label:N', scale=alt.Scale(range=['#28a745','#dc3545']), legend=None)
    ).properties(
        width=400, height=70,
        title={"text": f"Confidence for {filename}", "anchor":"middle", "fontSize":16, "color":"#e5e5e5"}
    ).configure_axis(labelColor='#e5e5e5', titleColor='#e5e5e5').configure_view(stroke='transparent')
    st.altair_chart(chart, use_container_width=True)

# =========================
# Main App
# =========================
def main():
    st.markdown("<h1 style='text-align:center;font-size:3rem;margin-bottom:0.5rem;'>🕵️‍♂️ Disentangling the Real: Deepfake Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:1rem;color:#aaaaaa;'>Detect synthetic imagery with AI</p>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_files = st.file_uploader("Upload images (JPG, PNG, JPEG):", type=['jpg','jpeg','png'], accept_multiple_files=True)

    if uploaded_files:
        results = []
        st.subheader("🔍 Analysis Results")
        for uploaded_file in uploaded_files:
            image_path = uploaded_file
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(image, caption=uploaded_file.name, use_column_width=True)

            with col2:
                prob = tflite_predict(image_path)
                st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin-top:0;'>Prediction Outcome</h3>", unsafe_allow_html=True)

                st.markdown("<div class='prediction-buttons'>", unsafe_allow_html=True)
                real_btn = st.button("REAL", key=f"real_{uploaded_file.name}")
                fake_btn = st.button("FAKE", key=f"fake_{uploaded_file.name}")
                st.markdown("</div>", unsafe_allow_html=True)

                if prob >= THRESHOLD:
                    st.error("🚨 FAKE IMAGE DETECTED!")
                else:
                    st.success("✅ REAL IMAGE")

                st.markdown(f"**Confidence:** {prob*100:.2f}%")
                display_confidence_bar(uploaded_file.name, prob)
                st.markdown("</div>", unsafe_allow_html=True)

                results.append({"Filename": uploaded_file.name,
                                "Prediction": "FAKE" if prob>=THRESHOLD else "REAL",
                                "Confidence": f"{prob*100:.2f}%"})

        st.markdown("### 📊 Overall Summary")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.info("Upload one or more images to check for deepfakes.")

if __name__ == "__main__":
    main()
