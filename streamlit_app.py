import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
import tempfile
import os

# ================= CONFIGURATION =================
IMAGE_SIZE = (240, 240)
THRESHOLD = 0.5
TFLITE_MODEL_PATH = "optimized_model.tflite"

# ================= ELA PREPROCESSING =================
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

def preprocess_ela_image(image_path):
    ela_img = perform_ela(image_path)
    ela_img = tf.image.resize(ela_img, IMAGE_SIZE)
    ela_img = tf.cast(ela_img, tf.float32)
    ela_img = tf.keras.applications.efficientnet.preprocess_input(ela_img)
    return np.expand_dims(ela_img, axis=0)

# ================= LOAD TFLITE MODEL =================
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model(TFLITE_MODEL_PATH)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ================= PREDICTION FUNCTION =================
def predict_image(image_path):
    img_array = preprocess_ela_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]  # Assuming binary output
    return float(prediction)

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Deepfake Detection", page_icon="🛡️", layout="centered")

st.markdown(
    """
    <style>
    body {
        background-color: white;
        color: black;
    }
    .box {
        border: 2px solid black;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛡️ Deepfake Detection")
st.write("Upload an image to detect if it's Real or Fake using Error Level Analysis (ELA) + EfficientNet.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img.save("temp_uploaded_image.jpg")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        confidence = predict_image("temp_uploaded_image.jpg")
        label = "Fake" if confidence >= THRESHOLD else "Real"
        confidence_percentage = confidence * 100 if label == "Fake" else (1 - confidence) * 100

    st.markdown(f"<div class='box'><b>Prediction:</b> {label}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='box'><b>Confidence:</b> {confidence_percentage:.2f}%</div>", unsafe_allow_html=True)
