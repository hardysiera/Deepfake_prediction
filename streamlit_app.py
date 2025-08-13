import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageChops, ImageEnhance
import tempfile
import os

# Config
IMAGE_SIZE = (240, 240)
THRESHOLD = 0.5
MODEL_PATH = "model_optimized.tflite"

# --- ELA Function ---
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

# --- Preprocess function ---
def preprocess_image(image_path, input_dtype):
    ela_img = perform_ela(image_path)
    ela_img = tf.image.resize(ela_img, IMAGE_SIZE)

    if input_dtype == np.float32:
        # Same normalization as EfficientNet training
        ela_img = tf.cast(ela_img, tf.float32)
        ela_img = tf.keras.applications.efficientnet.preprocess_input(ela_img)
    else:
        # For quantized models, keep as uint8 0-255
        ela_img = tf.cast(ela_img, tf.uint8)

    return np.expand_dims(ela_img, axis=0)

# --- Load TFLite model ---
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_dtype = input_details[0]['dtype']

st.title("Deepfake Detection (Debug Mode)")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Preprocess image
    input_data = preprocess_image(tmp_path, input_dtype)

    # Debug: show preprocessing info
    st.write("**Input details:**", input_details)
    st.write("**Model expects dtype:**", input_dtype)
    st.write("**Preprocessed input shape:**", input_data.shape)
    st.write("**Input data min/max:**", np.min(input_data), np.max(input_data))

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Debug: Show raw model output
    st.write("**Raw model output:**", raw_output)

    # Auto-detect output format
    if len(raw_output.shape) == 0:  # Single scalar
        fake_prob = float(raw_output)
    elif len(raw_output) == 1:      # Single probability in array
        fake_prob = float(raw_output[0])
    elif len(raw_output) == 2:      # Softmax with 2 outputs
        # Assuming [real_prob, fake_prob]
        fake_prob = float(raw_output[1])
    else:
        st.error("Unexpected model output shape")
        fake_prob = None

    # Final prediction
    if fake_prob is not None:
        st.write(f"**Fake Probability:** {fake_prob:.4f}")
        if fake_prob >= THRESHOLD:
            st.error("Prediction: FAKE")
        else:
            st.success("Prediction: REAL")

    os.remove(tmp_path)
