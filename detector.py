import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import pandas as pd
import altair as alt

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
)

# =========================
# Constants
# =========================
IMAGE_SIZE = (240, 240)
MODEL_PATH = "optimized_model.tflite"
THRESHOLD = 0.5

# =========================
# Load TFLite Model
# =========================
@st.cache_resource
def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# =========================
# Preprocess Image
# =========================
def preprocess_image(image):
    img_array = np.array(image.convert("RGB"))
    img_array = cv2.resize(img_array, IMAGE_SIZE)
    img_preprocessed = img_array / 255.0  # normalize
    img_batch = np.expand_dims(img_preprocessed, axis=0).astype(np.float32)
    return img_batch

# =========================
# Predict Deepfake with TFLite
# =========================
def predict_tflite(interpreter, input_details, output_details, img_batch):
    interpreter.set_tensor(input_details[0]["index"], img_batch)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    probability = float(output_data[0][0])
    is_fake = probability > THRESHOLD
    return is_fake, probability

# =========================
# Display Confidence Bar
# =========================
def display_confidence_bar(probability):
    data = pd.DataFrame({
        "Label": ["Real", "Fake"],
        "Confidence": [1 - probability, probability]
    })
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("Confidence:Q", axis=alt.Axis(format="%")),
        y=alt.Y("Label:N", sort=None),
        color=alt.Color("Label:N", scale=alt.Scale(range=["#2ca02c", "#d62728"]))
    ).properties(width=300, height=80)
    st.altair_chart(chart)

# =========================
# Main App
# =========================
def main():
    # Background white
    st.markdown(
        """
        <style>
        .main {background-color: #ffffff;}
        .stImage img {max-width: 100%; height: auto;}
        .stButton>button {border: 2px solid black;}
        .stDataFrame>div {border: 2px solid black;}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title & Description
    st.markdown("<h1 style='text-align: center;'>🔍 Deepfake Detection</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size:18px;'>"
        "This app detects whether an uploaded image is real or a deepfake using a fine-tuned "
        "<b>EfficientNetB0</b> model converted to TensorFlow Lite for fast inference. "
        "Upload one or more images and get predictions with confidence scores."
        "</p>",
        unsafe_allow_html=True
    )

    # Sidebar info
    st.sidebar.title("About Model")
    st.sidebar.info("""
    - Architecture: EfficientNetB0  
    - Training: Fine-tuned on real vs fake images  
    - Prediction threshold: 0.5 (probability >0.5 → FAKE)  
    """)

    # Load TFLite interpreter
    interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)

    # Upload images
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
            is_fake, probability = predict_tflite(interpreter, input_details, output_details, img_batch)

            # Layout: image and prediction
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("<div style='border:2px solid black; padding:5px;'>Uploaded Image</div>", unsafe_allow_html=True)
                st.image(image, use_container_width=True)

            with col2:
                st.markdown("<div style='border:2px solid black; padding:5px;'>Prediction</div>", unsafe_allow_html=True)
                if is_fake:
                    st.error(f"FAKE IMAGE DETECTED")
                else:
                    st.success(f"REAL IMAGE")

                st.write(f"Confidence: **{probability*100:.2f}%**")
                display_confidence_bar(probability)

            results.append({
                "Filename": uploaded_file.name,
                "Prediction": "FAKE" if is_fake else "REAL",
                "Confidence": f"{probability*100:.2f}%"
            })

        # Summary Table
        st.markdown("<h3>📊 Summary</h3>", unsafe_allow_html=True)
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, width=800)

    else:
        st.info("👆 Upload one or more images to check for deepfakes.")

if __name__ == "__main__":
    main()
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
MODEL_PATH = 'deepfake_inference_model.keras'
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
