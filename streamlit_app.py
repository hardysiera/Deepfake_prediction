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
# CSS Styling for White Background and Black Text with Visual Enhancements
# =========================
st.markdown(
    """
    <style>
    /* Global styles for white background and black text */
    html, body, .stApp {
        background-color: white !important;
        color: black !important;
        font-family: 'Inter', sans-serif !important; /* Apply Inter font for a modern look */
    }

    /* Ensure specific Streamlit text components are black */
    .stMarkdown, .stText, .stSpinner div, .stProgress div, .stDownloadButton {
        color: black !important;
    }

    /* Sidebar background and styling */
    [data-testid="stSidebar"] {
        background-color: white !important;
        color: black !important;
        box-shadow: 2px 0 5px -2px rgba(0,0,0,0.1); /* Subtle shadow for visual depth */
        border-radius: 0.75rem !important; /* Rounded corners for the sidebar itself */
    }

    /* Headers color */
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }

    /* Apply rounded corners and shadows to various Streamlit elements for a unified look */
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
        border-radius: 0.75rem !important; /* Tailwind's rounded-lg equivalent */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* Subtle shadow effect */
    }

    /* File uploader button styling */
    .stFileUpload > div > button {
        background-color: #f0f2f6 !important; /* A light gray background for the button */
        color: black !important; /* Black text */
        border: 1px solid black !important; /* Black border */
        padding: 0.5rem 1rem !important; /* Comfortable padding */
        transition: all 0.2s ease-in-out !important; /* Smooth transition for hover effects */
    }
    .stFileUpload > div > button:hover {
        background-color: #e2e8f0 !important; /* Slightly darker gray on hover */
        transform: translateY(-2px) !important; /* Lifts button on hover */
    }
    .stFileUpload > div > button:active {
        transform: translateY(0) !important; /* Pushes button down on click */
        box-shadow: none !important; /* Removes shadow on click */
    }

    /* General button styling (e.g., if you add a 'Clear Results' button) */
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

    /* Specific styling for the image display container */
    .stImage > div {
        border: 2px solid black !important;
        padding: 5px !important;
        background-color: white !important;
    }

    /* Specific styling for the prediction result box */
    .prediction-box {
        border: 2px solid black !important;
        padding: 15px !important; /* Increased padding for better appearance */
        background-color: white !important;
        margin-bottom: 1rem; /* Space below the box */
    }

    /* Dataframe container styling */
    .stDataFrame {
        border: 2px solid black !important;
        overflow: hidden !important; /* Ensures border-radius applies correctly to content */
    }

    /* Styling for Streamlit's alert boxes (info, success, error) */
    .stAlert {
        border-radius: 0.75rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Constants
# =========================
IMAGE_SIZE = (240, 240) # Desired image size for model input
MODEL_PATH = 'optimized_model.tflite' # Path to your TFLite model file
THRESHOLD = 0.5 # Probability threshold to classify as fake

# =========================
# Load TFLite Model
# =========================
@st.cache_resource # Caches the loaded model to prevent reloading on every Streamlit rerun
def load_tflite_model():
    """
    Loads the TFLite model from the specified path.
    Handles potential errors during model loading.
    """
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors() # Allocate tensors required for inference
        input_details = interpreter.get_input_details() # Get input tensor details
        output_details = interpreter.get_output_details() # Get output tensor details
        return interpreter, input_details, output_details
    except Exception as e:
        # Display a comprehensive error message if the model cannot be loaded
        st.error(f"🚨 **Error:** Could not load the TFLite model from `{MODEL_PATH}`. "
                 "Please ensure the `optimized_model.tflite` file is in the same directory as this script. "
                 f"**Details:** `{e}`")
        return None, None, None # Return None if loading fails

# =========================
# Preprocess Image
# =========================
def preprocess_image(image):
    """
    Preprocesses the input image for model inference.
    Steps: Convert PIL Image to numpy array, resize, apply EfficientNet preprocessing,
    and expand dimensions for batch prediction.
    """
    img_array = np.array(image) # Convert PIL Image object to NumPy array (RGB format)

    # ORIGINAL CODE: if img_array.shape[-1] == 3: img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    # Rationale for removal/comment:
    # tf.keras.applications.efficientnet.preprocess_input typically expects RGB input
    # as Keras models are usually trained with RGB images. PIL.Image.open() loads
    # images in RGB. Converting RGB to BGR here would feed the model BGR data,
    # which is likely incorrect for a standard EfficientNet model trained with Keras.
    # If your specific model was trained to expect BGR data after this conversion,
    # then this line should be re-enabled. For standard use, it's removed.

    img_resized = cv2.resize(img_array, IMAGE_SIZE) # Resize image to model's expected input size
    # Apply EfficientNet's specific preprocessing (e.g., scaling, mean subtraction)
    img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_resized)
    # Add a batch dimension (e.g., from (H, W, C) to (1, H, W, C)) and ensure float32 type
    img_batch = np.expand_dims(img_preprocessed, axis=0).astype(np.float32)
    return img_batch

# =========================
# Predict Deepfake
# =========================
def tflite_predict(interpreter, input_details, output_details, img_batch):
    """
    Performs inference using the loaded TFLite interpreter.
    Sets input tensor, invokes interpreter, and retrieves output tensor.
    """
    interpreter.set_tensor(input_details[0]['index'], img_batch) # Set the input tensor data
    interpreter.invoke() # Run inference
    output_data = interpreter.get_tensor(output_details[0]['index']) # Get the output tensor data
    # The model outputs a single probability score (e.g., for the "fake" class)
    probability = float(output_data[0][0])
    is_fake = probability > THRESHOLD # Classify based on the threshold
    return is_fake, probability

# =========================
# Display Confidence Bar
# =========================
def display_confidence_bar(filename, probability):
    """
    Displays a horizontal bar chart visualizing the confidence of
    "Real" vs. "Fake" predictions using Altair.
    """
    # Create a DataFrame for Altair chart
    data = pd.DataFrame({
        'Label': ['Real', 'Fake'],
        'Confidence': [1 - probability, probability] # Real confidence is 1 - fake probability
    })
    
    chart = alt.Chart(data).mark_bar().encode(
        # X-axis for confidence, formatted as a percentage with one decimal place
        x=alt.X('Confidence:Q', axis=alt.Axis(format='.1%', title='Confidence')),
        # Y-axis for labels, preventing alphabetical sorting
        y=alt.Y('Label:N', sort=None, title=''),
        # Color the bars based on 'Label', using more vibrant green for Real and red for Fake
        color=alt.Color('Label:N', scale=alt.Scale(range=['#28a745', '#dc3545']), legend=None) # No legend needed as labels are on Y-axis
    ).properties(
        width=400, # Fixed width for the chart
        height=70, # Adjusted height for a more compact look
        title={
            "text": f"Confidence for {filename}", # Chart title with filename
            "anchor": "middle", # Center the title
            "fontSize": 16, # Font size for the title
            "color": "black" # Title color
        }
    ).configure_axis(
        # Configure axis labels and titles to be black
        labelColor='black',
        titleColor='black'
    ).configure_view(
        stroke='transparent' # Remove the outer border of the chart view
    )
    st.altair_chart(chart, use_container_width=True) # Display the chart, making it responsive

# =========================
# Main App
# =========================
def main():
    """
    Main function to run the Streamlit Deepfake Image Detector application.
    Orchestrates UI, file uploads, prediction, and result display.
    """
    # Custom styled main title
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 0.5rem;'>🕵️‍♂️ Deepfake Image Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #555;'>Unmasking synthetic imagery with AI</p>", unsafe_allow_html=True)
    st.markdown("---") # Visual separator

    st.markdown("""
    This application leverages a **fine-tuned EfficientNetB0 TFLite model** to discern whether an image is
    authentic or a synthetically generated deepfake. Simply upload one or more images below, and
    the app will provide a real-time analysis, including a confidence score and visual metrics.
    """)

    # Sidebar for 'About' information
    st.sidebar.title("✨ About This Detector")
    st.sidebar.info("""
    - **Model Architecture:** EfficientNetB0 (a highly efficient and accurate convolutional neural network)
    - **Model Type:** TFLite Optimized (ideal for fast, on-device inference)
    - **Prediction Threshold:** `0.5` (images with a probability greater than `0.5` are classified as fake)
    - **Supported Uploads:** JPG, JPEG, PNG image formats
    """)
    

    # Load TFLite model at the start
    interpreter, input_details, output_details = load_tflite_model()
    if interpreter is None:
        st.stop() # Stop the application if the model fails to load

    st.markdown("---") # Visual separator
    st.subheader("📤 Upload Your Images for Analysis")
    uploaded_files = st.file_uploader(
        "Choose image files (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, # Allow multiple file uploads
        help="Upload one or more images to check if they are real or deepfakes. Max file size: 200MB."
    )

    if uploaded_files:
        results = [] # List to store prediction results for summary table
        st.markdown("---") # Visual separator
        st.subheader("🔍 Analysis Results")

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file) # Open the uploaded image using PIL
            st.write(f"### Analyzing: **{uploaded_file.name}**")

            # Use st.spinner for a loading indicator during the prediction process
            with st.spinner("Processing image and making a prediction..."):
                img_batch = preprocess_image(image) # Preprocess the image for the model
                is_fake, probability = tflite_predict(interpreter, input_details, output_details, img_batch) # Get prediction

            # Layout for displaying image and its prediction result side-by-side
            col1, col2 = st.columns([1, 2]) # Image column (1 part width), Prediction info column (2 parts width)

            with col1:
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

            with col2:
                # Custom styled prediction result box
                st.markdown(f"<div class='prediction-box'>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin-top: 0; color: black;'>Prediction Outcome</h3>", unsafe_allow_html=True)

                if is_fake:
                    st.error("🚨 **FAKE IMAGE DETECTED!** This image likely originated from an AI.")
                else:
                    st.success("✅ **REAL IMAGE.** This image appears to be authentic.")

                st.markdown(f"**Confidence:** `{probability*100:.2f}%`") # Display confidence percentage
                display_confidence_bar(uploaded_file.name, probability) # Display confidence bar chart
                st.markdown("</div>", unsafe_allow_html=True)

            # Append results to the list for the summary table
            results.append({
                "Filename": uploaded_file.name,
                "Prediction": "FAKE" if is_fake else "REAL",
                "Confidence": f"{probability*100:.2f}%"
            })
            st.markdown("---") # Separator after each image's results for clarity

        # Display overall summary table after all images are processed
        st.markdown("### 📊 Overall Summary")
        df_results = pd.DataFrame(results) # Convert results list to DataFrame
        st.dataframe(df_results, use_container_width=True) # Display DataFrame, responsive to container width
    else:
        # Message displayed when no files are uploaded
        st.info("👆 Please upload one or more images above to initiate the deepfake detection process.")
        st.markdown("---")
        st.markdown("Feel free to explore the sidebar for more details about the model and its capabilities.")

if __name__ == "__main__":
    main()
