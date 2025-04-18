import streamlit as st
import tensorflow as tf
import numpy as np
import os
import zipfile
import subprocess

# Kaggle model path and output
KAGGLE_MODEL = "saiyednajibullah/pdrss"
MODEL_ZIP = "pdrss.zip"
MODEL_FILE = "trained_model.keras"

# Ensure .kaggle folder exists and set permissions
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# Function to download the model from Kaggle
def download_model():
    if not os.path.exists(MODEL_FILE):
        try:
            subprocess.run([
                "kaggle", "models", "download", "-m", KAGGLE_MODEL,
                "-p", ".", "--unzip"
            ], check=True)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {str(e)}")
    else:
        st.info("📦 Model already downloaded.")

# Tensorflow Model Prediction
def model_prediction(test_image):
    if test_image is None:
        return None, "No image uploaded."

    try:
        model = tf.keras.models.load_model(MODEL_FILE)
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        return result_index, None
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

# Class labels
class_name = [ ... ]  # same as before — list all 38 classes

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg") if os.path.exists("home_page.jpeg") else st.warning("Image not found.")
    st.markdown("Welcome to the Plant Disease Recognition System! 🌿🔍")

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("Dataset info...")

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    download_model()

    st.markdown('<h1 style="color: #2E8B57;">🌿 Plant Disease Recognition System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 20px;">Upload a leaf image to detect disease</p>', unsafe_allow_html=True)

    test_image = st.file_uploader("🖼️ Upload an Image:", type=["jpg", "jpeg", "png"])

    if test_image:
        from PIL import Image
        image = Image.open(test_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Predict Disease"):
            with st.spinner("Analyzing the image..."):
                result_index, error = model_prediction(test_image)
                if error:
                    st.error(f"🚫 Error: {error}")
                else:
                    st.success(f"✅ Prediction: **{class_name[result_index]}**")
