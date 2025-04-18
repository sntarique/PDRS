import streamlit as st
import tensorflow as tf
import numpy as np
import os
import base64
from PIL import Image
import io
import requests
import zipfile
import json

# Set up Kaggle credentials
KAGGLE_JSON_URL = "https://raw.githubusercontent.com/sntarique/PDRS/main/kaggle.json"
kaggle_path = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_path, exist_ok=True)
kaggle_json_path = os.path.join(kaggle_path, "kaggle.json")

if not os.path.exists(kaggle_json_path):
    r = requests.get(KAGGLE_JSON_URL)
    with open(kaggle_json_path, "wb") as f:
        f.write(r.content)
    os.chmod(kaggle_json_path, 0o600)

# Download model from Kaggle if not present
MODEL_PATH = "trained_model.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Kaggle..."):
        os.system("kaggle models download saiyednajibullah/pdrss --model pdrss-model --path .")
        zip_path = "pdrss-model.zip"
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove(zip_path)
            st.success("Model downloaded and extracted successfully!")
        else:
            st.error("Failed to download model zip from Kaggle.")

# TensorFlow Model Prediction
def model_prediction(test_image):
    if test_image is None:
        return None, "No image uploaded."
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except (IOError, OSError, ValueError) as e:
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
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Streamlit App
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.markdown("""
        Welcome to the Plant Disease Recognition System! 🌿🔍

        Upload a leaf image to detect diseases using a trained deep learning model.
        
        - **Accurate**: Built using TensorFlow & a large dataset.
        - **Fast**: Real-time predictions.
        - **Easy to Use**: Just upload and click predict.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
        #### Dataset Info
        - 87,000+ plant leaf images
        - 38 different classes
        - Trained and validated on high-quality dataset
    """)

elif app_mode == "Disease Recognition":
    st.markdown("""
        <style>
            .title { font-size: 40px; font-weight: bold; color: #2E8B57; margin-bottom: 0.5rem; }
            .subtitle { font-size: 20px; margin-bottom: 1.5rem; }
            .uploaded-image { width: 70%; margin: 0 auto; display: block; border-radius: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2); }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">🌿 Plant Disease Recognition System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a leaf image to detect disease</div>', unsafe_allow_html=True)

    test_image = st.file_uploader("🖼️ Upload an Image:", type=["jpg", "jpeg", "png"])

    if test_image:
        image = Image.open(test_image)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        st.markdown(f'<img src="data:image/png;base64,{img_str}" class="uploaded-image"/>', unsafe_allow_html=True)

        if st.button("🔍 Predict Disease"):
            with st.spinner("🧠 Analyzing the image..."):
                result_index, error = model_prediction(test_image)
                if error:
                    st.error(f"🚫 Error: {error}")
                else:
                    st.success(f"✅ The model predicts: **{class_name[result_index]}**")
