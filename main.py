import streamlit as st
import tensorflow as tf
import numpy as np
import os
import base64
import io
import zipfile
from PIL import Image
import requests

# Download kaggle.json from GitHub
def download_kaggle_json():
    kaggle_json_url = "https://raw.githubusercontent.com/sntarique/PDRS/kaggle.json"
    response = requests.get(kaggle_json_url)
    if response.status_code == 200:
        os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
        with open(os.path.expanduser("~/.kaggle/kaggle.json"), "wb") as f:
            f.write(response.content)
        os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
        st.success("✅ Kaggle API key downloaded and saved.")
    else:
        st.error("🚫 Failed to download kaggle.json")

# Download the model from Kaggle
def download_model_from_kaggle():
    if not os.path.exists("trained_model.keras"):
        download_kaggle_json()
        os.system("kaggle models download saiyednajibullah/pdrs -p .")
        zip_path = "pdrss.zip"
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall()
            st.success("✅ Model downloaded and extracted from Kaggle.")
        else:
            st.error("🚫 Model zip file not found after Kaggle download.")
    else:
        st.success("✅ Model already exists.")

# Tensorflow Model Prediction
def model_prediction(test_image):
    try:
        model = tf.keras.models.load_model('trained_model.keras')
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

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    st.markdown("Welcome to the Plant Disease Recognition System! 🌿🔍 ...")

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("#### Dataset Info ...")

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    download_model_from_kaggle()

    st.markdown('<h1 style="color:#2E8B57;">🌿 Plant Disease Recognition System</h1>', unsafe_allow_html=True)
    st.subheader("Upload a leaf image to detect disease")

    test_image = st.file_uploader("🖼️ Upload an Image:", type=["jpg", "jpeg", "png"])

    if test_image:
        image = Image.open(test_image)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        st.markdown(f'<img src="data:image/png;base64,{img_str}" style="width:70%;border-radius:10px;"/>', unsafe_allow_html=True)

        if st.button("🔍 Predict Disease"):
            with st.spinner("🧠 Analyzing the image..."):
                result_index, error = model_prediction(test_image)
                if error:
                    st.error(f"🚫 Error: {error}")
                else:
                    st.success(f"✅ The model predicts: **{class_name[result_index]}**")
