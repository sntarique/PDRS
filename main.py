import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown

# Model file URL and path
MODEL_URL = "https://drive.google.com/uc?export=download&id=13rUvQVsdwfAPGsMhJvxppPus3wagToOG"
OUTPUT_PATH = 'trained_model.keras'

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(OUTPUT_PATH):
        try:
            gdown.download(MODEL_URL, OUTPUT_PATH, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")
    else:
        st.success("Model file already downloaded!")

# Tensorflow Model Prediction
def model_prediction(test_image):
    if test_image is None:
        return None, "No image uploaded."

    # Check if model exists
    if not os.path.exists(OUTPUT_PATH):
        return None, f"Model file '{OUTPUT_PATH}' not found. Please ensure it's in the app directory."

    # Check current working directory
    st.write(f"Current working directory: {os.getcwd()}")  # Debugging current working directory

    # Verify model file location
    if os.path.exists(OUTPUT_PATH):
        st.success(f"Model file found at: {OUTPUT_PATH}")
    else:
        st.error(f"Model file not found at: {OUTPUT_PATH}")

    try:
        model = tf.keras.models.load_model(OUTPUT_PATH)
    except (IOError, OSError, ValueError) as e:
        return None, f"Error loading model: {str(e)}"

    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to a batch
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        return result_index, None
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

# Class labels
class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
              'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
              'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
              'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
              'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
              'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
              'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
              'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
              'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
              'Tomato___healthy']

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"

    if os.path.exists(image_path):
        st.image(image_path)
    else:
        st.warning("⚠️ Image file 'home_page.jpeg' not found in the app directory.")

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### Dataset Info
    This dataset contains about 87,000 RGB images of healthy and diseased crop leaves, categorized into 38 classes.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    # Download model if not already downloaded
    download_model()

    st.markdown("<h2>🌿 Plant Disease Recognition System</h2>", unsafe_allow_html=True)
    test_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if test_image:
        result_index, error = model_prediction(test_image)
        if error:
            st.error(f"Error: {error}")
        else:
            st.success(f"Prediction: {class_name[result_index]}")
