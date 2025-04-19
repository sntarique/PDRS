import streamlit as st
import tensorflow as tf
import numpy as np
import os

# Tensorflow Model Prediction
def model_prediction(test_image):
    if test_image is None:
        return None, "No image uploaded."

    try:
        model_path = os.path.join(os.path.dirname(__file__), 'trained_model.h5')
        model = tf.keras.models.load_model(model_path)
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

# Initialize session state for app_mode
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Home"

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"], index=["Home", "About", "Disease Recognition"].index(st.session_state.app_mode))

# Update session state when sidebar selection changes
if app_mode != st.session_state.app_mode:
    st.session_state.app_mode = app_mode

# Home Page
if st.session_state.app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"

    if os.path.exists(image_path):
        st.image(image_path)
    else:
        st.warning("⚠️ Image file 'home_page.jpeg' not found in the app directory.")

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Upload an image of a plant, and our system will analyze it to detect any signs of diseases.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant.
    2. **Analysis:** Our system will process the image.
    3. **Results:** View the results and take informed action.

    ### Why Choose Us?
    - **Accuracy:** Advanced machine learning for detection.
    - **User-Friendly:** Simple and intuitive UI.
    - **Fast:** Get results in seconds!

    ### Start Now
    Click the button below to go to the **Disease Recognition** page.
    """)

    # Start Now Button
    if st.button("🚀 Start Now"):
        st.session_state.app_mode = "Disease Recognition"
        st.rerun()

# About Page
elif st.session_state.app_mode == "About":
    st.header("About")
    st.markdown("""
    #### Dataset Info
    This dataset contains about 87,000 RGB images of healthy and diseased crop leaves, categorized into 38 classes.

    - **Train:** 70,295 images  
    - **Validation:** 17,572 images  
    - **Test:** 33 images

    The dataset was augmented and structured for efficient model training and testing.
    """)

# Disease Recognition Page
elif st.session_state.app_mode == "Disease Recognition":
    # Custom CSS for title and subtitle
    st.markdown("""
        <style>
            .title {
                font-size: 40px;
                font-weight: bold;
                color: #2E8B57;
                margin-bottom: 0.5rem;
            }
            .subtitle {
                font-size: 20px;
                margin-bottom: 1.5rem;
            }
            .uploaded-image {
                width: 70%;
                margin: 0 auto;
                display: block;
                border-radius: 10px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">🌿 Plant Disease Recognition System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a leaf image to detect disease</div>', unsafe_allow_html=True)

    test_image = st.file_uploader("🖼️ Upload an Image:", type=["jpg", "jpeg", "png"])

    if test_image:
        import base64
        from PIL import Image
        import io

        image = Image.open(test_image)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        st.markdown(f'<img src="data:image/png;base64,{img_str}" class="uploaded-image"/>', unsafe_allow_html=True)

        if st.button("🔍 Predict Disease"):
            model_path = os.path.join(os.path.dirname(__file__), 'trained_model.h5')
            if not os.path.exists(model_path):
                st.error("❌ Model file 'trained_model.h5' not found. Please make sure it's in the app directory.")
            else:
                with st.spinner("🧠 Analyzing the image..."):
                    result_index, error = model_prediction(test_image)

                    if error:
                        st.error(f"🚫 Error: {error}")
                    else:
                        st.success(f"✅ The model predicts: **{class_name[result_index]}**")
