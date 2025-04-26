import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load model once at the start
model = tf.keras.models.load_model('trained_model.h5')

# TensorFlow model prediction
def model_predict(image):
    # image is already PIL Image object
    image = image.resize((128, 128))  # Resize to match model input
    input_arr = np.array(image)  # Convert to array
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    input_arr = input_arr / 255.0  # Normalize
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)  # Get index of highest probability
    return result_index

# Streamlit Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Prediction"])

# Prediction page
if app_mode == "Prediction":
    st.title("Crop Disease Prediction")

    uploaded_file = st.file_uploader("Upload an image of the crop leaf", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Read the uploaded image as PIL Image
        test_image = Image.open(uploaded_file)
        st.image(test_image, use_container_width=True)

        if st.button("Predict"):
            result_index = model_predict(test_image)  # Pass PIL Image directly
            class_name = ['Apple___Apple_scab',
                         'Apple___Black_rot',
                         'Apple___Cedar_apple_rust',
                         'Apple___healthy',
                         'Blueberry___healthy',
                         'Cherry_(including_sour)___Powdery_mildew',
                         'Cherry_(including_sour)___healthy',
                         'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                         'Corn_(maize)___Common_rust_',
                         'Corn_(maize)___Northern_Leaf_Blight',
                         'Corn_(maize)___healthy',
                         'Grape___Black_rot',
                         'Grape___Esca_(Black_Measles)',
                         'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                         'Grape___healthy',
                         'Orange___Haunglongbing_(Citrus_greening)',
                         'Peach___Bacterial_spot',
                         'Peach___healthy',
                         'Pepper,_bell___Bacterial_spot',
                         'Pepper,_bell___healthy',
                         'Potato___Early_blight',
                         'Potato___Late_blight',
                         'Potato___healthy',
                         'Raspberry___healthy',
                         'Soybean___healthy',
                         'Squash___Powdery_mildew',
                         'Strawberry___Leaf_scorch',
                         'Strawberry___healthy',
                         'Tomato___Bacterial_spot',
                         'Tomato___Early_blight',
                         'Tomato___Late_blight',
                         'Tomato___Leaf_Mold',
                         'Tomato___Septoria_leaf_spot',
                         'Tomato___Spider_mites Two-spotted_spider_mite',
                         'Tomato___Target_Spot',
                         'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                         'Tomato___Tomato_mosaic_virus',
                         'Tomato___healthy']
                         
            st.success(f"Model predicts: {class_name[result_index]}")
            st.balloons()
    else:
        st.warning("Please upload an image to proceed!")
