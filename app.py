import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('cnn_model.h5')

# Function to process the uploaded image and make a prediction
def predict_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

    prediction = model.predict(img_array)
    return prediction

# Streamlit app code
st.title("Mammal Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    prediction = predict_image(uploaded_file)
    # Display the prediction
    st.write(f'Predicted Class: {np.argmax(prediction[0])}')
