# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained model (adjust filename if needed)
model = tf.keras.models.load_model('best_transfer_model.h5')

# Define class labels (adjust if your labels differ)
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Set Streamlit app title
st.title("🧠 Brain Tumor Detection from MRI")
st.write("Upload an MRI image to predict tumor type.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an MRI image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_container_width=True)

    # Preprocess image for model prediction
    img = image.resize((224, 224))               # Resize to model input size
    img_array = np.array(img) / 255.0             # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # Predict using the model
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display prediction result
    st.subheader("Prediction:")
    st.write(f"**{predicted_class.upper()}** ({confidence:.2f}% confidence)")
