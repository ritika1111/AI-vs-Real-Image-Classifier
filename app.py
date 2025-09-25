import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model.keras")

# Define image size used in training
IMAGE_SIZE = (224, 224)

# Title
st.title("AI vs Real Image Classifier")
st.markdown("Upload an image and the model will predict whether it's AI-generated or Real.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction function
def predict(image: Image.Image):
    # Resize and preprocess
    image = image.resize(IMAGE_SIZE)
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Prediction
    prediction = model.predict(img_array)[0][0]
    return prediction

# Display and predict
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        score = predict(image)
        if score > 0.5:
            st.success(f"🧠 AI-Generated Image (Confidence: {score:.2f})")
        else:
            st.success(f"🧍 Real Image (Confidence: {1 - score:.2f})")
