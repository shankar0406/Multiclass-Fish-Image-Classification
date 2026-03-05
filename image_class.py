import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = "models/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

# ----------------------------
# Get Class Names Automatically
# ----------------------------
DATA_DIR = "data/train"

class_names = sorted(os.listdir(DATA_DIR))

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Multiclass-Fish-Image-Classification")

uploaded_file = st.file_uploader(
    "Upload Fish Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")
    # Preprocess
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction))

    # Safety check
    if class_idx < len(class_names):
        predicted_class = class_names[class_idx]
    else:
        predicted_class = "Unknown Class"

    # Display Results
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence*100:.2f}%")