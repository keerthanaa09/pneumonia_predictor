import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("pneumonia_model.keras")

st.title("Pneumonia Prediction System")

uploaded_file = st.file_uploader("Upload X-ray Image")

if uploaded_file is not None:

    img = image.load_img(uploaded_file, target_size=(64,64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("Pneumonia Detected")
    else:
        st.success("Normal")