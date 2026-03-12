import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from reportlab.pdfgen import canvas
import json
import os

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.h5")

model = load_model()

classes = ["Glioma","Meningioma","Notumor","Pituitary"]

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Model Information")

for c in classes:
    st.sidebar.write("•",c)

# -------------------------
# Title
# -------------------------
st.title("🧠 Brain Tumor Detection using MRI")
st.write("AI powered MRI tumor classification system")

# -------------------------
# Layout
# -------------------------
col1, col2, col3 = st.columns([1.2,1.6,1])

uploaded_file = None

# -------------------------
# Upload MRI
# -------------------------
with col1:

    st.subheader("MRI Image Preview")

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["jpg","png","jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(image, use_container_width=True)

# -------------------------
# Prediction
# -------------------------
with col2:

    st.subheader("Prediction Result")

    if uploaded_file:

        # Convert image to RGB
        image_rgb = image.convert("RGB")

        # Resize
        img = image_rgb.resize((224,224))

        # Convert to numpy
        img_array = np.array(img)

        # Normalize
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Debug shape
        st.write("Input Shape:", img_array.shape)

        # Prediction
        prediction = model.predict(img_array)[0]

        pred_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.success(f"Tumor Type: {pred_class}")
        st.write(f"Confidence: {confidence:.2f}")

        # -------------------------
        # Probability Chart
        # -------------------------
        st.subheader("Class Probabilities")

        df = pd.DataFrame({
            "Tumor Type": classes,
            "Probability": prediction
        })

        fig, ax = plt.subplots()

        ax.barh(df["Tumor Type"], df["Probability"])

        ax.set_xlim(0,1)

        st.pyplot(fig)

# -------------------------
# Model Info
# -------------------------
with col3:

    st.subheader("Model Information")

    for c in classes:
        st.write(c)

# -------------------------
# Heatmap
# -------------------------
if uploaded_file:

    st.markdown("---")
    st.subheader("Tumor Heatmap")

    img = cv2.resize(np.array(image.convert("RGB")), (224,224))

    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img,0.6,heatmap,0.4,0)

    st.image(overlay, use_container_width=True)

# -------------------------
# Accuracy graph
# -------------------------
if os.path.exists("history.json"):

    st.markdown("---")
    st.subheader("Training Accuracy")

    with open("history.json") as f:
        history = json.load(f)

    fig, ax = plt.subplots()

    ax.plot(history["accuracy"], label="Train Accuracy")
    ax.plot(history["val_accuracy"], label="Validation Accuracy")

    ax.legend()

    st.pyplot(fig)

# -------------------------
# PDF report
# -------------------------
if uploaded_file:

    st.markdown("---")

    if st.button("Generate Medical Report"):

        report = canvas.Canvas("report.pdf")

        report.drawString(100,750,"Brain Tumor Detection Report")

        report.drawString(100,700,f"Prediction: {pred_class}")
        report.drawString(100,680,f"Confidence: {confidence:.2f}")

        report.save()

        with open("report.pdf","rb") as f:

            st.download_button(
                "Download Report",
                f,
                file_name="brain_tumor_report.pdf"
            )