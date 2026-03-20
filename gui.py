import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import json
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.h5")

model = load_model()
classes = ["Glioma","Meningioma","Notumor","Pituitary"]

# -------------------------
# Grad-CAM (FINAL FIX)
# -------------------------
def get_gradcam(img_array, model):

    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer("Conv_1")

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_preds = grad_model(img_array)

        x = base_preds
        for layer in model.layers[1:]:
            x = layer(x)

        preds = x
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap   # ✅ FIXED HERE
# -------------------------
# Title
# -------------------------
st.markdown("<h2 style='text-align:center;'>🧠 Brain Tumor Detection using MRI</h2>", unsafe_allow_html=True)

col1, col2 = st.columns([1,1.3])

# -------------------------
# Upload
# -------------------------
with col1:
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

# -------------------------
# Prediction
# -------------------------
if uploaded_file:
    image_rgb = image.convert("RGB")
    img = image_rgb.resize((224,224))

    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    pred_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)

    with col2:
        st.success(f"Tumor: {pred_class}")
        st.write(f"Confidence: {confidence:.2%}")

        fig, ax = plt.subplots()
        ax.bar(classes, prediction)
        ax.set_ylim(0,1)
        st.pyplot(fig)

# -------------------------
# Grad-CAM Visualization
# -------------------------
if uploaded_file:
    st.markdown("---")
    st.subheader("Tumor Localization")

    heatmap = get_gradcam(img_array, model)

    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.resize(np.array(image_rgb), (224,224))
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    st.image(overlay, width=300)

# -------------------------
# History
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file:
    st.session_state.history.append({
        "Tumor": pred_class,
        "Confidence": round(confidence, 4)
    })

if st.session_state.history:
    st.markdown("---")
    st.subheader("Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))

# -------------------------
# Metrics
# -------------------------
st.markdown("---")
st.subheader("Model Evaluation")

if os.path.exists("metrics.json"):
    with open("metrics.json") as f:
        m = json.load(f)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{m['precision']*100:.2f}%")
    col3.metric("Recall", f"{m['recall']*100:.2f}%")
    col4.metric("F1 Score", f"{m['f1_score']*100:.2f}%")

else:
    st.warning("Run training first")