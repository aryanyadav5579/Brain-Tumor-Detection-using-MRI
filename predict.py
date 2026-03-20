import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# Load trained model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Class labels
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Image path
img_path = "test.jpg"

# Check file
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

# Load image
img = image.load_img(img_path, target_size=(224,224))

# Preprocess
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

predicted_index = np.argmax(prediction, axis=1)[0]
predicted_class = classes[predicted_index]
confidence = np.max(prediction)

# Output
print("Prediction:", predicted_class)
print("Confidence:", confidence)

print("\nAll class probabilities:")
for i, prob in enumerate(prediction[0]):
    print(f"{classes[i]}: {prob:.4f}")

# Show image
plt.imshow(img)
plt.title(f"{predicted_class} ({confidence:.2f})")
plt.axis("off")
plt.show()