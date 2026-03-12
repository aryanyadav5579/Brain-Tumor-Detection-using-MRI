import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Class labels
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Image path (change this)
img_path = "test.jpg"

# Load image
img = image.load_img(img_path, target_size=(224,224))

# Convert to array
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Expand dimension
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

predicted_class = classes[np.argmax(prediction)]

print("Prediction:", predicted_class)

# Show image
plt.imshow(img)
plt.title(f"Prediction: {predicted_class}")
plt.axis("off")
plt.show()