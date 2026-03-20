import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import classification_report

# ==============================
# Dataset Paths
# ==============================

train_dir = "dataset/Training"
test_dir = "dataset/Testing"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

# ==============================
# Data Preprocessing (FIXED)
# ==============================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=8,
    zoom_range=0.05,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False   # IMPORTANT FIX
)

print("Class Labels:", train_generator.class_indices)

# ==============================
# MobileNetV2 Base Model
# ==============================

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model (NO overfitting)
base_model.trainable = False

# ==============================
# Build Model (BALANCED)
# ==============================

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),   # FIXED (was 0.5)
    Dense(train_generator.num_classes, activation='softmax')
])

# ==============================
# Compile Model
# ==============================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# Callbacks (ANTI-OVERFITTING)
# ==============================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6
)

# ==============================
# Train Model
# ==============================

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, lr_reduce]
)

# ==============================
# Save Model
# ==============================

model.save("brain_tumor_model.h5")
print("Model saved!")

# ==============================
# Save History
# ==============================

history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}

with open("history.json", "w") as f:
    json.dump(history_dict, f)

print("history.json saved")

# ==============================
# Predictions
# ==============================

test_generator.reset()

y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# ==============================
# Metrics
# ==============================

report = classification_report(y_true, y_pred_classes, output_dict=True)

metrics = {
    "accuracy": float(report["accuracy"]),
    "precision": float(report["weighted avg"]["precision"]),
    "recall": float(report["weighted avg"]["recall"]),
    "f1_score": float(report["weighted avg"]["f1-score"])
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("metrics.json saved")

# ==============================
# Plot Results
# ==============================

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.legend(['Train','Validation'])
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.legend(['Train','Validation'])
plt.show()