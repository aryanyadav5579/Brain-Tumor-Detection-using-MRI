🧠 Brain Tumor Detection using MRI

Brain Tumor Detection using MRI is a deep learning project that classifies brain MRI scans into four categories using a Convolutional Neural Network (CNN):

Glioma Tumor

Meningioma Tumor

Pituitary Tumor

No Tumor

The project includes scripts for training, evaluation, prediction, and a simple GUI interface.

🚀 Features

Train a CNN model using TensorFlow / Keras

Export trained model as brain_tumor_model.h5

Evaluate model performance on a test dataset

Predict tumor class from a single MRI image

Simple GUI application to visualize predictions

Organized and reproducible project structure

📂 Project Structure
Brain-Tumor-Detection/
│
├── train.py                 # Model training script
├── evaluate.py              # Evaluate model performance
├── predict.py               # Predict tumor from single image
├── gui.py                   # Simple GUI for prediction
│
├── brain_tumor_model.h5     # Saved trained model
│
├── dataset/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── notumor/
│   │
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── pituitary/
│       └── notumor/
│
└── README.md
🛠 Requirements

Python 3.8+

TensorFlow / Keras

NumPy

Scikit-learn

OpenCV or Pillow

Matplotlib

Pandas

⚙️ Installation

Create and activate a virtual environment.

Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Install dependencies
pip install --upgrade pip
pip install tensorflow numpy scikit-learn opencv-python matplotlib pillow pandas

(Optional) Generate requirements file:

pip freeze > requirements.txt
📊 Dataset Setup

Place MRI images in the following directory structure:

dataset/
│
├── Training/
│   ├── glioma
│   ├── meningioma
│   ├── pituitary
│   └── notumor
│
└── Testing/
    ├── glioma
    ├── meningioma
    ├── pituitary
    └── notumor
🧠 Train the Model
python train.py

After training, the model will be saved as:

brain_tumor_model.h5
📈 Evaluate the Model
python evaluate.py

This script prints evaluation metrics such as:

Accuracy

Precision

Recall

Confusion Matrix

🔍 Predict from a Single Image
python predict.py --image path/to/image.jpg

Example:

python predict.py --image test.jpg
🖥 Run the GUI

Launch the simple graphical interface:

python gui.py

The GUI allows you to:

Upload an MRI image

Run prediction

View tumor classification

**Tips & Notes**
- If you want to commit the trained model to the repository, remove the `*.h5` line from `.gitignore`.
- For reproducible experiments, log hyperparameters and seed the RNG in `train.py`.
- For larger datasets, consider storing raw images outside the repo and mounting them during training.

**Contributing**
- Open an issue to discuss features or file a PR with small, focused changes.

**Author / Contact**
- Aryan Yadav — aryankyadav5579@gmail.com
