# Brain Tumor Detection using MRI

Brain Tumor Detection using MRI is a concise, production-ready pipeline for classifying brain MRI scans into the classes: `glioma`, `meningioma`, `pituitary`, and `notumor` using a convolutional neural network.

**Highlights**
- Trains and exports a Keras model (brain_tumor_model.h5).
- Scripts for training, evaluation, single-image prediction, and a small GUI.
- Opinionated, reproducible project layout for experiments and demos.

**Project Structure**
- `train.py` — training entrypoint. Adjust hyperparameters inside or via CLI.
- `evaluate.py` — runs evaluation on the test set and prints metrics.
- `predict.py` — single-image prediction script (CLI).
- `gui.py` — a minimal GUI for loading an image and viewing predictions.
- `brain_tumor_model.h5` — trained model (ignored by `.gitignore` by default).
- `dataset/` — expected layout:
  - `Training/` with class subfolders
  - `Testing/` with class subfolders
  - Download Dataset
  - https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data

You can download the dataset directly from Kaggle:
**Requirements (suggested)**
- Python 3.8+
- TensorFlow / Keras
- numpy, pandas
- scikit-learn
- opencv-python or Pillow
- matplotlib

Create a virtual environment and install the essentials:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install tensorflow numpy scikit-learn opencv-python matplotlib pillow
```

If you prefer a `requirements.txt`, generate one after installing packages:

```bash
pip freeze > requirements.txt
```

**Quick Start**

1) Prepare dataset

Place image folders under `dataset/Training/` and `dataset/Testing/` using the class labels: `glioma`, `meningioma`, `notumor`, `pituitary`.

2) Train

```bash
python train.py
```

Training may produce a `brain_tumor_model.h5` file in the project root. Note: large model files are ignored by default in `.gitignore`.

3) Evaluate

```bash
python evaluate.py
```

4) Predict (single image)

```bash
python predict.py --image path/to/image.jpg
```

5) Run GUI

```bash
python gui.py
```

**Tips & Notes**
- If you want to commit the trained model to the repository, remove the `*.h5` line from `.gitignore`.
- For reproducible experiments, log hyperparameters and seed the RNG in `train.py`.
- For larger datasets, consider storing raw images outside the repo and mounting them during training.

**Contributing**
- Open an issue to discuss features or file a PR with small, focused changes.
## 👥 Team Members & Individual Contributions

| Name | Roll No. | Contribution |
|------|---------|-------------|
| **Aryan Yadav** | 2305525 | Developed the CNN model architecture, implemented the training pipeline (train.py), performed dataset preprocessing, data organization, backend support for model training/testing, implemented evaluation and prediction modules (evaluate.py, predict.py), handled performance metrics and experimental analysis, and integrated the complete system. |
| **Sansthita Dey** | 2305564 | Assisted with data organization, project documentation, and methodology preparation. Contributed to result compilation, presentation development, and the formatting of the final research report.. |
| **Ishan Sinha** | 2305542 | Assisted with project review, testing, and documentation.. |
| **Shayank Gupta** | 2305569 | Developed the **frontend GUI (`gui.py`)** for uploading MRI images and displaying tumor predictions. Responsible for **project documentation and report preparation**, including methodology description, diagrams, and formatting of the final research report . |
| **Adyasha Pattanaik** | 2305590 | GUI development, documentation, methodology & report formatting 

                 


**Author / Contact**
- Aryan Yadav — aryankyadav5579@gmail.com
