# 🧠 Brain Tumor Classification Using Deep Learning

This project uses deep learning techniques to classify **brain MRI images** into four categories:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

The model leverages **custom CNN** and **transfer learning (MobileNetV2)** for improved accuracy and efficiency, and a **Streamlit web app** for real-time predictions from MRI images.

---

## 📌 Project Features

- **Custom CNN** and **MobileNetV2 Transfer Learning**
- Data Augmentation for better generalization
- Class weighting to handle class imbalance
- EarlyStopping, ReduceLROnPlateau, and Checkpoint saving
- Evaluation using Accuracy, Precision, Recall, F1-score, and Confusion Matrix
- Streamlit App for user-friendly predictions

---

## 📊 Dataset

MRI images split into:
- `train/`
- `validation/`
- `test/`

Images are categorized into four folders representing each class.

> **Note:** Dataset not included in the repository due to size. Follow dataset_instructions.txt or use your own dataset.

---

## 🚀 Model Performance

Achieved:
- **~93% test accuracy**
- Strong performance on all tumor types, with focus on improving Meningioma detection.
