# 🌿 Plant Disease Classification using Transfer Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Accuracy-98.48%25-brightgreen?style=for-the-badge)

A high-performance Deep Learning model designed to identify and classify plant leaf diseases. This project utilizes the **MobileNetV3Large** architecture and a strategic two-phase training approach to achieve state-of-the-art accuracy on the **PlantVillage** dataset.

##  1) Dataset Overview
The model is trained to recognize **15 distinct classes** of plant health states across multiple species:
* **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, and Healthy.
* **Potato:** Early Blight, Late Blight, and Healthy.
* **Pepper Bell:** Bacterial Spot and Healthy.

**Data Preparation:** - Dataset split: **80% Training / 20% Validation**.
- Augmentation: Applied rotation, shifts, shear, zoom, and flips using `ImageDataGenerator`.
- Resolution: All images resized to **224x224x3**.

##  2) Model Architecture
The solution is built on **Transfer Learning** using:
1.  **MobileNetV3Large:** Pre-trained on ImageNet as the feature extractor.
2.  **Global Average Pooling:** For spatial dimension reduction.
3.  **Fully Connected Layers:** - Dense (512 units, ReLU) + Dropout (0.3).
    - Dense (256 units, ReLU) + Dropout (0.3).
4.  **Output Layer:** Softmax activation for 15-class classification.

##  3) Training Strategy
The training was conducted in two precise phases to maximize convergence:

### Phase 1: Feature Extraction
* **Base Model:** Frozen.
* **Optimizer:** Adam ($1 \times 10^{-4}$).
* **Validation Accuracy:** **95.82%**.

### Phase 2: Fine-Tuning
* **Base Model:** Last **30 layers** unfrozen.
* **Optimizer:** Adam with a reduced learning rate ($1 \times 10^{-5}$) to prevent weight distortion.
* **Final Validation Accuracy:** **98.48%**.

##  4) Results & Evaluation
| Metric | Initial Training | After Fine-Tuning |
| :--- | :---: | :---: |
| **Accuracy** | 95.82% | **98.48%** |
| **Loss** | 0.1264 | **0.0514** |

###  Key Performance Indicators:
* **Confusion Matrix:** Shows high precision across all 15 classes, with minimal confusion between similar fungal infections.
* **Generalization:** The model successfully classified unseen images sourced from Google (Real-world test) with high confidence.

## 5) Setup & Requirements
To run this notebook locally, install the following dependencies:
```bash
pip install tensorflow numpy matplotlib pillow split-folders scikit-learn seaborn
