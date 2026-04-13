# 🧠 AI-Powered Medical Image Analysis System

---

## 📌 Overview

This project is an **AI-powered medical image classification system** that analyzes chest X-ray images and predicts whether they are **Normal or Pneumonia (Diseased)** using a Convolutional Neural Network (CNN).

The system processes grayscale medical images, extracts features automatically using deep learning, and performs binary classification to assist in early-stage analysis.

---

## ❗ Problem Statement

Analyzing medical images manually is:

* Time-consuming
* Requires expert knowledge
* Prone to human error

This project aims to:

* Automate medical image analysis
* Classify images as **Normal or Diseased (Pneumonia)**
* Provide a fast and efficient screening tool

---

## 🏥 Industry Relevance

AI-based medical image analysis is widely used in:

* Radiology (X-rays, CT scans)
* Early disease detection
* Healthcare automation systems

This project demonstrates how deep learning can assist doctors by:

* Reducing workload
* Increasing efficiency
* Supporting decision-making

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:**

  * TensorFlow / Keras (Deep Learning)
  * OpenCV (Image Processing)
  * NumPy
  * Scikit-learn
  * Matplotlib

---

## 📊 Dataset

* Dataset contains **Chest X-ray images**

* Two categories:

  * `normal/` → Healthy images
  * `pneumonia/` → Diseased images

* Images are:

  * Converted to grayscale
  * Resized to **128×128 pixels**

---

## 🧠 Architecture

### 🔷 Design Architecture Explanation

#### 1. Input (Medical Images)

* Input images are chest X-rays
* Loaded from dataset folders (`normal`, `pneumonia`)

---

#### 2. Preprocessing

* Convert images to grayscale
* Resize to 128×128
* Normalize pixel values (0–1)
* Reshape to CNN format `(128,128,1)`

---

#### 3. Feature Extraction

* Done automatically using CNN layers:

  * Convolution layers detect patterns
  * Pooling layers reduce dimensions
* Extracts features like edges, textures, patterns

---

#### 4. Model (Classification)

* CNN architecture:

  * Conv2D (32 filters) + MaxPooling
  * Conv2D (64 filters) + MaxPooling
  * Flatten layer
  * Dense layer (64 neurons)
  * Output layer (Sigmoid)

* Loss: Binary Crossentropy

* Optimizer: Adam

---

#### 5. Prediction

* Model outputs probability:

  * > 0.5 → Pneumonia
  * ≤ 0.5 → Normal

---

#### 6. Output Visualization

* Displays image using Matplotlib
* Shows predicted label (Normal / Pneumonia)

---

### 📦 Text-Based Block Diagram

```
[ Medical Image (X-ray) ]
            ↓
[ Preprocessing (Resize + Normalize) ]
            ↓
[ CNN Feature Extraction ]
            ↓
[ Fully Connected Layers ]
            ↓
[ Prediction (Normal / Pneumonia) ]
            ↓
[ Visualization (Image + Label) ]
```

---

### 🧩 Module Explanation

* **src/preprocess.py**

  * Loads dataset
  * Converts images to required format

* **src/model.py**

  * Defines CNN architecture

* **src/train.py**

  * Splits data
  * Trains model
  * Evaluates accuracy
  * Saves model

* **src/predict.py**

  * Loads trained model
  * Predicts class for new image

* **src/visualize.py**

  * Displays image with predicted label

* **main.py**

  * Integrates full pipeline

---

### 🔄 Data Flow

1. Load images from dataset
2. Preprocess (resize, normalize)
3. Pass data to CNN model
4. Train model on labeled data
5. Save trained model
6. Input new image
7. Predict Normal/Pneumonia
8. Display result

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/AI-Powered-Medical-Image-Analysis-System.git
cd AI-Powered-Medical-Image-Analysis-System
pip install -r requirements.txt
```

---

## ▶️ How to run

### Train the model:

```bash
python main.py
```

### Predict new image:

```bash
python -c "from src.predict import predict_image; print(predict_image('path_to_image'))"
```

---

## 📈 Results

* Model successfully classifies:

  * Normal images
  * Pneumonia images

* Achieves reasonable accuracy with small dataset

* Demonstrates effectiveness of CNN for image classification

---

## 🎯 Learning Outcomes

* Understanding of **CNN (Convolutional Neural Networks)**
* Medical image preprocessing techniques
* Binary image classification
* Model training and evaluation
* Working with real-world datasets
* Building end-to-end ML pipeline

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and should not be used for real medical diagnosis.

---

## 🚀 Future Improvements

* Use larger dataset
* Improve accuracy with more epochs
* Implement advanced CNN architectures
* Build web app (Streamlit/Flask)
* Add multi-disease classification

---

---
