import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

def predict_image(path):
    model = load_model("models/model.h5")

    # Debug: check file exists
    if not os.path.exists(path):
        print("❌ File does not exist:", path)
        return None

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Debug: check image loaded
    if img is None:
        print("❌ Failed to load image:", path)
        return None

    img = cv2.resize(img, (128,128))
    img = img.reshape(1,128,128,1) / 255.0

    pred = model.predict(img)[0][0]
    return "Pneumonia" if pred > 0.5 else "Normal"
