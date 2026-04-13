import os
import cv2
import numpy as np

def load_images(data_dir, size=128):
    X, y = [], []

    for label, category in enumerate(["normal", "pneumonia"]):
        path = os.path.join(data_dir, category)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (size, size))

            X.append(image)
            y.append(label)

    X = np.array(X).reshape(-1, size, size, 1) / 255.0
    y = np.array(y)

    return X, y