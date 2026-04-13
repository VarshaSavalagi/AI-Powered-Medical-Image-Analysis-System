from src.preprocess import load_images
from src.train import train
from src.predict import predict_image
from src.visualize import show_image
import os

print("Loading dataset...")
X, y = load_images("data")

print("Training...")
train(X, y)

print("Predicting...")

# Automatically pick one image from normal folder
folder = "data/normal"
img_name = os.listdir(folder)[0]
img_path = os.path.join(folder, img_name)

result = predict_image(img_path)

print("Prediction:", result)

show_image(img_path, result)
