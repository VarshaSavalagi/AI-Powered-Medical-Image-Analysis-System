import os
import cv2
import numpy as np

# Create folders
os.makedirs("data/normal", exist_ok=True)
os.makedirs("data/pneumonia", exist_ok=True)

# Generate NORMAL images (smooth patterns)
for i in range(20):
    img = np.random.normal(loc=120, scale=20, size=(128,128)).astype('uint8')
    cv2.imwrite(f"data/normal/normal_{i}.jpg", img)

# Generate PNEUMONIA images (noisy + bright spots)
for i in range(20):
    img = np.random.normal(loc=180, scale=50, size=(128,128)).astype('uint8')
    
    # add fake "infection spots"
    for _ in range(5):
        x, y = np.random.randint(0,128,2)
        cv2.circle(img, (x,y), 5, (255), -1)
    
    cv2.imwrite(f"data/pneumonia/pneumonia_{i}.jpg", img)

print("Dataset created")