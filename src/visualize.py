import matplotlib.pyplot as plt
import cv2

def show_image(path, label):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    plt.imshow(img, cmap='gray')
    plt.title(label)
    plt.show()