import cv2
import numpy as np
import matplotlib.pyplot as plt

def mark_text_regions(image_path, debug=False):
    image = cv2.imread(image_path)
    original_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    _, binary_image = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    closed_image = cv2.convertScaleAbs(closed_image)
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #largest_contours = contours[:2]
    cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h
    padding = 8
    x_min -= padding
    y_min -= padding
    x_max += padding
    y_max += padding

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)

    cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    text_region = image[y_min:y_max, x_min:x_max]

    if debug:
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    return text_region
