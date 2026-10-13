# Write a MATLAB/Python program to identify horizontal, vertical lines from 
# an image.  

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread(r"E:\3.2 semester\Image processing\lab\image5.jpg", cv2.IMREAD_GRAYSCALE)

# Check if image is found
if img is None:
    print("Image not found!")
    exit()

# Convert black lines to white
_, binary = cv2.threshold(
    img, 127, 255, cv2.THRESH_BINARY_INV
)

# -------------------------------
# Detect Horizontal Lines
# -------------------------------

horizontal_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (50, 1)
)

horizontal_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    horizontal_kernel
)

# -------------------------------
# Detect Vertical Lines
# -------------------------------

vertical_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (1, 50)
)

vertical_lines = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    vertical_kernel
)

# -------------------------------
# Display Results
# -------------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(binary, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(horizontal_lines, cmap='gray')
plt.title('Horizontal Lines')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(vertical_lines, cmap='gray')
plt.title('Vertical Lines')
plt.axis('off')

plt.tight_layout()
plt.show()