# Write a MATLAB/Python program to Character Segment of an image.  

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Read the image
# -----------------------------------
image = cv2.imread( r"E:\3.2 semester\Image processing\lab\image6.jpg", cv2.IMREAD_GRAYSCALE)

# Display original image
plt.figure(figsize=(10, 4))
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

# -----------------------------------
# 2. Convert image to binary
# -----------------------------------
_, binary = cv2.threshold(
    image, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Display binary image
plt.figure(figsize=(10, 4))
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")
plt.show()

# -----------------------------------
# 3. Find connected components
# -----------------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary,
    connectivity=8
)

# List to store character images
characters = []

# -----------------------------------
# 4. Extract individual characters
# -----------------------------------
for i in range(1, num_labels):

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]

    # Remove very small noise
    if area > 20:

        character = binary[y:y+h, x:x+w]

        characters.append((x, y, w, h, character))

# Sort characters from left to right
characters.sort(key=lambda c: c[0])

# -----------------------------------
# 5. Display segmented characters
# -----------------------------------

print("Number of segmented characters:", len(characters))

plt.figure(figsize=(12, 4))

for i, (x, y, w, h, character) in enumerate(characters):

    plt.subplot(1, len(characters), i + 1)
    plt.imshow(character, cmap="gray")
    plt.title(str(i + 1))
    plt.axis("off")

plt.tight_layout()
plt.show()
