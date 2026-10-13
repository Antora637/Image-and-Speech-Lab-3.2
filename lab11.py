# Perform image enhancement, smoothing and sharpening, in frequency 
# domain using different filters and compare the performances

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread(r"E:\3.2 semester\Image processing\lab\cameraman.jpg", cv2.IMREAD_GRAYSCALE)

# Check image
if img is None:
    print("Image not found!")
    exit()

# Convert image to float
img = np.float32(img)

# -------------------------------------------------
# 1. FFT
# -------------------------------------------------
f = np.fft.fft2(img)
f_shift = np.fft.fftshift(f)

rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# -------------------------------------------------
# 2. Distance Matrix
# -------------------------------------------------
y, x = np.ogrid[:rows, :cols]

distance = np.sqrt(
    (x - ccol) ** 2 +
    (y - crow) ** 2
)

# Cutoff frequency
D0 = 30

# -------------------------------------------------
# 3. Ideal Low Pass Filter - Smoothing
# -------------------------------------------------
low_pass = np.zeros((rows, cols))

low_pass[distance <= D0] = 1

# Apply Low Pass Filter
f_low = f_shift * low_pass

# Inverse FFT
smooth_img = np.fft.ifft2(
    np.fft.ifftshift(f_low)
)

smooth_img = np.abs(smooth_img)

# -------------------------------------------------
# 4. Ideal High Pass Filter - Edge Detection
# -------------------------------------------------
high_pass = 1 - low_pass

# Apply High Pass Filter
f_high = f_shift * high_pass

# Inverse FFT
edge_img = np.fft.ifft2(
    np.fft.ifftshift(f_high)
)

edge_img = np.abs(edge_img)

# Normalize edge image for display
edge_display = cv2.normalize(
    edge_img, None, 0, 255, cv2.NORM_MINMAX
)

# -------------------------------------------------
# 5. High Boost Filtering - Sharpening
# -------------------------------------------------

# High frequency component in frequency domain
high_freq = f_shift * high_pass

# High boost factor
k = 1.5

# Add high frequency information to original spectrum
f_sharpened = f_shift + k * high_freq

# Inverse FFT
sharpened_img = np.fft.ifft2(
    np.fft.ifftshift(f_sharpened)
)

sharpened_img = np.abs(sharpened_img)

# Clip values
sharpened_img = np.clip(
    sharpened_img, 0, 255
)

# -------------------------------------------------
# 6. Frequency Spectrum
# -------------------------------------------------
magnitude = np.log(
    1 + np.abs(f_shift)
)

# -------------------------------------------------
# 7. Display Results
# -------------------------------------------------
plt.figure(figsize=(15, 8))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Low Pass - Smoothing
plt.subplot(2, 3, 2)
plt.imshow(smooth_img, cmap='gray')
plt.title('Low Pass - Smoothing')
plt.axis('off')

# High Pass - Edge Detection
plt.subplot(2, 3, 3)
plt.imshow(edge_display, cmap='gray')
plt.title('High Pass - Edge Detection')
plt.axis('off')

# High Boost - Sharpening
plt.subplot(2, 3, 4)
plt.imshow(sharpened_img, cmap='gray')
plt.title('High Boost - Sharpening')
plt.axis('off')

# Frequency Spectrum
plt.subplot(2, 3, 5)
plt.imshow(magnitude, cmap='gray')
plt.title('Frequency Spectrum')
plt.axis('off')

# Low Pass Filter Mask
plt.subplot(2, 3, 6)
plt.imshow(low_pass, cmap='gray')
plt.title('Low Pass Filter Mask')
plt.axis('off')

plt.tight_layout()
plt.show()