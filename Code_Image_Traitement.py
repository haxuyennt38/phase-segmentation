import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Load image in grayscale mode
img = cv2.imread('H02.png', cv2.IMREAD_GRAYSCALE)

# Check if image is loaded successfully
if img is None:
    raise FileNotFoundError("Image 'H02.png' not found. Please check the file path.")

# 2. Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(img, (5, 5), 0)

# 3. Apply Otsu's thresholding to automatically segment the 2 phases
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. Display comparison results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('2-Phase Segmentation (Solid / Void)')
plt.imshow(thresh, cmap='gray')
plt.axis('off')

plt.show()

# 5. Calculate phase area fractions
total_pixels = thresh.size
phase_solid = np.sum(thresh == 255)  # Solid phase (white)
phase_void = np.sum(thresh == 0)     # Void phase (black)

print(f'Solid Phase Fraction: {phase_solid / total_pixels * 100:.2f}%')
print(f'Void Phase Fraction: {phase_void / total_pixels * 100:.2f}%')