import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(image):
    # Compute min and max intensity values
    min_intensity = np.min(image)
    max_intensity = np.max(image)

    # Apply contrast stretching
    stretched_image = ((image - min_intensity) / (max_intensity - min_intensity)) * 255
    stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)

    return stretched_image

# Read the satellite or aerial image
input_image = cv2.imread('../Images/image-1.jpg')

# Convert the image to grayscale (if needed)
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

print(type (gray_image), gray_image.shape)

# Apply contrast stretching
result_image = contrast_stretching(gray_image)

# Plot histograms
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.title('Original Image Histogram')
plt.hist(gray_image.flatten(), bins=256, range=(0, 256), color='gray', alpha=1)

plt.subplot(2, 2, 2)
plt.title('Contrast Stretched Image Histogram')
plt.hist(result_image.flatten(), bins=256, range=(0, 256), color='gray', alpha=1)

plt.subplot(2, 2, 3)
plt.title('Original Image')
plt.imshow(input_image, cmap='gray')

plt.subplot(2, 2, 4)
plt.title('Contrast Stretched Image')
plt.imshow(result_image, cmap='gray')

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
