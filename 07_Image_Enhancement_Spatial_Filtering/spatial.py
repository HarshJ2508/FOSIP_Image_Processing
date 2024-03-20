import cv2
import numpy as np
from scipy import fftpack

def high_pass_filter_fft(image, cutoff_frequency):
  """
  Applies a high pass filter to an image using FFT.

  Args:
      image: The grayscale image as a 2D numpy array.
      cutoff_frequency: The frequency threshold (0 to 1).

  Returns:
      The filtered image.
  """
  # Get image dimensions
  rows, cols = image.shape

  # Convert to grayscale if needed
  if len(image.shape) == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Perform 2D FFT
  fft_img = fftpack.fft2(image)

  # Create a mask for filtering
  mask = np.zeros_like(fft_img)
  center_row, center_col = int(rows / 2), int(cols / 2)
  radius = int(cutoff_frequency * min(center_row, center_col))
  mask[center_row - radius:center_row + radius + 1,
       center_col - radius:center_col + radius + 1] = 1

  # Apply mask and perform inverse FFT
  filtered_fft = fft_img * mask
  filtered_image = fftpack.ifft2(filtered_fft).real

  return filtered_image

def high_pass_filter_kernel(image):
  """
  Applies a high pass filter to an image using Laplacian kernel.

  Args:
      image: The grayscale image as a 2D numpy array.

  Returns:
      The filtered image.
  """
  # Convert to grayscale if needed
  if len(image.shape) == 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Define Laplacian kernel
  laplacian_kernel = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])

  # Apply convolution with the kernel
  filtered_image = cv2.filter2D(image, -1, laplacian_kernel)

  return filtered_image

def high_boost_filter(image, A, cutoff_frequency):
  """
  Applies a high boost filter to an image.

  Args:
      image: The grayscale image as a 2D numpy array.
      A: The amplification factor.
      cutoff_frequency: The frequency threshold (0 to 1) for the high pass filter.

  Returns:
      The filtered image.
  """
  # Apply high pass filtering
  filtered_image = high_pass_filter_fft(image, cutoff_frequency)

  # Perform high boost filtering
  average_intensity = np.mean(image)
  high_boost_image = A * filtered_image + image + average_intensity

  return high_boost_image

# Load your image
image = cv2.imread("../Images/p2219651326-5.webp")

# Choose high pass filtering method (comment out the other one)
# filtered_image = high_pass_filter_fft(image, 0.2)  # Frequency domain filtering (adjust cutoff_frequency)
filtered_image = high_pass_filter_kernel(image)  # Spatial filtering with Laplacian kernel

# Apply high boost filter (adjust A and cutoff_frequency)
high_boost_image = high_boost_filter(filtered_image, 1.5, 0.1)

# Display original, filtered, and high boost filtered image
cv2.imshow("Original Image", image)
cv2.imshow("Filtered Image", filtered_image)
cv2.imshow("High Boost Filtered Image", high_boost_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ______________________________________________________________________________________________________________________

image = cv2.imread("../Images/p2219651326-5.webp")
kernel2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
filtered_image = cv2.filter2D(image, -1, kernel)
cv2.imwrite("newimage.jpg", filtered_image)
