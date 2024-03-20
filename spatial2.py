import cv2
import numpy as np

image = cv2.imread("../Images/p2219651326-5.webp")
kernel2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
filtered_image = cv2.filter2D(image, -1, kernel)
cv2.imwrite("newimage.jpg", filtered_image)
