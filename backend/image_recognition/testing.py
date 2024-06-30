import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "images/shelf_images/1test4.jpeg"
image = cv2.imread(image_path)
original_image = np.copy(image)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Canny edge detection
edges = cv2.Canny(blur, 50, 150, apertureSize=3)

# Apply morphological operations to close gaps in edges
kernel = np.ones((5, 5), np.uint8)
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Perform Hough transform to detect lines
lines = cv2.HoughLinesP(edges_closed, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Draw detected lines on the original image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the results
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Detected Book Spines')
plt.axis('off')

plt.show()
