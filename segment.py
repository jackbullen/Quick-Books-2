from utils import *
import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("test.jpg")

img = np.copy(image)

img = resize_img(image)

ht, wd, channels = img.shape

# Convert to grayscale as color is not required
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur to remove noise when calling Canny
blur = cv2.GaussianBlur(gray, (21, 21), 3)

# Find candidate edges
edges = cv2.Canny(blur, 50, 70)

# Pronounce the desired edge directions
kernel = [[0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]]
kernel = np.array(kernel, dtype=np.uint8)
erosion = cv2.erode(edges, kernel, iterations=2)

# Run the hough line algorithm
lines = cv2.HoughLines(erosion, 1, np.pi/180, 100)

# Convert the lines to start and end points
lines = polarlines_to_startendpts(lines, ht, wd)

# Group the lines then find average in each group
lines = smooth_lines(lines)

# Rescale to the original image size
lines = rescale_lines(lines, image.shape, img.shape)

# Plot the lines on the image
# for line in lines:
#     ((x1, y1), (x2, y2)) = line
#     image = cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 10)

# Segment the image
segments = []
prev_y = 0
for line in sorted(lines, key=lambda x: x[0][1]):
    ((_, y), (_, _)) = line
    segments.append(image[prev_y:y, :, :])
    prev_y = y
segments.append(image[prev_y:, :, :])

plt.figure(figsize=(10, 10))
plt.imshow(segments[-1])
plt.show()