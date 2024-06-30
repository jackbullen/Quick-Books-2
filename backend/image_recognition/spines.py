"""
Segment shelf image into separate book spines
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import *

images_path = "images/shelf_images"
image_name = "easy.jpg"

image = cv2.imread(os.path.join(images_path, image_name))

img = np.copy(image)

# img = resize_img(image)

ht, wd, channels = img.shape

# Convert to grayscale as color is not required
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur to remove noise when calling Canny
blur = cv2.GaussianBlur(gray, (21, 21), 3)

# Find candidate edges
edges = cv2.Canny(blur, 50, 70)

# Pronounce the desired edge directions
kernel = [[0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0],]

kernel = np.array(kernel, dtype=np.uint8)
erosion = cv2.erode(edges, kernel, iterations=1)

# Run the hough line algorithm
lines = cv2.HoughLines(erosion, 1, np.pi/180, 100)

# Convert the lines to start and end points
lines = polarlines_to_startendpts(lines, ht)

lines = sorted(lines, key=lambda x: x[0][0])[:4]

# Group the lines then find average in each group
lines = smooth_lines(lines, tolerance=50)

# Rescale to the original image size
lines = rescale_lines(lines, image.shape, img.shape)

# # Plot the lines on the image
for line in sorted(lines, key=lambda x: x[0][0]):
    ((x1, y1), (x2, y2)) = line
    # print(f"({x1}, {y1}) -> ({x2}, {y2})")
    image = cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.show()

# Save segmented spine images
segments = []
prev_line = None
ct = 0
for i, line in enumerate(sorted(lines, key=lambda x: x[0][0])):
    ((x1, y1), (x2, y2)) = line
    
    if prev_line is not None:
        ((prev_x1, prev_y1), (prev_x2, prev_y2)) = prev_line
        
        polygon_points = np.array([
            [prev_x1, prev_y1],
            [x1, y1],
            [x2, y2],
            [prev_x2, prev_y2]
        ])
        
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [polygon_points], (255, 255, 255))
        segment = cv2.bitwise_and(image, mask)
        cv2.imwrite(f"images/spine_images/spine{ct}_{image_name}", segment)
        ct += 1
        segments.append(segment)
    prev_line = line
