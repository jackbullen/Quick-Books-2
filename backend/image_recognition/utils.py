"""
Utilities for image segmentation
"""

import cv2
import math
import numpy as np

def resize_img(image, new_width=1200):
    """Resizes to width while maintaining the aspect ratio"""
    img = image.copy()
    img_ht, img_wd, _ = img.shape
    ratio = img_ht / img_wd
    new_height = math.ceil(new_width * ratio)
    img = cv2.resize(img, (new_width, new_height))
    return img

def polarlines_to_startendpts(lines, length):
    """
    Converts a line that is described by an angle and distance from origin
    to a line that is described by its start and end points.
    """
    points = []

    for line in lines:
        
        # Set the r and theta values for this line.
        r, theta = line[0]
        
        # Unit vector in direction of closest point on the line.
        a = np.cos(theta)
        b = np.sin(theta)
        
        # Perp unit vector to above unit vector (aka the unit direction vector of the line)
        c = b
        d = -a
        
        # Closest point to origin that is on the line
        x0 = a * r
        y0 = b * r
        
        # Compute start and end points of the line by 
        x1 = int(x0 - (length)*c)
        y1 = int(y0 - (length)*d)
        x2 = int(x0 + (length)*c)
        y2 = int(y0 + (length)*d)
        
        points.append(((x1, y1), (x2, y2)))
    
    # Add a vertical line at the end and start of the image
    # points.append(((wd, 0), (wd, ht)))
    # points.append(((0, 0), (0, ht)))

    return points

def dist(x, y):
    """Return max distance between two lines x coords and y coords"""
    return max(abs(x[0]-y[0]), abs(x[1]-y[1]))

def group_lines(lines, tolerance=100):
    """Group lines if they are within tolerance pixels in the use coordinate"""
    groups = []
    for line in lines:
        added = 0 
        for group in groups:
            # print(line[0], group[0][0], dist(line[0], group[0][0]))
            if dist(line[0], group[0][0]) < tolerance:
                # print("HIIHIFHIHD")
                group.append(line)
                added = 1
                break
        if added == 0:
            groups.append([line])
    return groups

def average_lines(grouped_lines):
    """Find the average line in each group of lines"""
    averaged_lines = []
    for group in grouped_lines:
        left_points = [line[0] for line in group]
        x_left = [pt[0] for pt in left_points]
        y_left = [pt[1] for pt in left_points]
        avg_x_left = int(sum(x_left) / len(x_left))
        avg_y_left = int(sum(y_left) / len(y_left))

        right_points = [line[1] for line in group]
        x_right = [pt[0] for pt in right_points]
        y_right = [pt[1] for pt in right_points]
        avg_x_right = int(sum(x_right) / len(x_right))
        avg_y_right = int(sum(y_right) / len(y_right))

        averaged_lines.append(((avg_x_left, avg_y_left), (avg_x_right, avg_y_right)))

    return averaged_lines

def smooth_lines(lines, tolerance=100):
    groups = group_lines(lines, tolerance=tolerance)
    avg_lines = average_lines(groups)
    return avg_lines

def rescale_lines(lines, original_img_shape, resized_img_shape):
    img_ht, img_wd, _ = original_img_shape
    resized_img_ht, resized_img_wd, _ = resized_img_shape
    ratio_ht = img_ht / resized_img_ht
    ratio_wd = img_wd / resized_img_wd

    resized_lines = []
        
    for line in lines:
        ((x1, y1), (x2, y2)) = line

        x1_original = int(x1 * ratio_wd)
        y1_original = int(y1 * ratio_ht)
        x2_original = int(x2 * ratio_wd)
        y2_original = int(y2 * ratio_ht)

        resized_lines.append(((x1_original, y1_original), (x2_original, y2_original)))

    return resized_lines

def crop_spines(image, points):
    
    img = np.copy(image)
    ht, wd, _ = np.shape(img)
    
    prev_x1 = 0
    prev_x2 = 0
    
    cropped_spines = []
    
    for point in points:
        ((x1, y1), (x2, y2)) = point
        
        crop_pts = np.array([[prev_x2, 0],
                             [prev_x1, ht],
                             [x2, y2],
                             [x1, y1]])
        
        rect = cv2.boundingRect(crop_pts)
        x, y, w, h = rect
        cropped_spine = image[y: y + h, x: x + w].copy()
        
        cropped_spines.append(cropped_spine)
        
        prev_x1 = x1
        prev_x2 = x2
        
    return cropped_spines
