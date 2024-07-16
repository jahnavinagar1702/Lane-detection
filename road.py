import cv2
import numpy as np

# Read the image
img = cv2.imread('Road.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny
edges = cv2.Canny(blur, 50, 150)

# Create a mask
mask = np.zeros_like(edges)
height, width = mask.shape
polygon = np.array([[
    (0, height), (width, height), (width // 2, height // 2)
]])

# Fill the polygon
cv2.fillPoly(mask, [polygon], 255)

# Apply the mask to edges
masked_edges = cv2.bitwise_and(edges, mask)

# Detect lines using Hough transform
lines = cv2.HoughLinesP(masked_edges, rho=6, theta=np.pi/60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)

# Create an empty image to draw lines on
line_img = np.zeros_like(img)

# Draw lines on the empty image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 10)

# Combine the original image with the line image
result = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()