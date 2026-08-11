import cv2
import numpy as np

image_path = "grass2.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(image_path)

# Calculate average of every pixel
average_color = np.mean(image, axis=(0, 1))

# OpenCV uses BGR
b = int(average_color[0])
g = int(average_color[1])
r = int(average_color[2])

print("Average grass color:")
print(f"BGR: ({b}, {g}, {r})")
print(f"RGB: ({r}, {g}, {b})")