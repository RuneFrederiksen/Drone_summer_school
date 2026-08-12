import cv2

image = cv2.imread("img_3.jpg")

height, width = image.shape[:2]

print("Width:", width)
print("Height:", height)
print("Resolution:", width, "x", height)