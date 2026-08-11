import cv2
import numpy as np


def calculate_color_distance(image, reference_colors):
    image_float = image.astype(np.float32)

    minimum_distance = np.full(
        image.shape[:2],
        np.inf,
        dtype=np.float32
    )

    for color in reference_colors:
        reference = np.array(color, dtype=np.float32)

        distance = np.linalg.norm(
            image_float - reference,
            axis=2
        )

        minimum_distance = np.minimum(
            minimum_distance,
            distance
        )

    return minimum_distance


def threshold_distance(distance_image, threshold):
    mask = distance_image > threshold

    return mask.astype(np.uint8) * 255


def find_objects(mask, minimum_area):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []

    for contour in contours:
        if cv2.contourArea(contour) >= minimum_area:
            objects.append(contour)

    return objects

image = cv2.imread("animals.png")

if image is None:
    raise FileNotFoundError("animals.jpg")

# OpenCV uses BGR
grass_colors = [
    (35, 80, 30),
    (45, 100, 35),
    (60, 125, 50)
]

distance_image = calculate_color_distance(
    image,
    grass_colors
)

thresholded_image = threshold_distance(
    distance_image,
    threshold=60
)

objects = find_objects(
    thresholded_image,
    minimum_area=200
)

output = image.copy()

for contour in objects:
    x, y, width, height = cv2.boundingRect(contour)

    cv2.rectangle(
        output,
        (x, y),
        (x + width, y + height),
        (0, 255, 0),
        2
    )

cv2.imshow("Thresholded image", thresholded_image)
cv2.imshow("Detected objects", output)

cv2.waitKey(0)
cv2.destroyAllWindows()