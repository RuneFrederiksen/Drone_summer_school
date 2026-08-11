#!/usr/bin/env python3
import cv2
import numpy as np
import os


class ObjectDetector:

    def __init__(
        self,
        reference_colors,
        threshold=60,
        minimum_area=200,
        output_folder="output"
    ):
        self.reference_colors = reference_colors
        self.threshold = threshold
        self.minimum_area = minimum_area
        self.output_folder = output_folder

        os.makedirs(self.output_folder, exist_ok=True)


    def calculate_color_distance(self, image):
        image_float = image.astype(np.float32)

        minimum_distance = np.full(
            image.shape[:2],
            np.inf,
            dtype=np.float32
        )

        for color in self.reference_colors:
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


    def threshold_distance(self, distance_image):
        mask = distance_image > self.threshold

        return mask.astype(np.uint8) * 255


    def find_objects(self, mask):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        objects = []

        for contour in contours:
            if cv2.contourArea(contour) >= self.minimum_area:
                objects.append(contour)

        return objects


    def process_image(self, image, filename="image"):
        distance_image = self.calculate_color_distance(image)

        thresholded_image = self.threshold_distance(
            distance_image
        )

        objects = self.find_objects(
            thresholded_image
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

        cv2.imwrite(
            os.path.join(
                self.output_folder,
                f"{filename}_thresholded.jpg"
            ),
            thresholded_image
        )

        cv2.imwrite(
            os.path.join(
                self.output_folder,
                f"{filename}_detected.jpg"
            ),
            output
        )

        return objects, thresholded_image, output
    
grass_colors = [
    (35, 80, 30),
    (45, 100, 35),
    (60, 125, 50)
]

detector = ObjectDetector(
    reference_colors=grass_colors,
    threshold=60,
    minimum_area=200
)

image_path = "/home/pi/images/capture_4/img_7.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(image_path)

detector.process_image(
    image,
    "img_7"
)