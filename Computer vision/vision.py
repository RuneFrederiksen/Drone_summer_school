#!/usr/bin/env python3
import cv2
import numpy as np
import os
import math


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

    def get_box_centers(self, objects):
        centers = []

        for contour in objects:
            x, y, width, height = cv2.boundingRect(contour)
            center_x = x + width / 2.0
            center_y = y + height / 2.0
            centers.append((center_x, center_y))

        return centers

    def pixel_to_gps(
        self,
        center_x,
        center_y,
        image_width,
        image_height,
        drone_lat,
        drone_lon,
        drone_heading_deg,
        altitude_m,
        pixel_to_m_at_1m
    ):
        # Pixel offset from the center of the image.
        dx_pixels = center_x - image_width / 2.0
        dy_pixels = center_y - image_height / 2.0

        # Scale increases linearly with altitude for a downward-facing camera.
        meters_per_pixel = pixel_to_m_at_1m * altitude_m

        # Camera/image coordinates:
        # +x = right side of image
        # +y = bottom of image
        right_m = dx_pixels * meters_per_pixel
        forward_m = -dy_pixels * meters_per_pixel

        # Rotate image coordinates according to drone heading.
        # heading = 0 deg means drone nose points north.
        heading = math.radians(drone_heading_deg)

        north_m = forward_m * math.cos(heading) - right_m * math.sin(heading)
        east_m = forward_m * math.sin(heading) + right_m * math.cos(heading)

        # Convert local north/east offsets to latitude/longitude offsets.
        earth_radius_m = 6378137.0
        lat_rad = math.radians(drone_lat)

        delta_lat = north_m / earth_radius_m
        delta_lon = east_m / (earth_radius_m * math.cos(lat_rad))

        object_lat = drone_lat + math.degrees(delta_lat)
        object_lon = drone_lon + math.degrees(delta_lon)

        return object_lat, object_lon, north_m, east_m

    def process_image(
        self,
        image,
        filename="image",
        drone_position=None,
        altitude_m=None,
        pixel_to_m_at_1m=None
    ):
        distance_image = self.calculate_color_distance(image)
        thresholded_image = self.threshold_distance(distance_image)
        objects = self.find_objects(thresholded_image)

        output = image.copy()
        image_height, image_width = image.shape[:2]
        detections = []

        for contour in objects:
            x, y, width, height = cv2.boundingRect(contour)
            center_x = x + width / 2.0
            center_y = y + height / 2.0

            detection = {
                "center_pixel": (center_x, center_y),
                "bounding_box": (x, y, width, height),
            }

            # GPS calculation is only done if all required values are supplied.
            if (
                drone_position is not None
                and altitude_m is not None
                and pixel_to_m_at_1m is not None
            ):
                drone_lat, drone_lon, _, drone_heading_deg = drone_position

                gps_lat, gps_lon, north_m, east_m = self.pixel_to_gps(
                    center_x,
                    center_y,
                    image_width,
                    image_height,
                    drone_lat,
                    drone_lon,
                    drone_heading_deg,
                    altitude_m,
                    pixel_to_m_at_1m
                )

                detection["gps"] = (gps_lat, gps_lon)
                detection["offset_m"] = (north_m, east_m)

            detections.append(detection)

            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            cv2.circle(
                output,
                (int(center_x), int(center_y)),
                5,
                (0, 0, 255),
                -1
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

        return detections, thresholded_image, output

grass_colors = [
    (60, 104, 73),
    (180, 177, 172),
    (99, 135, 89),
]

detector = ObjectDetector(
    reference_colors=grass_colors,
    threshold=60,
    minimum_area=2000
)

# Image to test
image_path = "/home/pi/images/capture_17/img_5.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(image_path)


# Fake drone data for testing
drone_lat = 55.3682
drone_lon = 10.4281
drone_altitude = 20.0
drone_heading = 90.0

# (latitude, longitude, unused altitude, heading)
drone_position = (
    drone_lat,
    drone_lon,
    drone_altitude,
    drone_heading
)

# Example camera scale:
# meters per pixel when camera is 1 meter above ground
pixel_to_m_at_1m = 0.001


# Process image
detections, thresholded_image, output = detector.process_image(
    image,
    "img_4",
    drone_position=drone_position,
    altitude_m=drone_altitude,
    pixel_to_m_at_1m=pixel_to_m_at_1m
)


# Print detected objects
for i, detection in enumerate(detections):
    print(f"Object {i + 1}")
    print(f"Pixel center: {detection['center_pixel']}")

    if "gps" in detection:
        print(f"GPS: {detection['gps']}")
        print(f"Offset: {detection['offset_m']} meters")

    print()