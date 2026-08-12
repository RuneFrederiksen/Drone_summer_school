import cv2
import numpy as np
import os
import math
import csv
import re


# ============================================================
# SETTINGS
# ============================================================

CAPTURE_FOLDER = "capture_17"

GPS_LOG_NAME = "gps_locations.log"

OUTPUT_FOLDER_NAME = "vision_output"
OUTPUT_CSV_NAME = "object_locations.csv"


# Add MANY representative grass samples here.
# Format is BGR because OpenCV uses BGR.
GRASS_COLORS = [
    (60, 104, 73),
    (99, 135, 89),
    (80, 120, 70),
    (70, 110, 60),
    (110, 145, 95),
    (55, 90, 50),
    (130, 155, 110),
]


# Larger = more pixels accepted as grass.
GRASS_DISTANCE_THRESHOLD = 150.0


# Ignore small detected regions.
MINIMUM_OBJECT_AREA = 2000


# ============================================================
# PIXEL SIZE
# ============================================================

# Length of ONE pixel on the ground in meters.
#
# Example:
# 0.003 = 3 mm per pixel
# 0.005 = 5 mm per pixel
# 0.01  = 1 cm per pixel
#
# Change ONLY this value after calibrating your camera.
PIXEL_LENGTH_M = 0.00584


# Morphological cleanup.
MORPH_KERNEL_SIZE = 7


# ============================================================
# OBJECT DETECTOR
# ============================================================

class ObjectDetector:

    def __init__(
        self,
        grass_colors,
        grass_distance_threshold=12.0,
        minimum_area=2000
    ):

        self.grass_colors = grass_colors
        self.grass_distance_threshold = grass_distance_threshold
        self.minimum_area = minimum_area

        self.train_grass_model()


    def train_grass_model(self):

        colors = np.array(
            self.grass_colors,
            dtype=np.uint8
        ).reshape(-1, 1, 3)


        hsv = cv2.cvtColor(
            colors,
            cv2.COLOR_BGR2HSV
        ).reshape(-1, 3).astype(np.float32)


        self.grass_mean = hsv.mean(axis=0)

        self.grass_std = hsv.std(axis=0)


        # Prevent divide-by-zero if training values
        # are too similar.
        self.grass_std = np.maximum(
            self.grass_std,
            [3.0, 10.0, 10.0]
        )


        print("Grass model:")
        print("Mean HSV:", self.grass_mean)
        print("STD HSV:", self.grass_std)
        print(
            "Distance threshold:",
            self.grass_distance_threshold
        )
        print()


    def create_object_mask(self, image):

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        ).astype(np.float32)


        # Difference from learned grass color.
        diff = hsv - self.grass_mean


        # Hue wraps around in OpenCV:
        # 0 and 179 are actually close colors.
        hue_diff = np.abs(
            diff[:, :, 0]
        )

        hue_diff = np.minimum(
            hue_diff,
            180 - hue_diff
        )


        # Normalized distance from grass model.
        distance = (

            (hue_diff / self.grass_std[0]) ** 2

            +

            (
                diff[:, :, 1] /
                self.grass_std[1]
            ) ** 2

            +

            (
                diff[:, :, 2] /
                self.grass_std[2]
            ) ** 2
        )


        # Small distance = grass.
        grass_mask = (
            distance <
            self.grass_distance_threshold
        )


        # White = object
        # Black = grass
        object_mask = (
            (~grass_mask).astype(np.uint8)
            * 255
        )


        # Remove noise.
        if MORPH_KERNEL_SIZE > 0:

            kernel = np.ones(
                (
                    MORPH_KERNEL_SIZE,
                    MORPH_KERNEL_SIZE
                ),
                np.uint8
            )


            object_mask = cv2.morphologyEx(
                object_mask,
                cv2.MORPH_OPEN,
                kernel
            )


            object_mask = cv2.morphologyEx(
                object_mask,
                cv2.MORPH_CLOSE,
                kernel
            )


        return object_mask


    def find_objects(self, mask):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        objects = []


        for contour in contours:

            area = cv2.contourArea(
                contour
            )


            if area >= self.minimum_area:

                objects.append(
                    contour
                )


        return objects


# ============================================================
# GPS CONVERSION
# ============================================================

def pixel_to_gps(
    center_x,
    center_y,
    image_width,
    image_height,
    drone_lat,
    drone_lon,
    drone_heading_deg,
    pixel_length_m
):

    # --------------------------------------------------------
    # Distance from center of image in pixels
    # --------------------------------------------------------

    dx_pixels = (
        center_x -
        image_width / 2.0
    )

    dy_pixels = (
        center_y -
        image_height / 2.0
    )


    # --------------------------------------------------------
    # Convert pixels directly to meters
    # --------------------------------------------------------

    right_m = (
        dx_pixels *
        pixel_length_m
    )

    forward_m = (
        -dy_pixels *
        pixel_length_m
    )


    # --------------------------------------------------------
    # Rotate according to drone heading
    # --------------------------------------------------------

    heading = math.radians(
        drone_heading_deg
    )


    north_m = (
        forward_m * math.cos(heading)
        -
        right_m * math.sin(heading)
    )


    east_m = (
        forward_m * math.sin(heading)
        +
        right_m * math.cos(heading)
    )


    # --------------------------------------------------------
    # Convert meter offset to GPS coordinates
    # --------------------------------------------------------

    earth_radius_m = 6378137.0

    latitude_rad = math.radians(
        drone_lat
    )


    delta_lat = (
        north_m /
        earth_radius_m
    )


    delta_lon = (
        east_m /
        (
            earth_radius_m *
            math.cos(latitude_rad)
        )
    )


    object_lat = (
        drone_lat +
        math.degrees(delta_lat)
    )


    object_lon = (
        drone_lon +
        math.degrees(delta_lon)
    )


    return (
        object_lat,
        object_lon,
        north_m,
        east_m
    )


# ============================================================
# READ GPS LOG
# ============================================================

def read_gps_log(log_path):

    gps_data = {}


    with open(log_path, "r") as file:

        for line in file:

            line = line.strip()


            if not line:
                continue


            values = line.split(",")


            if len(values) < 5:
                continue


            try:

                image_index = int(
                    values[0]
                )

                latitude = float(
                    values[1]
                )

                longitude = float(
                    values[2]
                )

                altitude = float(
                    values[3]
                )

                heading = float(
                    values[4]
                )


                gps_data[image_index] = {

                    "latitude": latitude,

                    "longitude": longitude,

                    "altitude": altitude,

                    "heading": heading
                }


            except ValueError:

                print(
                    "Bad GPS line:",
                    line
                )


    return gps_data


# ============================================================
# GET IMAGE INDEX
# ============================================================

def get_image_index(filename):

    match = re.search(
        r"img_(\d+)",
        filename
    )


    if match:

        return int(
            match.group(1)
        )


    return None


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    detector,
    image_path,
    image_index,
    gps,
    output_folder
):

    image = cv2.imread(
        image_path
    )


    if image is None:

        print(
            "Could not read:",
            image_path
        )

        return []


    image_height, image_width = (
        image.shape[:2]
    )


    mask = detector.create_object_mask(
        image
    )


    objects = detector.find_objects(
        mask
    )


    output_image = image.copy()

    detections = []


    for object_index, contour in enumerate(objects):

        x, y, width, height = (
            cv2.boundingRect(
                contour
            )
        )


        center_x = (
            x + width / 2.0
        )

        center_y = (
            y + height / 2.0
        )


        area = cv2.contourArea(
            contour
        )


        (
            object_lat,
            object_lon,
            north_m,
            east_m

        ) = pixel_to_gps(

            center_x,
            center_y,

            image_width,
            image_height,

            gps["latitude"],
            gps["longitude"],

            gps["heading"],

            PIXEL_LENGTH_M
        )


        detection = {

            "image": image_index,

            "object": object_index,

            "center_x": center_x,

            "center_y": center_y,

            "area": area,

            "drone_lat": gps["latitude"],

            "drone_lon": gps["longitude"],

            "heading": gps["heading"],

            "object_lat": object_lat,

            "object_lon": object_lon,

            "north_offset_m": north_m,

            "east_offset_m": east_m
        }


        detections.append(
            detection
        )


        cv2.rectangle(

            output_image,

            (x, y),

            (
                x + width,
                y + height
            ),

            (0, 255, 0),

            3
        )


        cv2.circle(

            output_image,

            (
                int(center_x),
                int(center_y)
            ),

            7,

            (0, 0, 255),

            -1
        )


        cv2.putText(

            output_image,

            "Object {}".format(
                object_index
            ),

            (
                x,
                max(y - 10, 20)
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )


    mask_path = os.path.join(

        output_folder,

        "img_{}_thresholded.jpg".format(
            image_index
        )
    )


    detected_path = os.path.join(

        output_folder,

        "img_{}_detected.jpg".format(
            image_index
        )
    )


    cv2.imwrite(
        mask_path,
        mask
    )


    cv2.imwrite(
        detected_path,
        output_image
    )


    return detections


# ============================================================
# MAIN
# ============================================================

def main():

    gps_log_path = os.path.join(
        CAPTURE_FOLDER,
        GPS_LOG_NAME
    )


    output_folder = os.path.join(
        CAPTURE_FOLDER,
        OUTPUT_FOLDER_NAME
    )


    os.makedirs(
        output_folder,
        exist_ok=True
    )


    gps_data = read_gps_log(
        gps_log_path
    )


    detector = ObjectDetector(

        grass_colors=GRASS_COLORS,

        grass_distance_threshold=
        GRASS_DISTANCE_THRESHOLD,

        minimum_area=
        MINIMUM_OBJECT_AREA
    )


    filenames = [

        filename

        for filename in os.listdir(
            CAPTURE_FOLDER
        )

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )

        and filename.startswith(
            "img_"
        )
    ]


    filenames.sort(

        key=lambda filename:
        get_image_index(filename)
        if get_image_index(filename) is not None
        else 999999
    )


    all_detections = []


    for filename in filenames:

        image_index = get_image_index(
            filename
        )


        if image_index not in gps_data:

            print(
                "No GPS for",
                filename
            )

            continue


        print(
            "Processing:",
            filename
        )


        image_path = os.path.join(
            CAPTURE_FOLDER,
            filename
        )


        detections = process_image(

            detector,

            image_path,

            image_index,

            gps_data[image_index],

            output_folder
        )


        all_detections.extend(
            detections
        )


        print(
            "Objects:",
            len(detections)
        )


    csv_path = os.path.join(
        output_folder,
        OUTPUT_CSV_NAME
    )


    fieldnames = [

        "image",

        "object",

        "center_x",

        "center_y",

        "area",

        "drone_lat",

        "drone_lon",

        "heading",

        "object_lat",

        "object_lon",

        "north_offset_m",

        "east_offset_m"
    ]


    with open(
        csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )


        writer.writeheader()


        for detection in all_detections:

            writer.writerow(
                detection
            )


    print()
    print("Finished")

    print(
        "Objects found:",
        len(all_detections)
    )

    print(
        "Output:",
        output_folder
    )


if __name__ == "__main__":
    main()