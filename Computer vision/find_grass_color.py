import cv2

image_path = "capture_2\img_5.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(image_path)


def get_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = image[y, x]

        print(f"Clicked at x={x}, y={y}")
        print(f"BGR: ({b}, {g}, {r})")
        print(f"RGB: ({r}, {g}, {b})")
        print()


cv2.imshow("Click on grass colors", image)

cv2.setMouseCallback(
    "Click on grass colors",
    get_color
)

print("Click on different parts of the grass.")
print("Press Q to close.")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()