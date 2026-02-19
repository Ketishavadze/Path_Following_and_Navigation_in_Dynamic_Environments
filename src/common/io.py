import cv2
import numpy as np

def load_binary_path_mask(path_img_path: str) -> np.ndarray:
    img = cv2.imread(path_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path_img_path}")

    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (bw > 0).astype(np.uint8)


def load_A_B_from_snapshot(snapshot_path: str):
    """
    A = green, B = red. Returns (row, col).
    """
    img = cv2.imread(snapshot_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {snapshot_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, (35, 80, 80), (85, 255, 255))
    red1  = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2  = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    def centroid(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return int(np.mean(ys)), int(np.mean(xs))

    A = centroid(green)
    B = centroid(red)
    if A is None or B is None:
        raise ValueError("Could not detect A (green) or B (red).")
    return A, B
