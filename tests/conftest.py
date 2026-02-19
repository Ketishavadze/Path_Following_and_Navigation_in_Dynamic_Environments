import os
import cv2
import numpy as np
import pytest


def make_s_curve_mask(h=260, w=360, thickness=26):
    
    img = np.zeros((h, w), dtype=np.uint8)

    pts = np.array([
        (60, 40),
        (130, 40),
        (220, 70),
        (250, 130),
        (220, 190),
        (130, 220),
        (60, 220),
    ], dtype=np.int32)

    cv2.polylines(img, [pts], isClosed=False, color=255, thickness=thickness, lineType=cv2.LINE_AA)

    cv2.circle(img, tuple(pts[0]), thickness // 2, 255, -1)
    cv2.circle(img, tuple(pts[-1]), thickness // 2, 255, -1)

    mask = (img > 0).astype(np.uint8)
    return mask


def write_mask_png(tmp_path, mask, name="map_raw.png"):
    path = tmp_path / name
    cv2.imwrite(str(path), (mask * 255).astype(np.uint8))
    return path


def make_two_flow_frames(h=240, w=320, n=30, radius=6, dx=3):
   
    rng = np.random.default_rng(0)

    top = np.column_stack([rng.uniform(0, w, n), rng.uniform(0, h * 0.45, n)])
    bot = np.column_stack([rng.uniform(0, w, n), rng.uniform(h * 0.55, h, n)])

    f0 = np.ones((h, w, 3), dtype=np.uint8) * 255
    for x, y in top:
        cv2.circle(f0, (int(x), int(y)), radius, (0, 0, 0), -1)
    for x, y in bot:
        cv2.circle(f0, (int(x), int(y)), radius, (0, 0, 0), -1)

    top1 = top.copy()
    top1[:, 0] += dx

    bot1 = bot.copy()
    bot1[:, 0] -= dx

    f1 = np.ones((h, w, 3), dtype=np.uint8) * 255
    for x, y in top1:
        cv2.circle(f1, (int(x), int(y)), radius, (0, 0, 0), -1)
    for x, y in bot1:
        cv2.circle(f1, (int(x), int(y)), radius, (0, 0, 0), -1)

    return f0, f1


@pytest.fixture
def s_curve_mask():
    return make_s_curve_mask()
