# tests/test_task3_flow_follow.py
import cv2
import numpy as np


def test_task3_optical_flow_two_opposite_flows():
    h, w = 240, 320
    rng = np.random.default_rng(0)

    # Build two frames: top moves +dx, bottom moves -dx
    f0 = np.ones((h, w, 3), dtype=np.uint8) * 255
    f1 = np.ones((h, w, 3), dtype=np.uint8) * 255

    n = 25
    radius = 6
    dx = 3

    top = np.column_stack([rng.uniform(0, w, n), rng.uniform(0, h * 0.45, n)])
    bot = np.column_stack([rng.uniform(0, w, n), rng.uniform(h * 0.55, h, n)])

    for x, y in top:
        cv2.circle(f0, (int(x), int(y)), radius, (0, 0, 0), -1)
        cv2.circle(f1, (int(x + dx), int(y)), radius, (0, 0, 0), -1)

    for x, y in bot:
        cv2.circle(f0, (int(x), int(y)), radius, (0, 0, 0), -1)
        cv2.circle(f1, (int(x - dx), int(y)), radius, (0, 0, 0), -1)

    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        g0, g1, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    vx = flow[..., 0]
    top_vx = float(np.mean(vx[: int(h * 0.45), :]))
    bot_vx = float(np.mean(vx[int(h * 0.55):, :]))

    assert top_vx > 0.0
    assert bot_vx < 0.0
