import numpy as np
from scipy.interpolate import CubicSpline

def fit_centerline_spline(path_pixels):
    # (row,col) -> (x,y)
    pts = np.array([(c, r) for (r, c) in path_pixels], dtype=float)
    dif = np.diff(pts, axis=0)
    seg = np.sqrt((dif ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])

    keep = np.concatenate([[True], seg > 1e-9])
    s = s[keep]
    pts = pts[keep]

    sx = CubicSpline(s, pts[:, 0])
    sy = CubicSpline(s, pts[:, 1])
    return s, sx, sy

def closest_s_on_spline(s_samples: np.ndarray, sx, sy, x: np.ndarray) -> float:
    pts = np.stack([sx(s_samples), sy(s_samples)], axis=1)
    d2 = np.sum((pts - x[None, :]) ** 2, axis=1)
    return float(s_samples[int(np.argmin(d2))])
