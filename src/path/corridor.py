import numpy as np

def corridor_normal(sx, sy, s_val: float) -> np.ndarray:
    dx = float(sx.derivative()(s_val))
    dy = float(sy.derivative()(s_val))
    t = np.array([dx, dy], dtype=float)
    nrm = np.linalg.norm(t)
    if nrm < 1e-9:
        return np.array([0.0, 0.0])
    t = t / nrm
    return np.array([-t[1], t[0]])

def corridor_bounds(sx, sy, s_grid, width: float):
    center = np.stack([sx(s_grid), sy(s_grid)], axis=1)
    left = []
    right = []
    for sv in s_grid:
        n = corridor_normal(sx, sy, float(sv))
        c = np.array([sx(sv), sy(sv)], dtype=float)
        left.append(c + (width/2)*n)
        right.append(c - (width/2)*n)
    return center, np.array(left), np.array(right)

def border_force(x: np.ndarray, s0: float, sx, sy, width: float, kb: float) -> np.ndarray:
    center = np.array([sx(s0), sy(s0)], dtype=float)
    n = corridor_normal(sx, sy, s0)
    lat = float(np.dot(x - center, n))
    half = width / 2.0
    if abs(lat) <= half:
        return np.array([0.0, 0.0])
    outside = abs(lat) - half
    direction = -np.sign(lat) * n
    return kb * outside * direction
