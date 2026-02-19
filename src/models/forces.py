import numpy as np

def clamp_speed(v: np.ndarray, vmax: float) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    if n <= vmax:
        return v
    return v * (vmax / n)

def repulsion_smooth(xi, xj, krep: float, Rsafe: float):
  
    dvec = xi - xj
    d = np.linalg.norm(dvec)
    if d < 1e-6:
        return np.random.uniform(-1, 1, size=2) * 0.1 * krep
    if d >= Rsafe:
        return np.array([0.0, 0.0])
    dir = dvec / d
    mag = (1.0/d - 1.0/Rsafe)
    return krep * mag * dir

def clamp_vec(v: np.ndarray, vmax: float) -> np.ndarray:
    return clamp_speed(v, vmax)


def repulsion_sum(x: np.ndarray, centers: list, krep: float, Rsafe: float) -> np.ndarray:
 
    F = np.zeros(2, dtype=float)
    for c in centers:
        F += repulsion_smooth(x, c, krep=krep, Rsafe=Rsafe)

    if not np.all(np.isfinite(F)):
        return np.array([0.0, 0.0], dtype=float)
    return F
