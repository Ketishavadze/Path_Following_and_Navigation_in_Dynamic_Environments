import numpy as np

def rk4_step(state: np.ndarray, dt: float, accel_func):
    """
    state = [x, y, vx, vy]
    accel_func(x, v) -> [ax, ay]
    """
    def f(st):
        x = st[0:2]
        v = st[2:4]
        a = accel_func(x, v)
        return np.array([v[0], v[1], a[0], a[1]], dtype=float)

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
