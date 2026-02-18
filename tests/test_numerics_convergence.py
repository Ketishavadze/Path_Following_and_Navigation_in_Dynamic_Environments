# tests/test_numerics_convergence.py
import numpy as np
from src.numerics.rk import rk4_step


def test_rk4_convergence_harmonic_oscillator():
    """
    1D harmonic oscillator embedded in your 2D state:
      x'' = -x
    Exact: x(t)=cos(t), v(t)=-sin(t) for x(0)=1, v(0)=0
    RK4 should get more accurate when dt is halved.
    """
    def accel(x, v):
        ax = -x[0]
        return np.array([ax, 0.0], dtype=float)

    def run(dt):
        T = 2.0  # seconds
        steps = int(T / dt)
        state = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # x=1, vx=0
        for _ in range(steps):
            state = rk4_step(state, dt, accel)
        return state

    s1 = run(0.10)
    s2 = run(0.05)

    T = 2.0
    exact_x = np.cos(T)

    e1 = abs(s1[0] - exact_x)
    e2 = abs(s2[0] - exact_x)

    assert e2 < e1
    assert e2 < 0.5 * e1  # should improve clearly
