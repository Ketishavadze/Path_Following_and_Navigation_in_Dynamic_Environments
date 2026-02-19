import numpy as np

def tracking_accel(x, v, T, kp, kd, m=1.0):
    return (kp * (T - x) - kd * v) / m

def desired_velocity_with_direction(flow_v: np.ndarray, direction_sign: int, vmin_x: float, clip_vy: float) -> np.ndarray:

    vdes = flow_v.astype(float).copy()
    ax = max(abs(float(vdes[0])), float(vmin_x))
    vdes[0] = float(direction_sign) * ax
    vdes[1] = float(np.clip(vdes[1], -float(clip_vy), float(clip_vy)))
    return vdes


def step_state_2d(pos: np.ndarray, vel: np.ndarray, accel: np.ndarray,
                  dt: float, amax: float, vmax: float, w: int, h: int):

    from src.models.forces import clamp_vec

    if not np.all(np.isfinite(accel)):
        accel = np.array([0.0, 0.0], dtype=float)
    accel = clamp_vec(accel, float(amax))

    vel = vel + accel * float(dt)
    if not np.all(np.isfinite(vel)):
        vel = np.array([0.0, 0.0], dtype=float)
    vel = clamp_vec(vel, float(vmax))

    pos = pos + vel * float(dt)
    if not np.all(np.isfinite(pos)):
        return None, None

    pos[0] = float(np.clip(pos[0], 0, w - 1))
    pos[1] = float(np.clip(pos[1], 0, h - 1))
    return pos, vel
