import numpy as np
from src.models.forces import repulsion_smooth
from src.path.splines import closest_s_on_spline
from src.path.corridor import border_force, corridor_normal

def make_swarm_states(sx, sy, s_start, count, lane_offset):
    states = []
    for i in range(count):
        si = float(s_start + i * 7.0)
        pos = np.array([sx(si), sy(si)], dtype=float)
        n = corridor_normal(sx, sy, si)
        pos = pos + lane_offset * n + np.random.uniform(-2, 2, size=2)
        v = np.array([0.0, 0.0], dtype=float)
        states.append(np.array([pos[0], pos[1], v[0], v[1]], dtype=float))
    return np.array(states)

def min_pairwise_distance(all_pos: np.ndarray) -> float:
    dmin = 1e9
    for i in range(all_pos.shape[0]):
        for j in range(i+1, all_pos.shape[0]):
            d = np.linalg.norm(all_pos[i] - all_pos[j])
            if d < dmin:
                dmin = d
    return dmin

def accel_bidirectional(
    group: str, idx: int, my_gid: int,
    x: np.ndarray, v: np.ndarray,
    sx, sy, s_max: float,
    s_samples: np.ndarray,
    width: float,
    params: dict,
    prog_A: np.ndarray,
    prog_B: np.ndarray,
    all_positions: np.ndarray
):
    m = params["m"]
    kp = params["kp"]
    kd = params["kd"]
    kb = params["kb"]
    krep = params["krep"]
    Rsafe = params["Rsafe"]
    lookahead = params["lookahead"]

    s_closest = closest_s_on_spline(s_samples, sx, sy, x)

    if group == "A":
        prog_A[idx] = max(prog_A[idx], s_closest)
        s_target = min(prog_A[idx] + lookahead, s_max)
    else:
        prog_B[idx] = min(prog_B[idx], s_closest)
        s_target = max(prog_B[idx] - lookahead, 0.0)

    T = np.array([sx(s_target), sy(s_target)], dtype=float)
    a_track = (kp * (T - x) - kd * v) / m
    a_border = border_force(x, s_closest, sx, sy, width, kb) / m

    a_rep = np.array([0.0, 0.0])
    for j, xj in enumerate(all_positions):
        if j == my_gid:
            continue
        a_rep += repulsion_smooth(x, xj, krep, Rsafe) / m

    return a_track + a_border + a_rep
