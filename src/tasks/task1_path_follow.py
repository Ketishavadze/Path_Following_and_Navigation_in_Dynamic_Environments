import os
import sys
import numpy as np

# --- allow running this file directly ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.common.io import load_binary_path_mask, load_A_B_from_snapshot
from src.path.skeleton_route import skeleton_path, nearest_skeleton_pixel, bfs_shortest_path
from src.path.splines import fit_centerline_spline, closest_s_on_spline
from src.path.corridor import border_force
from src.numerics.rk import rk4_step
from src.common.config import load_config
from src.common.viz import save_task1_outputs


def simulate_task1(sx, sy, s_max: float, cfg: dict):
    width = float(cfg["width"])

    dt = float(cfg["numerics"]["dt"])
    steps = int(cfg["numerics"]["steps"])
    s_samples_n = int(cfg["numerics"].get("s_samples", 900))


    robot = cfg["robot"]
    m = float(robot["m"])
    kp = float(robot["kp"])
    kd = float(robot["kd"])
    kb = float(robot["kb"])
    lookahead = float(robot["lookahead"])

    end_tol = float(cfg.get("stop", {}).get("end_tol", 6.0))


    # start state
    x0 = np.array([sx(0.0), sy(0.0)], dtype=float)
    v0 = np.array([0.0, 0.0], dtype=float)
    state = np.array([x0[0], x0[1], v0[0], v0[1]], dtype=float)

    s_samples = np.linspace(0.0, s_max, s_samples_n)
    s_progress = 0.0

    traj = []
    border_dist = []

    def accel(x, v):
        nonlocal s_progress
        s_closest = closest_s_on_spline(s_samples, sx, sy, x)
        s_progress = max(s_progress, s_closest)

        s_target = min(s_progress + lookahead, s_max)
        T = np.array([sx(s_target), sy(s_target)], dtype=float)

        a_track = (kp * (T - x) - kd * v) / m
        a_border = border_force(x, s_closest, sx, sy, width, kb) / m
        return a_track + a_border

    for _ in range(steps):
        traj.append(state.copy())

        # border violation proxy (0 = inside)
        s_closest = closest_s_on_spline(s_samples, sx, sy, state[0:2])
        bf = border_force(state[0:2], s_closest, sx, sy, width, kb)
        outside_amount = np.linalg.norm(bf) / max(kb, 1e-9)
        border_dist.append(outside_amount)

        state = rk4_step(state, dt, accel)

        # stop near end
        end = np.array([sx(s_max), sy(s_max)], dtype=float)
        if np.linalg.norm(state[0:2] - end) < end_tol:
            traj.append(state.copy())
            break

    return np.array(traj), np.array(border_dist), width


def main():
    cfg = load_config("data/params/task1_default.yaml")

    raw_path = "data/map/map_raw.png"
    snap_path = "data/map/map_snapshot_A_B.png"

    mask = load_binary_path_mask(raw_path)
    A, B = load_A_B_from_snapshot(snap_path)

    skel = skeleton_path(mask)
    A_s = nearest_skeleton_pixel(skel, A)
    B_s = nearest_skeleton_pixel(skel, B)

    path_pixels = bfs_shortest_path(skel, A_s, B_s)
    s, sx, sy = fit_centerline_spline(path_pixels)
    s_max = float(s[-1])

    traj, border_dist, width = simulate_task1(sx, sy, s_max, cfg)
    save_task1_outputs(mask, sx, sy, s_max, width, traj, border_dist)


if __name__ == "__main__":
    main()
