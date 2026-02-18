import os
import sys
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.common.io import load_binary_path_mask, load_A_B_from_snapshot
from src.path.skeleton_route import skeleton_path, nearest_skeleton_pixel, bfs_shortest_path
from src.path.splines import fit_centerline_spline
from src.numerics.rk import rk4_step
from src.models.forces import clamp_speed
from src.models.swarm import make_swarm_states, accel_bidirectional, min_pairwise_distance
from src.common.config import load_config
from src.common.viz import save_task2_plots, save_task2_gif


def simulate_task2(sx, sy, s_max: float, cfg: dict):
    width = float(cfg["width"])

    dt = float(cfg["numerics"]["dt"])
    steps = int(cfg["numerics"]["steps"])
    sample_min_dist_every = int(cfg["numerics"]["sample_min_dist_every"])

    NA = int(cfg["swarm"]["NA"])
    NB = int(cfg["swarm"]["NB"])
    lane_ratio = float(cfg["swarm"]["lane_ratio"])
    lane = width * lane_ratio

    params = dict(cfg["robot"])  # kp,kd,krep,Rsafe,lookahead,vmax...
    params["m"] = float(params["m"])
    params["kp"] = float(params["kp"])
    params["kd"] = float(params["kd"])
    params["kb"] = float(params["kb"])
    params["krep"] = float(params["krep"])
    params["Rsafe"] = float(params["Rsafe"])
    params["lookahead"] = float(params["lookahead"])
    params["vmax"] = float(params["vmax"])

    s_samples = np.linspace(0.0, s_max, 1200)

    states_A = make_swarm_states(sx, sy, s_start=0.0, count=NA, lane_offset=+lane)
    states_B = make_swarm_states(sx, sy, s_start=max(0.0, s_max - 1.0), count=NB, lane_offset=-lane)

    prog_A = np.zeros(NA)
    prog_B = np.ones(NB) * s_max

    traj_A = []
    traj_B = []
    min_dists = []

    for step in range(steps):
        pos_A = states_A[:, 0:2]
        pos_B = states_B[:, 0:2]
        all_pos = np.vstack([pos_A, pos_B])

        traj_A.append(pos_A.copy())
        traj_B.append(pos_B.copy())

        if step % sample_min_dist_every == 0:
            min_dists.append(min_pairwise_distance(all_pos))

        # Update A
        new_A = states_A.copy()
        for i in range(NA):
            my_gid = i

            def acc(x, v, i=i, my_gid=my_gid):
                return accel_bidirectional(
                    "A", i, my_gid, x, v,
                    sx, sy, s_max,
                    s_samples, width,
                    params, prog_A, prog_B,
                    all_pos
                )

            st = rk4_step(states_A[i], dt, acc)
            st[2:4] = clamp_speed(st[2:4], params["vmax"])
            new_A[i] = st

        # Update B
        new_B = states_B.copy()
        for i in range(NB):
            my_gid = NA + i

            def acc(x, v, i=i, my_gid=my_gid):
                return accel_bidirectional(
                    "B", i, my_gid, x, v,
                    sx, sy, s_max,
                    s_samples, width,
                    params, prog_A, prog_B,
                    all_pos
                )

            st = rk4_step(states_B[i], dt, acc)
            st[2:4] = clamp_speed(st[2:4], params["vmax"])
            new_B[i] = st

        states_A, states_B = new_A, new_B

        # Early stop when most reached ends
        endB = np.array([sx(s_max), sy(s_max)], dtype=float)
        endA = np.array([sx(0.0), sy(0.0)], dtype=float)

        reached_A = np.mean(np.linalg.norm(states_A[:, 0:2] - endB[None, :], axis=1) < 14.0)
        reached_B = np.mean(np.linalg.norm(states_B[:, 0:2] - endA[None, :], axis=1) < 14.0)
        if reached_A > 0.8 and reached_B > 0.8:
            break

    return np.array(traj_A), np.array(traj_B), np.array(min_dists), params, width


def main():
    cfg = load_config("data/params/task2_default.yaml")

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

    traj_A, traj_B, min_dists, params, width = simulate_task2(sx, sy, s_max, cfg)

    save_task2_plots(mask, sx, sy, s_max, width, traj_A, traj_B, min_dists, params)
    save_task2_gif(mask, sx, sy, s_max, width, traj_A, traj_B, cfg.get("gif", {}))


if __name__ == "__main__":
    main()
