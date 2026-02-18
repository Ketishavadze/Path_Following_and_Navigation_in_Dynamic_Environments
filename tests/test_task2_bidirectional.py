# tests/test_task2_bidirectional.py
import numpy as np

from src.path.skeleton_route import skeleton_path, nearest_skeleton_pixel, bfs_shortest_path
from src.path.splines import fit_centerline_spline
from src.tasks.task2_swarm_corridor import simulate_task2


def test_task2_bidirectional_progress(s_curve_mask):
    mask = s_curve_mask
    A = (60, 40)
    B = (60, 220)

    skel = skeleton_path(mask)
    A_s = nearest_skeleton_pixel(skel, A)
    B_s = nearest_skeleton_pixel(skel, B)
    path_pixels = bfs_shortest_path(skel, A_s, B_s)

    s, sx, sy = fit_centerline_spline(path_pixels)
    s_max = float(s[-1])

    cfg = {
        "width": 90.0,
        "numerics": {"dt": 0.01, "steps": 1200, "sample_min_dist_every": 20},
        "swarm": {"NA": 4, "NB": 4, "lane_ratio": 0.33},
        "robot": {
            "m": 1.0, "kp": 8.0, "kd": 10.0, "kb": 140.0,
            "krep": 2500.0, "Rsafe": 26.0, "lookahead": 16.0, "vmax": 180.0
        },
        "gif": {"step_skip": 999999, "duration": 0.1},
    }

    traj_A, traj_B, min_dists, params, width = simulate_task2(sx, sy, s_max, cfg)

    # Use robot 0 from each group: check net displacement roughly toward opposite ends
    A0_start = traj_A[0, 0]
    A0_end = traj_A[-1, 0]
    B0_start = traj_B[0, 0]
    B0_end = traj_B[-1, 0]

    # They should not end at exactly the same place; should move noticeably
    assert np.linalg.norm(A0_end - A0_start) > 20.0
    assert np.linalg.norm(B0_end - B0_start) > 20.0
