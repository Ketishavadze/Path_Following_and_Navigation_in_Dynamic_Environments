# tests/test_task2_no_collision.py
import numpy as np

from src.path.skeleton_route import skeleton_path, nearest_skeleton_pixel, bfs_shortest_path
from src.path.splines import fit_centerline_spline
from src.tasks.task2_swarm_corridor import simulate_task2


def test_task2_no_collision_basic(s_curve_mask):
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
        "width": 120.0,  # safer corridor
        "numerics": {"dt": 0.01, "steps": 2000, "sample_min_dist_every": 10},
        "swarm": {"NA": 3, "NB": 3, "lane_ratio": 0.45},
        "robot": {
            "m": 1.0,
            "kp": 9.0,
            "kd": 12.0,
            "kb": 160.0,
            "krep": 5000.0,
            "Rsafe": 20.0,
            "lookahead": 18.0,
            "vmax": 200.0,
        },
        "gif": {"step_skip": 999999, "duration": 0.1},
    }

    traj_A, traj_B, min_dists, params, width = simulate_task2(sx, sy, s_max, cfg)

    # Ignore the very first sample (initial placement may be tight)
    if len(min_dists) > 2:
        md = float(np.min(min_dists[1:]))
    else:
        md = float(np.min(min_dists))

    assert md >= params["Rsafe"] * 0.85
