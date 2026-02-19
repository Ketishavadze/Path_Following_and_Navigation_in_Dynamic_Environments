import numpy as np
import pytest

from src.path.skeleton_route import skeleton_path, nearest_skeleton_pixel, bfs_shortest_path
from src.path.splines import fit_centerline_spline
from src.tasks.task1_path_follow import simulate_task1


def test_task1_following_inside_corridor(s_curve_mask):
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
        "numerics": {"dt": 0.02, "steps": 2500, "s_samples": 600},
        "robot": {"m": 1.0, "kp": 8.0, "kd": 6.0, "kb": 140.0, "lookahead": 10.0},
        "stop": {"end_tol": 8.0},
    }

    traj, border_dist, width = simulate_task1(sx, sy, s_max, cfg)

    end = np.array([sx(s_max), sy(s_max)], dtype=float)
    assert np.linalg.norm(traj[-1, 0:2] - end) < 20.0

    assert float(np.max(border_dist)) < 0.25


@pytest.mark.xfail(reason="Limitation: corridor too narrow for this robot/controller -> border violations expected.")
def test_task1_fails_when_corridor_too_narrow(s_curve_mask):
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
        "width": 25.0, 
        "numerics": {"dt": 0.02, "steps": 2500, "s_samples": 600},
        "robot": {"m": 1.0, "kp": 9.0, "kd": 3.0, "kb": 120.0, "lookahead": 10.0},
        "stop": {"end_tol": 8.0},
    }

    traj, border_dist, _ = simulate_task1(sx, sy, s_max, cfg)

    assert float(np.max(border_dist)) < 0.25
