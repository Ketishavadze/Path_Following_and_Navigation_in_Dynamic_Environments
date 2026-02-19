import os
import sys
import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.common.config import load_config
from src.models.robot import Robot
from src.models.controllers import desired_velocity_with_direction, step_state_2d
from src.models.forces import repulsion_sum
from src.common.viz import draw_ped_centers_cv, draw_robot_cv


def detect_pedestrians(frame, thresh=60, area_min=40, area_max=400):
    """Detect black circles on white background and return their centers."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if float(area_min) < area < float(area_max):
            M = cv2.moments(c)
            if M["m00"] > 1e-9:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
                centers.append(np.array([cx, cy], dtype=float))
    return centers


def sample_flow(flow, pos, w, h, flow_scale, patch=2):
    """
    Average flow in a (2*patch+1)x(2*patch+1) neighborhood for stability.
    patch=2 -> 5x5 area.
    """
    x = int(np.clip(pos[0], 0, w - 1))
    y = int(np.clip(pos[1], 0, h - 1))

    x0 = max(0, x - patch)
    x1 = min(w, x + patch + 1)
    y0 = max(0, y - patch)
    y1 = min(h, y + patch + 1)

    V = np.mean(flow[y0:y1, x0:x1, :], axis=(0, 1)).astype(float) * float(flow_scale)
    if not np.all(np.isfinite(V)):
        return np.array([0.0, 0.0], dtype=float)
    return V



def update_robot(robot: Robot, flow, ped_centers, cfg, w, h):
    of = cfg["optical_flow"]
    ctrl = cfg["control"]
    lim = cfg["limits"]

    V = sample_flow(flow, robot.pos, w, h, of["flow_scale"])
    vdes = desired_velocity_with_direction(V, robot.direction, lim["vmin_x"], lim["clip_vy"])

    a_track = float(ctrl["kv"]) * (vdes - robot.vel)
    a_damp = -float(ctrl["kd"]) * robot.vel
    a_rep = repulsion_sum(robot.pos, ped_centers, float(ctrl["krep"]), float(ctrl["rsafe"]))

    robot_cfg = cfg["robots"]["kvrc"] if robot.name == cfg["robots"]["kvrc"]["name"] else cfg["robots"]["xbot"]
    y_target = float(h) * float(robot_cfg["lane_y_ratio"])
    klane = float(ctrl.get("klane", 0.0))
    klane_d = float(ctrl.get("klane_d", 0.0))
    a_lane = np.array([0.0, klane * (y_target - robot.pos[1]) - klane_d * robot.vel[1]], dtype=float)

    accel = (a_track + a_damp + a_rep + a_lane) / float(ctrl["m"])


    new_pos, new_vel = step_state_2d(
        robot.pos, robot.vel, accel,
        dt=float(lim["dt"]),
        amax=float(lim["amax"]),
        vmax=float(lim["vmax"]),
        w=w, h=h
    )
    if new_pos is None:
        return False
    robot.pos, robot.vel = new_pos, new_vel
    return True


def run_task3(cfg):
    os.makedirs("outputs/task3", exist_ok=True)

    t3 = cfg["task3"]
    of = cfg["optical_flow"]
    det = cfg["pedestrian_detection"]
    ren = cfg["render"]
    robots_cfg = cfg["robots"]

    video_path = t3["video_path"]
    output_path = t3["output_path"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame.")

    h, w = prev_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    kvrc_cfg = robots_cfg["kvrc"]
    xbot_cfg = robots_cfg["xbot"]

    kvrc = Robot(
        name=kvrc_cfg["name"],
        pos=np.array([float(kvrc_cfg["start_x"]), float(h) * float(kvrc_cfg["start_y_ratio"])], dtype=float),
        vel=np.array([0.0, 0.0], dtype=float),
        direction=int(kvrc_cfg["start_dir"]),
        color_bgr=tuple(kvrc_cfg["color_bgr"]),
    )

    xbot = Robot(
        name=xbot_cfg["name"],
        pos=np.array([float(w) - float(xbot_cfg["start_x_from_right"]),
                      float(h) * float(xbot_cfg["start_y_ratio"])], dtype=float),
        vel=np.array([0.0, 0.0], dtype=float),
        direction=int(xbot_cfg["start_dir"]),
        color_bgr=tuple(xbot_cfg["color_bgr"]),
    )

    margin = float(ren.get("bounds_margin", 30))
    left_x = margin
    right_x = float(w) - margin

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            float(of["pyr_scale"]),
            int(of["levels"]),
            int(of["winsize"]),
            int(of["iterations"]),
            int(of["poly_n"]),
            float(of["poly_sigma"]),
            int(of["flags"]),
        )

        ped_centers = detect_pedestrians(
            frame,
            thresh=det["thresh"],
            area_min=det["area_min"],
            area_max=det["area_max"],
        )

        if not update_robot(kvrc, flow, ped_centers, cfg, w, h):
            kvrc.pos = np.array([float(kvrc_cfg["start_x"]), float(h) * float(kvrc_cfg["start_y_ratio"])], dtype=float)
            kvrc.vel = np.array([0.0, 0.0], dtype=float)
            kvrc.direction = int(kvrc_cfg["start_dir"])

        if not update_robot(xbot, flow, ped_centers, cfg, w, h):
            xbot.pos = np.array([float(w) - float(xbot_cfg["start_x_from_right"]),
                                 float(h) * float(xbot_cfg["start_y_ratio"])], dtype=float)
            xbot.vel = np.array([0.0, 0.0], dtype=float)
            xbot.direction = int(xbot_cfg["start_dir"])

        if kvrc.direction > 0 and kvrc.pos[0] >= right_x - 5:
            kvrc.direction = -1
        elif kvrc.direction < 0 and kvrc.pos[0] <= left_x + 5:
            kvrc.direction = +1

        if xbot.direction < 0 and xbot.pos[0] <= left_x + 5:
            xbot.direction = +1
        elif xbot.direction > 0 and xbot.pos[0] >= right_x - 5:
            xbot.direction = -1

        if ren.get("draw_ped_centers", True):
            draw_ped_centers_cv(frame, ped_centers, radius=ren["ped_center_radius"])

        draw_robot_cv(frame, kvrc.name, kvrc.pos, kvrc.color_bgr,
                      radius=ren["robot_radius"],
                      label_scale=ren["label_scale"],
                      label_thickness=ren["label_thickness"])

        draw_robot_cv(frame, xbot.name, xbot.pos, xbot.color_bgr,
                      radius=ren["robot_radius"],
                      label_scale=ren["label_scale"],
                      label_thickness=ren["label_thickness"])

        cv2.putText(frame, f"frame {frame_index}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 2)

        writer.write(frame)
        prev_gray = gray
        frame_index += 1

    cap.release()
    writer.release()
    print(f"Saved result video to: {output_path}")


def main():
    cfg = load_config("data/params/task3_default.yaml")
    run_task3(cfg)


if __name__ == "__main__":
    main()
