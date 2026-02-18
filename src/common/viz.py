import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2


def save_task1_outputs(mask, sx, sy, s_max, width, traj, border_dist, out_dir="outputs/task1"):
    os.makedirs(out_dir, exist_ok=True)

    from src.path.corridor import corridor_bounds  # local import

    s_grid = np.linspace(0.0, s_max, 700)
    center, left, right = corridor_bounds(sx, sy, s_grid, width)

    # Map plot
    plt.figure(figsize=(9, 9))
    plt.imshow(mask, cmap="gray", origin="upper")
    plt.plot(center[:, 0], center[:, 1], linewidth=2)
    plt.plot(left[:, 0], left[:, 1], linewidth=1)
    plt.plot(right[:, 0], right[:, 1], linewidth=1)

    plt.plot(traj[:, 0], traj[:, 1], linewidth=2)
    plt.scatter([traj[0, 0]], [traj[0, 1]], s=60)
    plt.scatter([traj[-1, 0]], [traj[-1, 1]], s=60)

    plt.title("Task 1: Path + corridor + robot trajectory")
    plt.tight_layout()
    p1 = os.path.join(out_dir, "task1_map.png")
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"Saved: {p1}")

    # Border violation plot
    plt.figure(figsize=(7, 4))
    plt.plot(border_dist)
    plt.title("Task 1: Border violation indicator (0 = inside)")
    plt.xlabel("step")
    plt.ylabel("outside amount (proxy)")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "task1_border_violation.png")
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"Saved: {p2}")

def save_task2_plots(mask, sx, sy, s_max, width, traj_A, traj_B, min_dists, params, out_dir="outputs/task2"):
    os.makedirs(out_dir, exist_ok=True)

    from src.path.corridor import corridor_bounds  # local import to avoid circulars

    s_grid = np.linspace(0.0, s_max, 700)
    center, left, right = corridor_bounds(sx, sy, s_grid, width)

    # Map plot
    plt.figure(figsize=(9, 9))
    plt.imshow(mask, cmap="gray", origin="upper")
    plt.plot(center[:, 0], center[:, 1], linewidth=2)
    plt.plot(left[:, 0], left[:, 1], linewidth=1)
    plt.plot(right[:, 0], right[:, 1], linewidth=1)

    if len(traj_A) > 0:
        A_last = traj_A[-1]
        B_last = traj_B[-1]
        plt.scatter(A_last[:, 0], A_last[:, 1], s=25, label="A→B")
        plt.scatter(B_last[:, 0], B_last[:, 1], s=25, label="B→A")

        # One trajectory from each group for clarity
        plt.plot(traj_A[:, 0, 0], traj_A[:, 0, 1], linewidth=2)
        plt.plot(traj_B[:, 0, 0], traj_B[:, 0, 1], linewidth=2)

    plt.title(f"Task 2: Bidirectional swarms (Rsafe={params['Rsafe']}px)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    path1 = os.path.join(out_dir, "task2_map.png")
    plt.savefig(path1, dpi=200)
    plt.close()
    print(f"Saved: {path1}")

    # Min distance plot
    plt.figure(figsize=(7, 4))
    plt.plot(min_dists)
    plt.axhline(params["Rsafe"], linestyle="--")
    plt.title("Task 2: Minimum pairwise distance (sampled)")
    plt.xlabel("sample index")
    plt.ylabel("min distance (pixels)")
    plt.tight_layout()
    path2 = os.path.join(out_dir, "task2_min_distance.png")
    plt.savefig(path2, dpi=200)
    plt.close()
    print(f"Saved: {path2}")


def save_task2_gif(mask, sx, sy, s_max, width, traj_A, traj_B, gif_cfg, out_dir="outputs/task2"):
    os.makedirs(out_dir, exist_ok=True)

    from src.path.corridor import corridor_bounds  # local import

    s_grid = np.linspace(0.0, s_max, 700)
    center, left, right = corridor_bounds(sx, sy, s_grid, width)

    step_skip = int(gif_cfg.get("step_skip", 20))
    duration = float(gif_cfg.get("duration", 0.08))

    frames = []
    for k in range(0, len(traj_A), step_skip):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(mask, cmap="gray", origin="upper")

        ax.plot(center[:, 0], center[:, 1], linewidth=2)
        ax.plot(left[:, 0], left[:, 1], linewidth=1)
        ax.plot(right[:, 0], right[:, 1], linewidth=1)

        A_pos = traj_A[k]
        B_pos = traj_B[k]
        ax.scatter(A_pos[:, 0], A_pos[:, 1], s=40)
        ax.scatter(B_pos[:, 0], B_pos[:, 1], s=40)

        ax.set_title(f"Task 2 animation (frame {k})")
        ax.set_xlim(0, mask.shape[1])
        ax.set_ylim(mask.shape[0], 0)

        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(frame)
        plt.close(fig)

    gif_path = os.path.join(out_dir, "task2_animation.gif")
    imageio.mimsave(gif_path, frames, duration=duration)
    print(f"Saved: {gif_path}")



def draw_ped_centers_cv(frame, centers, radius=2, color=(0, 200, 0)):
    for c in centers:
        cv2.circle(frame, (int(c[0]), int(c[1])), int(radius), color, -1)

def draw_robot_cv(frame, name, pos, color_bgr, radius=8, label_scale=0.55, label_thickness=2):
    cv2.circle(frame, (int(pos[0]), int(pos[1])), int(radius), color_bgr, -1)
    cv2.putText(
        frame, name,
        (int(pos[0]) + 10, int(pos[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(label_scale),
        color_bgr,
        int(label_thickness),
    )
