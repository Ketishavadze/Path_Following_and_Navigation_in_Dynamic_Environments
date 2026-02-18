import os
import cv2
import numpy as np

# ----------------------------
# Configuration
# ----------------------------
WIDTH = 640
HEIGHT = 480
FPS = 25
DURATION_SECONDS = 20
N_PEDESTRIANS_PER_FLOW = 25

RADIUS = 6
SPEED = 2.5  # pixels per frame
NOISE_STD = 0.3

OUTPUT_PATH = "data/video/pedestrians.mp4"


# ----------------------------
# Pedestrian Class
# ----------------------------
class Pedestrian:
    def __init__(self, x, y, vx, vy):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)

    def update(self):
        noise = np.random.normal(0, NOISE_STD, 2)
        self.pos += self.vel + noise

        # Wrap around horizontally
        if self.pos[0] < -20:
            self.pos[0] = WIDTH + 20
        if self.pos[0] > WIDTH + 20:
            self.pos[0] = -20

    def draw(self, frame):
        cv2.circle(frame,
                   (int(self.pos[0]), int(self.pos[1])),
                   RADIUS,
                   (0, 0, 0),
                   -1)


# ----------------------------
# Generate pedestrians
# ----------------------------
def create_pedestrians():
    pedestrians = []

    # Flow 1: Left → Right
    for _ in range(N_PEDESTRIANS_PER_FLOW):
        x = np.random.uniform(0, WIDTH)
        y = np.random.uniform(0, HEIGHT * 0.45)
        pedestrians.append(Pedestrian(x, y, SPEED, 0))

    # Flow 2: Right → Left
    for _ in range(N_PEDESTRIANS_PER_FLOW):
        x = np.random.uniform(0, WIDTH)
        y = np.random.uniform(HEIGHT * 0.55, HEIGHT)
        pedestrians.append(Pedestrian(x, y, -SPEED, 0))

    return pedestrians


# ----------------------------
# Main generation
# ----------------------------
def main():
    os.makedirs("data/video", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    pedestrians = create_pedestrians()
    total_frames = FPS * DURATION_SECONDS

    for frame_idx in range(total_frames):
        frame = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

        # Optional: draw separation line
        cv2.line(frame,
                 (0, HEIGHT // 2),
                 (WIDTH, HEIGHT // 2),
                 (200, 200, 200),
                 1)

        for ped in pedestrians:
            ped.update()
            ped.draw(frame)

        writer.write(frame)

        if frame_idx % 50 == 0:
            print(f"Generating frame {frame_idx}/{total_frames}")

    writer.release()
    print(f"\nVideo saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
