import numpy as np
from skimage.morphology import skeletonize

def skeleton_path(mask: np.ndarray) -> np.ndarray:
    skel = skeletonize(mask.astype(bool))
    return skel.astype(np.uint8)

def nearest_skeleton_pixel(skel: np.ndarray, p):
    ys, xs = np.where(skel > 0)
    pts = np.stack([ys, xs], axis=1)
    d2 = np.sum((pts - np.array(p)) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return int(pts[idx, 0]), int(pts[idx, 1])

def bfs_shortest_path(skel: np.ndarray, start, goal):
    from collections import deque
    H, W = skel.shape
    start = tuple(start)
    goal = tuple(goal)

    nbrs = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0), (1, 1)]

    q = deque([start])
    prev = {start: None}

    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for dy, dx in nbrs:
            ny, nx = cur[0] + dy, cur[1] + dx
            if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] > 0:
                nxt = (ny, nx)
                if nxt not in prev:
                    prev[nxt] = cur
                    q.append(nxt)

    if goal not in prev:
        raise RuntimeError("No skeleton path found between A and B.")

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path
