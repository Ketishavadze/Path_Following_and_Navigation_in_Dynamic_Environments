from dataclasses import dataclass
import numpy as np

@dataclass
class Robot:
    name: str
    pos: np.ndarray  # shape (2,)
    vel: np.ndarray  # shape (2,)
    direction: int   # +1 right, -1 left
    color_bgr: tuple
