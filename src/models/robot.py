from dataclasses import dataclass
import numpy as np

@dataclass
class Robot:
    name: str
    pos: np.ndarray  
    vel: np.ndarray  
    direction: int   
    color_bgr: tuple
