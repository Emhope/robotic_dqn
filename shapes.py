import numpy as np


def create_rect(w, h, ppm):
    return np.ones((int(w*ppm), int(h*ppm)), dtype=bool)
