import numpy as np
import cv2


def create_rect(w, h, ppm, add_arrow=True):
    pix_w, pix_h = int(w*ppm), int(h*ppm)
    rect = np.ones((pix_w, pix_h))
    if add_arrow:
        cv2.arrowedLine(rect, (pix_h//2, pix_w//2), (pix_h//4*3, pix_w//2), 0, thickness=2, tipLength=0.5)
    return rect.astype(bool)
