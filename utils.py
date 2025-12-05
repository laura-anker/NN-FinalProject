import cv2
import numpy as np

# Converts a LAB image to RGB
def lab_to_rgb(L_norm, ab_norm):
    # Undo the normalization
    L = L_norm * 255
    a = ab_norm[...,0] * 128.0 + 128.0
    b = ab_norm[...,1] * 128.0 + 128.0

    # Merge the three channels 
    lab = np.stack([L, a, b], axis=-1).astype("uint8")

    # Convert from LAB to RGB and normalize to [0, 1]
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb = np.clip(rgb / 255, 0, 1)
    return rgb